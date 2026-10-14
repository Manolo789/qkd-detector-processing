"""
QKD_NODE.py -- sessão QKD (BBM92) de um nó, Alice ou Bob, em uma única função,
e interface de linha de comando com saída JSON para gerenciadores de chave
externos (por exemplo, o qpass).

Este módulo NÃO implementa nenhuma etapa nova do protocolo: ele apenas
encadeia o que já existe no projeto

    BBM92.py         captura + process_time_tags + sifting
    CLASSIC_CHANNEL  canal TCP persistente
    POSTPROCESS.py   PE -> EC -> PA -> AUTH  (run_postprocessing)
    AUXILIARY.py     ClassicalLink, AuthKeyPool, pack_bits

e entrega o resultado como estrutura de dados (SessionResult), em vez de
imprimi-lo. ALICE.py e BOB.py passam a ser invólucros finos sobre
`run_session`, e o gerenciador de chaves usa o subcomando `session --json`.

Contrato da CLI (para o gerenciador de chaves)
----------------------------------------------
    python QKD_NODE.py session --role alice|bob [--peer-host IP] [--port N]
                               [--capture-s S] [--slot-ps PS] [--ec cascade|ldpc]
                               [--auth-key-file F] [--simulate [--sim-seed N] [--sim-eve]]
                               --json

  * stdout: UMA linha JSON (ver `SessionResult.to_json`); nada mais.
  * stderr: todos os logs do protocolo.
  * código de saída: 0 = chave gerada; 3 = chave vazia (tamanho finito);
    2 = protocolo abortado (QBER, verificação, autenticação, sifting);
    1 = erro de infraestrutura (rede, arquivo, parâmetros).
  * A chave só é emitida em stdout se stdout NÃO for um terminal (pipe), para
    não parar em tela/logs (documentation.txt, 16.2). Use --allow-tty para testes.

    python QKD_NODE.py auth-status --auth-key-file F     -> {"n","pos","remaining"}
    python QKD_NODE.py auth-init   --auth-key-file F --bits N [--force]
"""

import argparse
import contextlib
import hashlib
import json
import sys
import uuid
from dataclasses import dataclass, asdict
from typing import Dict, Optional

from AUXILIARY import (AuthKeyPool, ClassicalLink, PostProcessingError,
                       pack_bits)
from POSTPROCESS import run_postprocessing

PROTOCOL = "bbm92"

# Atrasos de calibração por nó (os mesmos usados antes em ALICE.py e BOB.py).
# setInputDelay SOMA o valor aos timestamps: atraso = -desvio (ver 6.5 e 16.4).
DEFAULT_DELAYS = {
    "alice": {1: 0, 2: -50, 3: 0, 4: -50},
    "bob": {5: 12500, 6: 12550, 7: 12500, 8: 12550},
}

EXIT_OK, EXIT_ERROR, EXIT_ABORTED, EXIT_EMPTY = 0, 1, 2, 3


@dataclass
class SessionConfig:
    role: str                          # 'alice' (servidor TCP) ou 'bob' (cliente)
    peer_host: str = ""                # IP de Alice; obrigatório em Bob
    port: int = 65432
    capture_s: float = 0.01
    slot_ps: int = 800
    ec_method: str = "cascade"
    auth_key_file: str = "auth_key.json"
    delays: Optional[Dict[int, int]] = None
    serial_number: Optional[str] = None
    simulate: bool = False             # instala o SIMLINK (sem valor criptográfico!)
    sim_seed: int = 42
    sim_eve: bool = False
    sim_pair_rate_hz: float = 3e6
    verbose: bool = True


@dataclass
class SessionResult:
    status: str                        # 'ok' | 'empty' | 'aborted' | 'error'
    role: str
    protocol: str = PROTOCOL
    key_id: Optional[str] = None
    key_bits: int = 0
    key_hex: Optional[str] = None      # pack_bits(final_key); zeros à direita até completar o byte
    raw_bits: int = 0                  # slots ocupados antes do sifting
    sifted_bits: int = 0
    qber: Optional[float] = None
    qber_upper: Optional[float] = None
    ec_method: Optional[str] = None
    ec_efficiency: Optional[float] = None
    leaked_bits: Optional[int] = None
    authenticated: bool = False
    replenished_bits: int = 0
    auth_bits_remaining: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    def to_json(self) -> str:
        d = asdict(self)
        # NaN (eficiência indefinida quando QBER=0) não é JSON válido
        if d["ec_efficiency"] is not None and d["ec_efficiency"] != d["ec_efficiency"]:
            d["ec_efficiency"] = None
        return json.dumps(d)

    @property
    def exit_code(self) -> int:
        return {"ok": EXIT_OK, "empty": EXIT_EMPTY, "aborted": EXIT_ABORTED}.get(
            self.status, EXIT_ERROR)


def key_id_from_transcript(link: ClassicalLink) -> str:
    """Identificador da chave, IGUAL nos dois nós sem nenhuma mensagem extra.

    A transcrição autenticada é idêntica em Alice e Bob (é exatamente o que a
    tag Wegman-Carter verifica) e é pública, então seu hash serve de ID público
    e único por sessão. Formato UUID (compatível com o key_ID do ETSI GS QKD 014).
    """
    digest = hashlib.sha256(b"qkd-key-id/v1" + link.transcript_bytes()).digest()
    return str(uuid.UUID(bytes=digest[:16]))


def _install_simulator(cfg: SessionConfig) -> None:
    import SIMLINK
    SIMLINK.install_simulated_hardware(pair_rate_hz=cfg.sim_pair_rate_hz,
                                       eve_intercept_resend=cfg.sim_eve,
                                       seed=cfg.sim_seed)


def run_session(cfg: SessionConfig) -> SessionResult:
    """Executa uma sessão completa de um nó e devolve o resultado.

    Resultados do PROTOCOLO (chave, chave vazia, aborto) voltam em
    SessionResult. Erros de infraestrutura (rede, auth_key.json ausente,
    parâmetros inválidos) são propagados como exceção.
    """
    role = cfg.role.lower()
    if role not in ("alice", "bob"):
        raise ValueError("role deve ser 'alice' ou 'bob'")
    tag = role.upper()
    say = print if cfg.verbose else (lambda *a, **k: None)

    if cfg.simulate:
        _install_simulator(cfg)
    # Importação tardia: BBM92 importa Swabian.TimeTagger (real ou simulado).
    from BBM92 import BBM92HardwareManager, BBM92ProtocolEngine
    from CLASSIC_CHANNEL import AliceClassicalChannel, BobClassicalChannel

    # Carrega o reservatório ANTES de abrir o canal: falha cedo se faltar
    # (antes isso só era descoberto depois do sifting, ver 16.7).
    pool = AuthKeyPool.load(cfg.auth_key_file)

    if role == "alice":
        channel_cm = AliceClassicalChannel(port=cfg.port)
    else:
        if not cfg.peer_host:
            raise ValueError("Bob precisa do IP de Alice (peer_host / ALICE_HOST).")
        channel_cm = BobClassicalChannel(host=cfg.peer_host, port=cfg.port)

    with channel_cm as channel:
        hw = BBM92HardwareManager(node_type=role.capitalize(),
                                  serial_number=cfg.serial_number)
        try:
            hw.calibrate_delays(cfg.delays or DEFAULT_DELAYS[role])
            say(f"[{tag}] Realizando capturas do SPAD...")
            timestamps, channels = hw.capture_stream(duration_s=cfg.capture_s)
        finally:
            hw.close()

        engine = BBM92ProtocolEngine(slot_ps=cfg.slot_ps)
        key, raw_len = engine.process_time_tags(timestamps, channels)
        say(f"[{tag}] Chave gerada (não peneirada): {raw_len} bits adquiridos.")

        # Sifting: só índices de slot + códigos de base (nunca timestamps finos)
        my_msg = json.dumps(engine.announcement())
        say(f"[{tag}] Trocando BASES (Sifting) via canal clássico...")
        peer_msg = channel.exchange(my_msg)
        try:
            sifted_key, sifted_len = engine.perform_sifting(
                key, json.loads(peer_msg), is_bob=(role == "bob"))
        except ValueError as exc:
            return SessionResult("aborted", role, raw_bits=raw_len,
                                 error_type="SiftingError", error_message=str(exc))
        if sifted_len == 0:
            return SessionResult("aborted", role, raw_bits=raw_len,
                                 error_type="NoCoincidences",
                                 error_message="Nenhuma coincidência entre Alice e Bob "
                                               "(seed/duração de captura/SLOT_PS iguais?).")
        say(f"[{tag}] Chave peneirada (sifting): {sifted_len} bits")

        # Pós-processamento: tudo na MESMA conexão; o sifting entra na transcrição
        link = ClassicalLink(role=role, channel=channel)
        if role == "alice":
            link.record(alice_msg=my_msg, bob_msg=peer_msg)
        else:
            link.record(alice_msg=peer_msg, bob_msg=my_msg)
        try:
            res = run_postprocessing(sifted_key, link, pool, ec_method=cfg.ec_method,
                                     verbose=cfg.verbose)
        except PostProcessingError as exc:
            return SessionResult("aborted", role, raw_bits=raw_len, sifted_bits=sifted_len,
                                 auth_bits_remaining=pool.remaining(),
                                 error_type=type(exc).__name__, error_message=str(exc))

    n_final = int(len(res.final_key))
    return SessionResult(
        status="ok" if n_final > 0 else "empty",
        role=role,
        key_id=key_id_from_transcript(link) if n_final > 0 else None,
        key_bits=n_final,
        key_hex=pack_bits(res.final_key) if n_final > 0 else None,
        raw_bits=int(raw_len),
        sifted_bits=int(sifted_len),
        qber=float(res.qber),
        qber_upper=float(res.qber_upper),
        ec_method=res.ec.method,
        ec_efficiency=float(res.ec.efficiency),
        leaked_bits=int(res.ec.leaked_bits),
        authenticated=bool(res.authenticated),
        replenished_bits=int(res.replenished_bits),
        auth_bits_remaining=pool.remaining(),
    )


def print_report(res: SessionResult, show_key: bool = True) -> None:
    """Relatório legível (o mesmo bloco 'RESULTADOS FINAIS' de ALICE.py/BOB.py)."""
    tag = res.role.upper()
    if res.status in ("aborted", "error"):
        print(f"[{tag}][ABORTADO] {res.error_type}: {res.error_message}")
        return
    print(f"\n=== RESULTADOS FINAIS EM {tag} ===")
    print(f"Largura Total Peneirada: {res.sifted_bits} bits")
    print(f"Taxa QBER (amostra): {res.qber * 100:.2f}%  "
          f"(limite superior: {res.qber_upper * 100:.2f}%)")
    print(f"Correção de erros ({res.ec_method}): eficiência f = {res.ec_efficiency:.3f}, "
          f"{res.leaked_bits} bits revelados")
    if show_key and res.key_hex:
        print(f"Chave Secreta Final (hex): {res.key_hex}")
    print(f"Largura da Chave Secreta (Final Key Size): {res.key_bits} bits")
    if res.status == "empty":
        print("[ALERTA] Chave final vazia (custo de tamanho finito / QBER alta). "
              "Aumente a duração de captura.")
    else:
        print(f"ID da chave: {res.key_id}")
        print(f"[SUCESSO] Chave segura e autenticada gerada em {res.role.capitalize()}.")


# =============================================================================
# CLI
# =============================================================================
def _emit(obj, code: int) -> int:
    sys.stdout.write((obj if isinstance(obj, str) else json.dumps(obj)) + "\n")
    sys.stdout.flush()
    return code


def _cmd_session(a) -> int:
    if a.json and sys.stdout.isatty() and not a.allow_tty:
        print("Recusado: --json escreve a chave em stdout, que é um terminal. "
              "Redirecione para um pipe ou use --allow-tty.", file=sys.stderr)
        return EXIT_ERROR
    cfg = SessionConfig(role=a.role, peer_host=a.peer_host, port=a.port,
                        capture_s=a.capture_s, slot_ps=a.slot_ps, ec_method=a.ec,
                        auth_key_file=a.auth_key_file,
                        delays={int(k): int(v) for k, v in json.loads(a.delays).items()}
                        if a.delays else None,
                        serial_number=a.serial, simulate=a.simulate,
                        sim_seed=a.sim_seed, sim_eve=a.sim_eve)
    if not a.json:
        res = run_session(cfg)
        print_report(res)
        return res.exit_code
    # Modo máquina: logs -> stderr, stdout reservado para UMA linha JSON
    try:
        with contextlib.redirect_stdout(sys.stderr):
            res = run_session(cfg)
    except Exception as exc:  # erro de infraestrutura -> JSON de erro
        res = SessionResult("error", cfg.role.lower(), error_type=type(exc).__name__,
                            error_message=str(exc))
    return _emit(res.to_json(), res.exit_code)


def _cmd_auth_status(a) -> int:
    try:
        pool = AuthKeyPool.load(a.auth_key_file)
    except FileNotFoundError:
        return _emit({"exists": False, "path": a.auth_key_file}, EXIT_ERROR)
    n = pool.remaining() + pool._pos
    return _emit({"exists": True, "path": a.auth_key_file, "n": n, "pos": pool._pos,
                  "remaining": pool.remaining()}, EXIT_OK)


def _cmd_auth_init(a) -> int:
    import os
    if os.path.exists(a.auth_key_file) and not a.force:
        print(f"{a.auth_key_file} já existe. Sobrescrever descarta o reservatório atual "
              f"(e dessincroniza do outro nó). Use --force se for intencional.",
              file=sys.stderr)
        return EXIT_ERROR
    AuthKeyPool.generate(a.bits, a.auth_key_file)
    return _emit({"path": a.auth_key_file, "n": a.bits}, EXIT_OK)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Nó QKD BBM92 (Alice/Bob) com saída para gerenciador de chaves.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("session", help="executa uma sessão QKD completa")
    s.add_argument("--role", required=True, choices=["alice", "bob"])
    s.add_argument("--peer-host", default="", help="IP de Alice (obrigatório em Bob)")
    s.add_argument("--port", type=int, default=65432)
    s.add_argument("--capture-s", type=float, default=0.01)
    s.add_argument("--slot-ps", type=int, default=800)
    s.add_argument("--ec", default="cascade", choices=["cascade", "ldpc"])
    s.add_argument("--auth-key-file", default="auth_key.json")
    s.add_argument("--delays", default=None, help='JSON {"canal": atraso_ps, ...}')
    s.add_argument("--serial", default=None, help="número de série do Time Tagger")
    s.add_argument("--simulate", action="store_true", help="usa o SIMLINK (sem valor criptográfico)")
    s.add_argument("--sim-seed", type=int, default=42)
    s.add_argument("--sim-eve", action="store_true")
    s.add_argument("--json", action="store_true", help="saída JSON em stdout, logs em stderr")
    s.add_argument("--allow-tty", action="store_true")
    s.set_defaults(fn=_cmd_session)

    st = sub.add_parser("auth-status", help="estado do reservatório de autenticação")
    st.add_argument("--auth-key-file", default="auth_key.json")
    st.set_defaults(fn=_cmd_auth_status)

    ai = sub.add_parser("auth-init", help="gera o reservatório de autenticação")
    ai.add_argument("--auth-key-file", default="auth_key.json")
    ai.add_argument("--bits", type=int, default=8192)
    ai.add_argument("--force", action="store_true")
    ai.set_defaults(fn=_cmd_auth_init)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
