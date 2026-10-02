import json
import os
import numpy as np
from BBM92 import BBM92HardwareManager, BBM92ProtocolEngine
from CLASSIC_CHANNEL import BobClassicalChannel
from AUXILIARY import AuthKeyPool, ClassicalLink, PostProcessingError
from POSTPROCESS import run_postprocessing

# --- Configuração do pós-processamento (deve ser IGUAL à de ALICE.py) ---
EC_METHOD = os.environ.get("EC_METHOD", "cascade")
AUTH_KEY_FILE = os.environ.get("AUTH_KEY_FILE", "auth_key.json")
CAPTURE_S = float(os.environ.get("CAPTURE_S", "0.01"))       # duração da captura (s); IGUAL em Alice e Bob
CLASSIC_PORT = int(os.environ.get("CLASSIC_PORT", "65432"))   # ÚNICA porta TCP (sifting + pós-processamento); IGUAL em Alice e Bob

# Utilizar apenas no teste de hardware simulado
#import SIMLINK
#SIMLINK.install_simulated_hardware(pair_rate_hz=3e6, eve_intercept_resend=False, seed=42)

# ALICE_HOST=<IP DE ALICE> python BOB.py
ALICE_HOST = os.environ.get("ALICE_HOST", "").strip()

def main():
    if not ALICE_HOST:
        raise SystemExit(
            "Defina o IP de Alice na rede antes de rodar BOB.py, por "
            "exemplo:\n    ALICE_HOST=192.168.1.10 python BOB.py\n"
            "(rode ALICE.py primeiro -- ele imprime esse IP no início da execução)."
        )

    # 1. Canal clássico ÚNICO e persistente (porta única): é aberto aqui, usado
    #    no sifting e reaproveitado, pelo ClassicalLink, em todo o pós-processamento.
    with BobClassicalChannel(host=ALICE_HOST, port=CLASSIC_PORT) as channel:
        hw = BBM92HardwareManager(node_type='Bob')
        hw.calibrate_delays({5: 12500, 6: 12550, 7: 12500, 8: 12550})

        print("[BOB] Realizando capturas do SPAD...")
        timestamps, channels = hw.capture_stream(duration_s=CAPTURE_S)

        engine = BBM92ProtocolEngine(coincidence_window_ps=800)
        key, sifted_len = engine.process_time_tags(timestamps, channels)
        print(f"[BOB] Chave gerada (não peneirada): {sifted_len} bits adquiridos.")

        dados_bases_bob = {
            "ts": engine.valid_ts.tolist(),
            "bases": engine.bases.tolist()
        }

        # Troca de bases (sifting) pela conexão persistente
        print("[BOB] Trocando BASES (Sifting) via canal clássico...")
        msg_alice = channel.exchange(json.dumps(dados_bases_bob))
        dados_alice = json.loads(msg_alice)

        ts_alice = np.array(dados_alice["ts"])
        bases_alice = np.array(dados_alice["bases"])

        sifted_key, final_sifted_len = engine.perform_sifting(key, ts_alice, bases_alice, is_bob=True)
        if final_sifted_len == 0:
            print("[BOB][ERRO] Nenhuma coincidência foi encontrada entre Alice e "
                  "Bob. Verifique a conectividade de rede (host/porta) e se os "
                  "dois processos usam a mesma seed/duração de captura. "
                  "Encerrando sem gerar chave.")
            return
        print(f"Chave bruta: {key} bits")
        print(f"[BOB] Chave peneirada (sifting): {final_sifted_len} bits")

        # === PÓS-PROCESSAMENTO (independente de nó: POSTPROCESS.py) ===
        # PE (QBER) -> correção de erros -> privacy amplification -> autenticação,
        # tudo na MESMA conexão TCP do sifting.
        link = ClassicalLink(role="bob", channel=channel)
        # As bases/timestamps trocados no sifting também entram na transcrição autenticada
        # (mesma ordem de Alice: primeiro a mensagem de Alice, depois a de Bob):
        link.record(alice_msg=msg_alice, bob_msg=json.dumps(dados_bases_bob))
        pool = AuthKeyPool.load(AUTH_KEY_FILE)
        try:
            res = run_postprocessing(sifted_key, link, pool, ec_method=EC_METHOD)
        except PostProcessingError as exc:
            print(f"[BOB][ABORTADO] {type(exc).__name__}: {exc}")
            return

        print("\n=== RESULTADOS FINAIS EM BOB ===")
        print(f"Largura Total Peneirada: {final_sifted_len} bits")
        print(f"Taxa QBER (amostra): {res.qber * 100:.2f}%  (limite superior: {res.qber_upper * 100:.2f}%)")
        print(f"Correção de erros ({res.ec.method}): eficiência f = {res.ec.efficiency:.3f}, "
              f"{res.ec.leaked_bits} bits revelados")
        print(f"Chave Secreta Final: {res.final_key}")
        print(f"Largura da Chave Secreta (Final Key Size): {len(res.final_key)} bits")
        if len(res.final_key) == 0:
            print("[ALERTA] Chave final vazia (custo de tamanho finito / QBER alta). "
                  "Aumente a duração de captura.")
        else:
            print("[SUCESSO] Chave segura e autenticada gerada em Bob.")

if __name__ == "__main__":
    main()
