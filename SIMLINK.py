"""
SIMLINK.py
==========

Backend de hardware SIMULADO para o BBM92.py, dispensando TimeTagger, fonte de
pares emaranhados, transmissor e receptores reais.

Ideia
-----
`BBM92.py` importa o driver real com `import Swabian.TimeTagger as TT` e usa
apenas 4 chamadas desse driver: `createTimeTagger()`, `tagger.setInputDelay()`,
`tagger.sync()` e `TT.TimeTagStream(...)` (com `.start()`, `.stop()` e
`.getEvents()`). Este arquivo implementa um módulo Python "Swabian.TimeTagger"
simulado, com essa mesma interface, e o registra em `sys.modules` ANTES de
`BBM92.py` ser executado. Assim, quando `BBM92.py` faz o `import`, o Python
encontra nosso backend simulado em vez do driver real — o arquivo BBM92.py
roda sem nenhuma modificação, linha por linha, inclusive o bloco
`if __name__ == "__main__":`.

Por trás da interface simulada, `SimulatedEntangledLink` gera, via Monte Carlo,
marcas de tempo (timestamps) e canais fisicamente plausíveis para um enlace
BBM92 completo: fonte de pares emaranhados (EPS) por conversão paramétrica
descendente (SPDC), escolha passiva e aleatória de base em cada módulo de
análise de polarização (PAM), estado singleto |Psi-> (anticorrelacionado),
perdas/eficiência de detecção em cada lado, jitter temporal dos SPADs e
contagens escuras (dark counts) — seguindo a arquitetura e os valores
medidos descritos na dissertação A. Krzic, "Development and
Characterization of an Entanglement-based Free-Space QKD System", cap. 2 e 3):
    - arquitetura EPS + PAM (4 canais H/V/D/A) + SPADs + time tagger + Rb clock
      (Fig. 3.1 / Seção 3.1.1);
    - resolução temporal dos SPADs de ~350 ps FWHM e eficiência quântica >60%
      a 810 nm (Seção 3.1.5);
    - taxa de contagens escuras medida: 1742 cps (Alice) e 776 cps (Bob),
      somadas sobre os 4 detectores (Seção 3.2.4);
    - janela de coincidência de 800 ps usada nos experimentos (Seção 3.4.1) —
      o mesmo valor já usado no bloco de exemplo de BBM92.py;
    - QBER experimental tipicamente <2% (noite) a <3.4% (dia) (Seção 3.3.3);
    - ataque de interceptação-reenvio (intercept-resend) introduzindo ~25% de
      erro na base compatível (Seção 2.1.1, Fig. 2.3), disponível aqui como
      opção `--eve` para demonstrar a detecção de espionagem via QBER.

IMPORTANTE — o que este simulador NÃO é
----------------------------------------
Ele não modela óptica de feixe, turbulência atmosférica, filtragem espectral/
espacial, nem o link de rádio clássico de pós-processamento — isso está muito
além do escopo dessa simulação e um trabalho nesse sentido já foi realizado 
no projeto https://github.com/Manolo789/SeQUeNCe-QKDprotocols. O que é
modelado com cuidado é exatamente o que BBM92.py consome: os timestamps e os
canais de detecção brutos, com estatística de ruído e de correlação
quanticamente plausível.

Uso
---
    python SIMLINK.py                       # roda BBM92.py normalmente (sem Eve)
    python SIMLINK.py --eve                 # com espiã fazendo intercept-resend
    python SIMLINK.py --pair-rate 5e6 --seed 42
    python SIMLINK.py --bbm92 /caminho/BBM92.py

Ou, de dentro de outro script Python:

    import SIMLINK
    SIMLINK.install_simulated_hardware(pair_rate_hz=3e6, eve_intercept_resend=True)
    import Swabian.TimeTagger as TT   # agora resolve para o backend simulado
    ...

Nota sobre calibração de canais
--------------------------------
O bloco `__main__` de BBM92.py só chama
    hw.calibrate_delays({1: 0, 2: 50, 5: 12500, 6: 12550})
ou seja, só calibra os canais H/V (base Z). Este simulador dá aos canais D/A
(base X, canais 3, 4, 7 e 8) o MESMO desalinhamento intrínseco de cabeamento
que H/V. Rodando o exemplo tal como está, portanto, você deve observar: QBER
baixo e boa contagem de coincidências sifted na base Z, mas coincidências
quase inexistentes na base X (o desvio de ~12.5 ns nos canais 7/8 cai bem
fora da janela de 800 ps). Isso ilustra, na prática, por que a etapa de
calibração de atraso (Seção 3.2.4 da dissertação) é indispensável — para
corrigir, basta estender a calibração:
    hw.calibrate_delays({1: 0, 2: 50, 3: 0, 4: 50, 5: 12500, 6: 12550, 7: 12500, 8: 12550})
"""

from __future__ import annotations

import argparse
import runpy
import sys
import time as _walltime
import types

import numpy as np

# =============================================================================
# 0. Interceptação de time.sleep para permitir reprodutibilidade ENTRE
#    PROCESSOS (correção de bug crítico)
# =============================================================================
# ALICE.py e BOB.py rodam em computadores separados. Cada processo cria seu
# próprio `_DEFAULT_LINK` com a MESMA seed (42) -- essa é claramente a forma
# como o autor original pretendia fazer os dois lados "enxergarem" o mesmo
# enlace emaranhado simulado sem nenhum estado realmente compartilhado entre
# as máquinas: se as duas instâncias de SimulatedEntangledLink recebem a
# mesma seed E o mesmo argumento `duration_s` em generate(), os dois
# processos sorteiam, de forma totalmente determinística, os MESMOS pares
# (mesmos tempos de criação, mesmas bases, mesmos bits) -- e cada lado só
# fica com os canais que lhe pertencem.
#
# O problema: BBM92HardwareManager.capture_stream() mede a duração pelo
# relógio de parede (time.sleep(duration_s) seguido de
# tempo_final - tempo_inicial), e essa duração MEDIDA nunca é exatamente
# igual entre dois computadores (jitter do agendador do SO, carga da
# máquina, etc.). Uma diferença de microssegundos já é suficiente para que
# rng.poisson(pair_rate_hz * duration_s) sorteie um número de pares
# diferente em cada lado, o que diverge TODA a sequência aleatória seguinte.
# Resultado: Alice e Bob geram fótons simulados totalmente descorrelacionados
# entre si, a etapa de coincidência (perform_sifting) não encontra pares
# reais, e o QBER calculado sobre um conjunto vazio é reportado como 0%
# ("sucesso" falso) mesmo com uma chave final de 0 bits.
#
# Correção (sem tocar em BBM92.py, que deve continuar rodando linha por linha
# como com o driver real): interceptamos a chamada `time.sleep(duration_s)`
# feita dentro de capture_stream() e guardamos o valor literal pedido;
# TimeTagStream.stop() usa esse valor exato em vez do tempo medido. Como
# ALICE.py e BOB.py pedem literalmente o mesmo `duration_s` (ex.: 0.001),
# essa correção restaura a reprodutibilidade bit-a-bit entre os dois
# processos.
# NOTA: `_walltime` É o módulo `time` (importado acima como `import time as
# _walltime`). BBM92.py faz `import time; time.sleep(duration_s)` dentro de
# capture_stream() -- como módulos Python são singletons em sys.modules,
# `_walltime` e o `time` que BBM92.py importa são o MESMO objeto de módulo,
# então corrigir `_walltime.sleep` também corrige o `time.sleep` que
# BBM92.py chama, sem precisar tocar em BBM92.py.
_real_sleep = _walltime.sleep
_last_requested_sleep_s: float | None = None


def _recording_sleep(seconds: float) -> None:
    """Substitui time.sleep: grava a duração pedida e realmente dorme por ela
    (mantém o comportamento/tempo real do programa, só adiciona o registro)."""
    global _last_requested_sleep_s
    _last_requested_sleep_s = float(seconds)
    _real_sleep(seconds)


_walltime.sleep = _recording_sleep


# =============================================================================
# 1. Constantes de canal — devem espelhar exatamente BBM92.py
# =============================================================================
ALICE_CHANNELS = {"H": 1, "V": 2, "D": 3, "A": 4}
BOB_CHANNELS = {"H": 5, "V": 6, "D": 7, "A": 8}

# (base, bit) por canal — 0 = base Z (H/V), 1 = base X (D/A)
BASE_Z, BASE_X = 0, 1
CHANNEL_MAP = {
    1: (BASE_Z, 0), 2: (BASE_Z, 1), 3: (BASE_X, 0), 4: (BASE_X, 1),
    5: (BASE_Z, 0), 6: (BASE_Z, 1), 7: (BASE_X, 0), 8: (BASE_X, 1),
}


# =============================================================================
# 2. Modelo físico do enlace emaranhado simulado
# =============================================================================
class SimulatedEntangledLink:
    """
    Gera, em lote, os timestamps/canais que um EPS + 2 PAMs + 8 SPADs + 1
    time tagger compartilhado teriam produzido durante `duration_s` segundos.

    Parâmetros (valores padrão ancorados na dissertação anexa)
    ------------------------------------------------------------------
    pair_rate_hz            taxa intrínseca de geração de pares do EPS (pares/s).
    eta_alice, eta_bob      eficiência total de detecção (ótica + SPAD) de cada
                            lado; determina, junto com pair_rate_hz, a taxa de
                            coincidências verdadeiras (~pair_rate*eta_a*eta_b).
    dark_counts_*_cps       taxa de contagens escuras somada sobre os 4
                            detectores de cada lado (medido: 1742/776 cps).
    detector_jitter_fwhm_ps resolução temporal (FWHM) de cada SPAD (~350 ps).
    intrinsic_qber          erro residual do sistema mesmo sem espiã (imperfeição
                            de fonte/óptica/extinção de polarização); tipicamente
                            <2% à noite, <3.4% de dia, na dissertação.
    eve_intercept_resend    se True, simula uma Eve fazendo interceptação e
                            reenvio em cada fóton enviado a Bob (Seção 2.1.1).
    seed                    semente do gerador aleatório (reprodutibilidade).
    """

    # Desalinhamentos "verdadeiros" (não calibrados) de cabeamento/eletrônica
    # de cada canal, em picossegundos — o que calibrate_delays() deve corrigir.
    # Alice: pequeno descasamento interno entre os dois detectores de cada par
    # (H/D vs V/A). Bob: grande atraso sistemático (~12.5 ns) representando o
    # cabo/fibra mais longa até o time tagger compartilhado, replicado
    # igualmente nos canais H/V e D/A.
    _OFFSET_PS = {
        1: 0, 2: 50, 3: 0, 4: 50,                      # Alice
        5: -12_500, 6: -12_550, 7: -12_500, 8: -12_550,  # Bob
    }
    _EPOCH_BASELINE_PS = 20_000  # margem para manter todos os timestamps >= 0

    def __init__(
        self,
        pair_rate_hz: float = 2.0e6,
        eta_alice: float = 0.05,
        eta_bob: float = 0.04,
        dark_counts_alice_cps: float = 1742.0,
        dark_counts_bob_cps: float = 776.0,
        detector_jitter_fwhm_ps: float = 350.0,
        intrinsic_qber: float = 0.02,
        eve_intercept_resend: bool = False,
        seed: int | None = None,
    ):
        self.pair_rate_hz = float(pair_rate_hz)
        self.eta_alice = float(eta_alice)
        self.eta_bob = float(eta_bob)
        self.dark_alice_per_ch = dark_counts_alice_cps / 4.0
        self.dark_bob_per_ch = dark_counts_bob_cps / 4.0
        # FWHM -> sigma de uma Gaussiana
        self.jitter_sigma_ps = detector_jitter_fwhm_ps / 2.3548
        self.intrinsic_qber = float(intrinsic_qber)
        self.eve = bool(eve_intercept_resend)
        self.rng = np.random.default_rng(seed)

        # LUT canal -> atraso intrínseco, indexável diretamente por array de canais (1..8)
        self._offset_lut = np.zeros(9, dtype=np.float64)
        for ch, off in self._OFFSET_PS.items():
            self._offset_lut[ch] = off

    def generate(self, duration_s: float):
        """Retorna (timestamps_ps int64, channels int64) ordenados por tempo."""
        if duration_s <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        rng = self.rng
        window_ps = duration_s * 1e12

        # -- 1. Geração de pares emaranhados: processo de Poisson no tempo -----
        n_pairs = int(rng.poisson(self.pair_rate_hz * duration_s))
        t_create = np.sort(rng.uniform(0.0, window_ps, size=n_pairs))

        # -- 2. Escolha passiva e aleatória de base em cada PAM (50/50) --------
        alice_basis = rng.integers(0, 2, size=n_pairs)
        bob_basis = rng.integers(0, 2, size=n_pairs)

        # -- 3. Resultado individualmente aleatório de Alice --------------------
        alice_bit = rng.integers(0, 2, size=n_pairs)

        # -- 4. Resultado de Bob: estado singlete |Psi-> => anticorrelação ------
        #       nas bases compatíveis; aleatório e descorrelacionado nas
        #       bases incompatíveis (mutuamente não enviesadas).
        same_basis = alice_basis == bob_basis
        no_eve_ok = rng.random(n_pairs) >= self.intrinsic_qber
        bob_bit_ideal = np.where(no_eve_ok, 1 - alice_bit, alice_bit)
        bob_bit_random = rng.integers(0, 2, size=n_pairs)
        bob_bit = np.where(same_basis, bob_bit_ideal, bob_bit_random)

        # -- 4b. Espiã opcional: interceptação e reenvio (Seção 2.1.1) -----------
        if self.eve:
            eve_basis = rng.integers(0, 2, size=n_pairs)
            eve_matches_alice = eve_basis == alice_basis
            eve_ok = rng.random(n_pairs) >= self.intrinsic_qber
            eve_bit_if_match = np.where(eve_ok, 1 - alice_bit, alice_bit)
            eve_bit_if_mismatch = rng.integers(0, 2, size=n_pairs)
            eve_bit = np.where(eve_matches_alice, eve_bit_if_match, eve_bit_if_mismatch)

            # Eve reenvia um fóton de preparação-e-medida (não emaranhado) na
            # SUA base/bit. Diferente do par original (singlete, anticorrelato),
            # aqui é um único fóton preparado: se Bob mede na MESMA base em que
            # Eve preparou, ele obtém o MESMO bit dela (correlacionado, não
            # anticorrelacionado); se mede em base diferente, o resultado é
            # aleatório (mede-se em base errada de um estado bem definido).
            
            bob_matches_eve = bob_basis == eve_basis
            resend_ok = rng.random(n_pairs) >= self.intrinsic_qber
            bob_bit_if_match_eve = np.where(resend_ok, eve_bit, 1 - eve_bit)
            bob_bit_if_mismatch_eve = rng.integers(0, 2, size=n_pairs)
            bob_bit_after_eve = np.where(bob_matches_eve, bob_bit_if_match_eve, bob_bit_if_mismatch_eve)

            bob_bit = np.where(same_basis, bob_bit_after_eve, bob_bit)

        # -- 5. Mapeia (base, bit) -> número de canal físico ---------------------
        alice_ch = np.where(alice_basis == BASE_Z,
                             np.where(alice_bit == 0, 1, 2),
                             np.where(alice_bit == 0, 3, 4))
        bob_ch = np.where(bob_basis == BASE_Z,
                           np.where(bob_bit == 0, 5, 6),
                           np.where(bob_bit == 0, 7, 8))

        # -- 6. Eficiência de detecção: heraldo independente em cada lado --------
        alice_seen = rng.random(n_pairs) < self.eta_alice
        bob_seen = rng.random(n_pairs) < self.eta_bob

        # -- 7. Timestamps brutos: criação + desalinhamento de hardware + jitter -
        base = self._EPOCH_BASELINE_PS
        alice_ts_all = base + t_create + self._offset_lut[alice_ch] + rng.normal(0.0, self.jitter_sigma_ps, n_pairs)
        bob_ts_all = base + t_create + self._offset_lut[bob_ch] + rng.normal(0.0, self.jitter_sigma_ps, n_pairs)

        alice_ts = alice_ts_all[alice_seen]
        alice_c = alice_ch[alice_seen]
        bob_ts = bob_ts_all[bob_seen]
        bob_c = bob_ch[bob_seen]

        # -- 8. Contagens escuras: ruído de Poisson descorrelacionado, por canal -
        alice_dc_ts, alice_dc_ch = [], []
        for ch in (1, 2, 3, 4):
            n_dc = int(rng.poisson(self.dark_alice_per_ch * duration_s))
            alice_dc_ts.append(base + rng.uniform(0.0, window_ps, n_dc) + self._offset_lut[ch])
            alice_dc_ch.append(np.full(n_dc, ch))

        bob_dc_ts, bob_dc_ch = [], []
        for ch in (5, 6, 7, 8):
            n_dc = int(rng.poisson(self.dark_bob_per_ch * duration_s))
            bob_dc_ts.append(base + rng.uniform(0.0, window_ps, n_dc) + self._offset_lut[ch])
            bob_dc_ch.append(np.full(n_dc, ch))

        timestamps = np.concatenate([alice_ts, bob_ts, *alice_dc_ts, *bob_dc_ts])
        channels = np.concatenate([alice_c, bob_c, *alice_dc_ch, *bob_dc_ch]).astype(np.int64)

        order = np.argsort(timestamps, kind="stable")
        return np.rint(timestamps[order]).astype(np.int64), channels[order]


# Link "físico" compartilhado por padrão entre todos os createTimeTagger() —
# representa o único EPS existente no bancada simulada.
_DEFAULT_LINK = SimulatedEntangledLink()


def configure_link(**kwargs) -> SimulatedEntangledLink:
    """Reconfigura o enlace simulado padrão (taxa de pares, eficiências, Eve, ...)."""
    global _DEFAULT_LINK
    _DEFAULT_LINK = SimulatedEntangledLink(**kwargs)
    return _DEFAULT_LINK


# =============================================================================
# 3. Backend simulado "Swabian.TimeTagger" — mesma interface usada por BBM92.py
# =============================================================================
class _SimulatedEvents:
    """Espelha o objeto retornado por stream.getEvents() na API real."""

    def __init__(self, timestamps: np.ndarray, channels: np.ndarray):
        self._timestamps = timestamps
        self._channels = channels

    def getTimestamps(self) -> np.ndarray:
        return self._timestamps

    def getChannels(self) -> np.ndarray:
        return self._channels


class SimulatedTimeTagger:
    """Substitui TT.createTimeTagger(...) — mesma interface mínima usada."""

    def __init__(self, serial_number: str | None = None, link: SimulatedEntangledLink | None = None):
        self.serial_number = serial_number or "SIM-TT-8CH-0001"
        self._input_delays: dict[int, int] = {}
        self._link = link if link is not None else _DEFAULT_LINK

    def setInputDelay(self, channel: int, delay_ps: int) -> None:
        """Compensa (aditivamente) o atraso de cabeamento/eletrônica de um canal."""
        self._input_delays[int(channel)] = int(delay_ps)

    def sync(self) -> None:
        """No hardware real, alinha domínios de clock internos; aqui é um no-op."""
        return None

    def get_input_delays(self) -> dict:
        return dict(self._input_delays)


class SimulatedTimeTagStream:
    """Substitui TT.TimeTagStream(tagger, buffer_size, channels)."""

    def __init__(self, tagger: SimulatedTimeTagger, buffer_size: int = 10_000_000, channels=None):
        self.tagger = tagger
        self.buffer_size = int(buffer_size)
        self.channels = list(channels) if channels else []
        self._running = False
        self._t0_wall = None
        self._duration_s = 0.0

    def start(self) -> None:
        self._running = True
        self._t0_wall = _walltime.time()
        global _last_requested_sleep_s
        _last_requested_sleep_s = None

    def stop(self) -> None:
        if not self._running:
            raise RuntimeError("TimeTagStream.stop() chamado sem um start() correspondente.")
        medido = _walltime.time() - self._t0_wall
        if _last_requested_sleep_s is not None:
            # Usa a duração literal pedida via time.sleep(...) entre start()/
            # stop() -- garante que dois processos independentes (Alice e
            # Bob, em computadores diferentes) com a mesma seed sorteiem
            # exatamente os mesmos pares simulados. Ver nota no topo do
            # arquivo sobre por que o tempo medido não serve para isso.
            self._duration_s = _last_requested_sleep_s
        else:
            # Nenhum time.sleep() foi observado entre start() e stop() (uso
            # fora do padrão de BBM92HardwareManager.capture_stream) -- cai de
            # volta para o tempo medido.
            self._duration_s = medido
        self._running = False

    def getEvents(self) -> _SimulatedEvents:
        if self._running:
            raise RuntimeError("Chame stop() antes de ler os eventos do buffer.")

        timestamps, channels = self.tagger._link.generate(self._duration_s)

        # Mantém somente os canais configurados no stream, como no hardware real
        if self.channels:
            mask = np.isin(channels, self.channels)
            timestamps, channels = timestamps[mask], channels[mask]

        # Aplica a calibração de atraso definida via tagger.setInputDelay(...)
        delays = self.tagger.get_input_delays()
        if delays:
            timestamps = timestamps.copy()
            for ch, delay_ps in delays.items():
                timestamps[channels == ch] += delay_ps

        # Respeita o tamanho de buffer configurado, como no hardware real
        if len(timestamps) > self.buffer_size:
            timestamps = timestamps[: self.buffer_size]
            channels = channels[: self.buffer_size]

        order = np.argsort(timestamps, kind="stable")
        return _SimulatedEvents(timestamps[order], channels[order])


def createTimeTagger(serial_number: str | None = None) -> SimulatedTimeTagger:
    """Substitui TT.createTimeTagger(serial_number)."""
    return SimulatedTimeTagger(serial_number=serial_number, link=_DEFAULT_LINK)


def install_simulated_hardware(**link_kwargs) -> None:
    """
    Registra o módulo simulado 'Swabian.TimeTagger' em sys.modules.

    Deve ser chamado ANTES de qualquer `import Swabian.TimeTagger`
    (em particular, antes de importar ou executar BBM92.py).
    """
    if link_kwargs:
        configure_link(**link_kwargs)

    swabian_pkg = types.ModuleType("Swabian")
    tt_mod = types.ModuleType("Swabian.TimeTagger")
    tt_mod.createTimeTagger = createTimeTagger
    tt_mod.TimeTagStream = SimulatedTimeTagStream
    tt_mod.configure_link = configure_link
    tt_mod.SimulatedEntangledLink = SimulatedEntangledLink

    swabian_pkg.TimeTagger = tt_mod
    sys.modules["Swabian"] = swabian_pkg
    sys.modules["Swabian.TimeTagger"] = tt_mod


def run_bbm92(bbm92_path: str = "BBM92.py", **link_kwargs) -> None:
    """Instala o hardware simulado e executa BBM92.py sem modificá-lo."""
    install_simulated_hardware(**link_kwargs)
    runpy.run_path(bbm92_path, run_name="__main__")


# =============================================================================
# 4. Execução via linha de comando
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backend de hardware simulado (fonte emaranhada + time tagger) para BBM92.py"
    )
    parser.add_argument("--bbm92", default="BBM92.py", help="caminho para BBM92.py")
    parser.add_argument("--pair-rate", type=float, default=2.0e6, help="taxa de geração de pares do EPS [pares/s]")
    parser.add_argument("--eta-alice", type=float, default=0.05, help="eficiência total de detecção em Alice")
    parser.add_argument("--eta-bob", type=float, default=0.04, help="eficiência total de detecção em Bob")
    parser.add_argument("--dark-alice", type=float, default=1742.0, help="cps de ruído escuro somado (4 detectores) em Alice")
    parser.add_argument("--dark-bob", type=float, default=776.0, help="cps de ruído escuro somado (4 detectores) em Bob")
    parser.add_argument("--jitter-fwhm", type=float, default=350.0, help="FWHM do jitter de cada SPAD [ps]")
    parser.add_argument("--intrinsic-qber", type=float, default=0.02, help="QBER residual do sistema, sem espionagem")
    parser.add_argument("--eve", action="store_true", help="simula uma espiã com ataque intercept-resend")
    parser.add_argument("--seed", type=int, default=None, help="semente do gerador aleatório")
    args = parser.parse_args()

    run_bbm92(
        args.bbm92,
        pair_rate_hz=args.pair_rate,
        eta_alice=args.eta_alice,
        eta_bob=args.eta_bob,
        dark_counts_alice_cps=args.dark_alice,
        dark_counts_bob_cps=args.dark_bob,
        detector_jitter_fwhm_ps=args.jitter_fwhm,
        intrinsic_qber=args.intrinsic_qber,
        eve_intercept_resend=args.eve,
        seed=args.seed,
    )
