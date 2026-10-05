"""
SIMLINK.py
==========

Backend de hardware SIMULADO para o BBM92.py, dispensando TimeTagger, fonte de
pares emaranhados, transmissor e receptores reais.

Ideia
-----
`BBM92.py` importa o driver real com `import Swabian.TimeTagger as TT` e usa
estas chamadas desse driver: `createTimeTagger()`, `freeTimeTagger()`,
`tagger.setInputDelay()`, `tagger.sync()`, `tagger.getOverflowsAndClear()` e
`TT.TimeTagStream(tagger, n_max_events, channels)` (com `.startFor()`,
`.waitUntilFinished()`, `.stop()` e `.getData()`, que retorna um
`TimeTagStreamBuffer` com `getTimestamps()`, `getChannels()`,
`getEventTypes()`, `getMissedEvents()`, `size` e `hasOverflows`). Este arquivo implementa um módulo Python "Swabian.TimeTagger"
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
Cada nó calibra apenas os SEUS canais (BBM92HardwareManager ignora, com aviso,
canais de outro nó): ALICE.py chama
    hw.calibrate_delays({1: 0, 2: 50, 3: 0, 4: 50})
e BOB.py chama
    hw.calibrate_delays({5: 12500, 6: 12550, 7: 12500, 8: 12550})
Este simulador dá aos canais D/A (base X, canais 3, 4, 7 e 8) o MESMO
desalinhamento intrínseco de cabeamento que H/V. Se algum desses canais não for
calibrado, quase não haverá coincidências naquela base (um desvio de ~12.5 ns
cai bem fora da janela de 800 ps) — o que ilustra por que a etapa de
calibração de atraso (Seção 3.2.4 da dissertação) é indispensável.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import time as _walltime
import types

import numpy as np

# NOTA sobre reprodutibilidade entre processos
# --------------------------------------------
# ALICE.py e BOB.py rodam em computadores separados e cada processo cria seu
# próprio `_DEFAULT_LINK` com a MESMA seed. Se os dois pedirem a MESMA duração
# em generate(), sorteiam exatamente os mesmos pares (cada lado fica só com os
# seus canais). BBM92.py agora captura com `stream.startFor(duração_em_ps)`, que
# informa a duração EXATA pedida (como no driver real, onde ela é medida no
# tempo do fluxo de dados). Por isso não é mais necessário interceptar
# `time.sleep` globalmente, como era feito quando a captura usava sleep()+stop().

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
# 3. Backend simulado "Swabian.TimeTagger" — espelha a API real (manual 2.22.6)
# =============================================================================
CHANNEL_UNUSED = -134217728  # igual a TimeTagger.CHANNEL_UNUSED


class TagType(int):
    """Espelha TimeTagger.TagType (valores inteiros iguais aos do driver real)."""


TagType.TimeTag = TagType(0)
TagType.Error = TagType(1)
TagType.OverflowBegin = TagType(2)
TagType.OverflowEnd = TagType(3)
TagType.MissedEvents = TagType(4)


class SimulatedTimeTagStreamBuffer:
    """Espelha TimeTagStreamBuffer, retornado por TimeTagStream.getData()."""

    def __init__(self, timestamps: np.ndarray, channels: np.ndarray,
                 t_start: int = 0, t_get_data: int = 0):
        self._timestamps = np.asarray(timestamps, dtype=np.int64)
        self._channels = np.asarray(channels, dtype=np.int32)
        self.size = int(len(self._timestamps))   # atributo, como no driver real
        self.hasOverflows = False
        self.tStart = int(t_start)
        self.tGetData = int(t_get_data)

    def getTimestamps(self) -> np.ndarray:
        return self._timestamps

    def getChannels(self) -> np.ndarray:
        return self._channels

    def getEventTypes(self) -> np.ndarray:
        # A simulação só produz detecções reais (TagType.TimeTag).
        return np.zeros(self.size, dtype=np.uint8)

    def getMissedEvents(self) -> np.ndarray:
        return np.zeros(self.size, dtype=np.uint16)


class SimulatedTimeTagger:
    """Substitui TT.createTimeTagger(...): mesma interface usada pelo projeto."""

    def __init__(self, serial: str = "", link: SimulatedEntangledLink | None = None):
        self.serial_number = serial or "SIM-TT-8CH-0001"
        self._input_delays: dict[int, int] = {}
        self._link = link if link is not None else _DEFAULT_LINK
        self._freed = False

    def setInputDelay(self, channel: int, delay: int) -> None:
        """Atraso artificial (ps, int) somado aos timestamps do canal."""
        self._input_delays[int(channel)] = int(delay)

    def getInputDelay(self, channel: int) -> int:
        return self._input_delays.get(int(channel), 0)

    def sync(self, timeout: int = -1) -> bool:
        """No hardware real, aguarda o pipeline/FPGA; aqui retorna True imediatamente."""
        return True

    def getOverflows(self) -> int:
        return 0

    def getOverflowsAndClear(self) -> int:
        return 0

    def clearOverflows(self) -> None:
        return None

    def get_input_delays(self) -> dict:
        return dict(self._input_delays)


class SimulatedTimeTagStream:
    """
    Substitui TT.TimeTagStream(tagger, n_max_events, channels).

    Como no driver real, a medição começa a acumular dados já na criação do
    objeto; getData() devolve o que foi acumulado desde a última chamada e
    esvazia o buffer.
    """

    def __init__(self, tagger: SimulatedTimeTagger, n_max_events: int, channels):
        self.tagger = tagger
        self.n_max_events = int(n_max_events)
        self.channels = list(channels)
        self._running = True
        self._t0_wall = _walltime.time()
        self._pending_s = 0.0     # duração acumulada ainda não lida por getData()
        self._elapsed_ps = 0      # tempo total do fluxo, para tStart/tGetData
        self._last_get_ps = 0

    # -- controle da medição ---------------------------------------------
    def start(self) -> None:
        if not self._running:
            self._running = True
            self._t0_wall = _walltime.time()

    def startFor(self, capture_duration: int, clear: bool = True) -> None:
        """Captura por `capture_duration` ps. A duração é EXATA (determinística)."""
        if clear:
            self.clear()
        self._pending_s += float(capture_duration) / 1e12
        self._elapsed_ps += int(capture_duration)
        self._running = False

    def stop(self) -> None:
        if self._running:
            self._accumulate_wall()
            self._running = False

    def clear(self) -> None:
        self._pending_s = 0.0

    def isRunning(self) -> bool:
        return self._running

    def waitUntilFinished(self, timeout: int = -1) -> bool:
        # A simulação conclui startFor() instantaneamente.
        return not self._running

    def getCounts(self) -> int:
        return len(self._build()[0])

    # -- leitura -------------------------------------------------------------
    def _accumulate_wall(self) -> None:
        medido = _walltime.time() - self._t0_wall
        self._pending_s += medido
        self._elapsed_ps += int(medido * 1e12)
        self._t0_wall = _walltime.time()

    def _build(self):
        if self._pending_s <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        timestamps, channels = self.tagger._link.generate(self._pending_s)

        # Mantém somente os canais configurados no stream, como no hardware real
        if self.channels:
            mask = np.isin(channels, self.channels)
            timestamps, channels = timestamps[mask], channels[mask]

        # Aplica a calibração definida via tagger.setInputDelay(...)
        delays = self.tagger.get_input_delays()
        if delays:
            timestamps = timestamps.copy()
            for ch, delay_ps in delays.items():
                timestamps[channels == ch] += delay_ps

        # O hardware entrega as tags em ordem temporal; só então o buffer corta.
        order = np.argsort(timestamps, kind="stable")
        timestamps, channels = timestamps[order], channels[order]
        if len(timestamps) > self.n_max_events:
            timestamps = timestamps[: self.n_max_events]
            channels = channels[: self.n_max_events]
        return timestamps, channels

    def getData(self) -> SimulatedTimeTagStreamBuffer:
        if self._running:
            self._accumulate_wall()
        timestamps, channels = self._build()
        buf = SimulatedTimeTagStreamBuffer(
            timestamps, channels, t_start=self._last_get_ps, t_get_data=self._elapsed_ps)
        self._last_get_ps = self._elapsed_ps
        self._pending_s = 0.0   # esvazia o buffer: cada tag é entregue uma só vez
        return buf


def createTimeTagger(serial: str = "", resolution: int = 0) -> SimulatedTimeTagger:
    """Substitui TT.createTimeTagger(serial="", resolution=Standard)."""
    return SimulatedTimeTagger(serial=serial, link=_DEFAULT_LINK)


def freeTimeTagger(tagger: SimulatedTimeTagger) -> None:
    """Substitui TT.freeTimeTagger(tagger)."""
    tagger._freed = True


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
    tt_mod.freeTimeTagger = freeTimeTagger
    tt_mod.TimeTagStream = SimulatedTimeTagStream
    tt_mod.TagType = TagType
    tt_mod.CHANNEL_UNUSED = CHANNEL_UNUSED
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