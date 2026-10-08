# Utilizar apenas no teste de hardware simulado
#import SIMLINK
#SIMLINK.install_simulated_hardware(pair_rate_hz=3e6, eve_intercept_resend=False)


import warnings

import Swabian.TimeTagger as TT
import numpy as np

# Install Swabian with 'pip install Swabian-TimeTagger' in Linux and Windows
# API de referência: Time Tagger User Manual (release 2.22.6.0).

class BBM92HardwareManager:
    """Interface com o Swabian TimeTagger configurável por nó."""

    # Tamanho padrão do buffer do TimeTagStream (n_max_events), em eventos.
    DEFAULT_MAX_EVENTS = 10_000_000

    def __init__(self, node_type, serial_number=None):
        # Valida o nó ANTES de abrir o dispositivo, para não deixar o Time
        # Tagger ocupado se o argumento estiver errado.
        if node_type.upper() == 'ALICE':
            channels = {'H': 1, 'V': 2, 'D': 3, 'A': 4}
        elif node_type.upper() == 'BOB':
            channels = {'H': 5, 'V': 6, 'D': 7, 'A': 8}
        else:
            raise ValueError("O nó deve ser 'Alice' ou 'Bob'")
        self.channels = channels

        # createTimeTagger(serial="") conecta ao primeiro dispositivo livre;
        # com serial, ao dispositivo específico. Levanta RuntimeError se não
        # houver dispositivo ou se o serial estiver incorreto.
        if serial_number:
            self.tagger = TT.createTimeTagger(serial_number)
        else:
            self.tagger = TT.createTimeTagger()

        # Metadados da última captura (preenchidos por capture_stream).
        self.last_capture = {}

    def calibrate_delays(self, delays_ps: dict):
        """
        Compensa diferenças nos comprimentos de fibra óptica/cabeamento.
        delays_ps: ex. {1: 0, 2: 120, ...} em picossegundos (inteiros).
        Canais que não pertencem a este nó são ignorados (com aviso).
        """
        own = set(self.channels.values())
        for ch, delay in delays_ps.items():
            if ch not in own:
                warnings.warn(
                    f"calibrate_delays: canal {ch} não pertence a este nó "
                    f"({sorted(own)}); ignorado.", stacklevel=2)
                continue
            # setInputDelay(channel, delay) espera o atraso em ps como inteiro.
            self.tagger.setInputDelay(int(ch), int(round(delay)))
        # Garante que os atrasos já estão ativos no FPGA antes de capturar.
        self.tagger.sync()

    def capture_stream(self, duration_s: float, n_max_events: int = None):
        """
        Coleta marcas de tempo brutas por `duration_s` segundos de tempo do
        fluxo de dados do Time Tagger (não do relógio do computador).

        Retorna (timestamps_ps, channels) apenas com eventos do tipo
        TagType.TimeTag, ou seja, detecções reais. Tags de erro, de
        overflow e de eventos perdidos (MissedEvents) são descartadas, mas
        contabilizadas em `self.last_capture` e sinalizadas por warnings.
        """
        if n_max_events is None:
            n_max_events = self.DEFAULT_MAX_EVENTS

        # TimeTagStream(tagger, n_max_events, channels): a medição começa a
        # acumular dados já na criação do objeto.
        stream = TT.TimeTagStream(self.tagger, n_max_events,
                                  list(self.channels.values()))

        # startFor() limpa o buffer e interrompe a captura sozinho após a
        # duração dada EM PICOSSEGUNDOS (tempo do fluxo de dados). Isso evita
        # perder tags ainda em trânsito no pipeline (latência de até ~100 ms)
        # quando se usa sleep() + stop().
        stream.startFor(int(round(duration_s * 1e12)))
        timeout_ms = int((duration_s + 10.0) * 1000)
        if not stream.waitUntilFinished(timeout_ms):
            stream.stop()
            raise RuntimeError(
                f"TimeTagStream não concluiu a captura em {duration_s:.3f} s "
                f"(+10 s de tolerância).")

        data = stream.getData()
        timestamps = np.asarray(data.getTimestamps())
        channels = np.asarray(data.getChannels())
        event_types = np.asarray(data.getEventTypes())

        is_tag = event_types == int(TT.TagType.TimeTag)
        n_missed = int(np.sum(np.asarray(data.getMissedEvents(), dtype=np.int64)))
        tagger_overflows = self.tagger.getOverflowsAndClear()
        buffer_full = int(data.size) >= n_max_events

        self.last_capture = {
            "n_events": int(data.size),
            "n_time_tags": int(np.sum(is_tag)),
            "n_missed_events": n_missed,
            "has_overflows": bool(data.hasOverflows),
            "tagger_overflows": int(tagger_overflows),
            "buffer_full": bool(buffer_full),
        }
        if buffer_full:
            warnings.warn(
                f"TimeTagStream atingiu n_max_events={n_max_events}: eventos "
                f"provavelmente foram descartados. Reduza a duração da captura "
                f"ou aumente n_max_events.")
        if data.hasOverflows or tagger_overflows or n_missed:
            warnings.warn(
                f"Overflow no Time Tagger (overflows={tagger_overflows}, "
                f"eventos perdidos={n_missed}): taxa de contagem acima do "
                f"limite do link; a chave bruta pode estar incompleta.")

        return timestamps[is_tag], channels[is_tag]

    def close(self):
        """Libera o Time Tagger (freeTimeTagger). Seguro chamar mais de uma vez."""
        tagger, self.tagger = getattr(self, "tagger", None), None
        if tagger is not None:
            TT.freeTimeTagger(tagger)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class BBM92ProtocolEngine:
    """
    Motor do protocolo BBM92 para mapeamento, sifting e cálculo do QBER.

    Sifting por ÍNDICE DE SLOT (sem timestamps na rede)
    ---------------------------------------------------
    Cada nó divide a sua linha do tempo (já calibrada, na base de tempo comum)
    em slots de largura fixa `slot_ps` e anuncia, para cada slot em que houve
    detecção, apenas:

        * o índice do slot:  k = floor(t / slot_ps)
        * o código da base:  1 = retilínea (Z: H/V)   2 = diagonal (X: D/A)

    O instante fino da detecção (resolução de ps) NUNCA sai do nó. Antes, os
    timestamps brutos de Alice e Bob eram publicados; com os dois, um espião
    calculava t_Alice - t_Bob de cada par e, como cada detector tem atraso e
    jitter próprios (resíduo de calibração), podia inferir qual detector
    disparou, isto é, o VALOR do bit (canal lateral temporal; Lamas-Linares
    e Kurtsiefer, Opt. Express 15, 9388, 2007). Com índices de slot, a posição
    da detecção dentro do slot fica oculta.

    Coincidência = mesmo índice de slot nos dois nós. Pares cujo atraso
    relativo os faz cair em slots vizinhos são perdidos (fração ~E|dt|/slot_ps,
    cerca de 10% com jitter de 350 ps FWHM e slot de 1600 ps). Esses pares
    são descartados nos dois lados e não entram na chave.

    Se um nó tiver mais de uma detecção no mesmo slot (clique múltiplo,
    dark count ou coincidência acidental), só a mais antiga é usada, como na
    varredura de dois ponteiros anterior. O número de slots com várias
    detecções fica em `self.n_multi_click` (local, não é anunciado).
    """

    BASIS_CODE = {'Z': 1, 'X': 2}          # 1 = retilínea (H/V), 2 = diagonal (D/A)
    CODE_BASIS = {1: 'Z', 2: 'X'}

    def __init__(self, coincidence_window_ps: int = 1000, slot_ps: int = None):
        # `coincidence_window_ps` é mantido por compatibilidade: a janela antiga
        # aceitava |dt| <= janela, ou seja, uma largura total de 2 x janela.
        # O slot padrão tem essa mesma largura, o que mantém a taxa de
        # coincidências acidentais aproximadamente igual.
        self.window = int(coincidence_window_ps)
        self.slot_ps = int(slot_ps) if slot_ps else 2 * self.window
        if self.slot_ps <= 0:
            raise ValueError("slot_ps deve ser um inteiro positivo (picossegundos).")

        # Mapeamento de canais para base (Z ou X) e valor de bit
        self.channel_map = {
            1: ('Z', 0), 2: ('Z', 1), 3: ('X', 0), 4: ('X', 1),  # Alice
            5: ('Z', 0), 6: ('Z', 1), 7: ('X', 0), 8: ('X', 1)   # Bob
        }
        self._known_channels = np.array(sorted(self.channel_map), dtype=np.int64)
        lut_size = int(self._known_channels.max()) + 1
        self._lut_code = np.zeros(lut_size, dtype=np.uint8)
        self._lut_bit = np.zeros(lut_size, dtype=np.uint8)
        for ch, (base, bit) in self.channel_map.items():
            self._lut_code[ch] = self.BASIS_CODE[base]
            self._lut_bit[ch] = bit

        # Estado local pós-medição (um elemento por slot ocupado).
        # NADA disto além de `slots` e `basis_codes` pode ir para a rede.
        self.slots = np.array([], dtype=np.int64)        # índices de slot (crescentes)
        self.basis_codes = np.array([], dtype=np.uint8)  # 1 ou 2 por slot
        self.bases = np.array([])                        # 'Z'/'X' por slot (compatibilidade)
        self.n_detections = 0
        self.n_multi_click = 0
        self.n_coincidences = 0

    def process_time_tags(self, timestamps, channels):
        """
        Converte as detecções locais em (bit, base, índice de slot).

        Retorna (key, n): o bit de cada slot ocupado (detecção mais antiga do
        slot) e o número de slots. Guarda `slots`, `basis_codes` e `bases`.
        """
        ts = np.asarray(timestamps, dtype=np.int64).ravel()
        ch = np.asarray(channels, dtype=np.int64).ravel()
        if ts.shape != ch.shape:
            raise ValueError("timestamps e channels devem ter o mesmo tamanho.")

        keep = np.isin(ch, self._known_channels)
        ts, ch = ts[keep], ch[keep]
        order = np.argsort(ts, kind="stable")
        ts, ch = ts[order], ch[order]

        slots = np.floor_divide(ts, self.slot_ps)
        # `first` = posição da 1ª (mais antiga) detecção de cada slot
        uniq, first = np.unique(slots, return_index=True)
        sel = ch[first]

        self.slots = uniq.astype(np.int64)
        self.basis_codes = self._lut_code[sel]
        self.bases = np.where(self.basis_codes == 1, 'Z', 'X')
        self.n_detections = int(len(ts))
        self.n_multi_click = int(len(ts) - len(uniq))

        key = self._lut_bit[sel]
        return key, len(key)

    def announcement(self) -> dict:
        """
        Mensagem pública do sifting: índices de slot e códigos de base.
        Formato: {"slot_ps": int, "idx": [k, ...], "bases": "1212..."}.
        """
        codes = (self.basis_codes.astype(np.uint8) + ord('0')).tobytes().decode("ascii")
        return {"slot_ps": self.slot_ps, "idx": self.slots.tolist(), "bases": codes}

    def _parse_announcement(self, remote):
        """Valida a mensagem do outro nó e devolve (índices, códigos)."""
        if not isinstance(remote, dict) or not {"slot_ps", "idx", "bases"} <= remote.keys():
            raise ValueError("Mensagem de sifting malformada: esperado "
                             "{'slot_ps', 'idx', 'bases'}.")
        if int(remote["slot_ps"]) != self.slot_ps:
            raise ValueError(f"Largura de slot diferente entre os nós "
                             f"(local={self.slot_ps} ps, remoto={remote['slot_ps']} ps). "
                             f"Use o mesmo SLOT_PS em Alice e Bob.")
        idx = np.asarray(remote["idx"], dtype=np.int64).ravel()
        codes = np.frombuffer(str(remote["bases"]).encode("ascii"), dtype=np.uint8) - ord('0')
        if len(idx) != len(codes):
            raise ValueError(f"Mensagem de sifting inconsistente: {len(idx)} índices "
                             f"e {len(codes)} códigos de base.")
        if len(codes) and not np.all((codes == 1) | (codes == 2)):
            raise ValueError("Código de base inválido (esperado 1 = retilínea ou 2 = diagonal).")
        if len(idx) > 1 and not np.all(np.diff(idx) > 0):
            raise ValueError("Índices de slot remotos devem ser estritamente crescentes.")
        return idx, codes

    def perform_sifting(self, local_key, remote, is_bob=False):
        """
        Peneiramento por índice de slot.

        `remote` é a mensagem do outro nó (dict de `announcement()`). Um slot
        ocupado nos DOIS nós é uma coincidência; se os códigos de base forem
        iguais, o bit local entra na chave peneirada (em Bob, invertido:
        1 - bit, por causa da anticorrelação do singleto). Os slots comuns são
        percorridos em ordem crescente nos dois lados, então as chaves saem
        alinhadas. Os bits nunca transitam na rede.

        Levanta ValueError se a mensagem remota for inválida.
        """
        local_key = np.asarray(local_key, dtype=np.uint8).ravel()
        if len(local_key) != len(self.slots):
            raise ValueError("local_key deve ser o vetor devolvido por process_time_tags.")
        r_idx, r_codes = self._parse_announcement(remote)

        _, li, ri = np.intersect1d(self.slots, r_idx, assume_unique=True,
                                   return_indices=True)
        self.n_coincidences = int(len(li))
        same_basis = self.basis_codes[li] == r_codes[ri]
        sifted_key = local_key[li[same_basis]]
        if is_bob:
            sifted_key = 1 - sifted_key
        return sifted_key.astype(np.uint8), int(len(sifted_key))

    def calculate_qber(self, qber_bits_a, qber_bits_b):
        """Calcula o QBER baseando-se apenas no subconjunto de bits revelados."""
        min_len = min(len(qber_bits_a), len(qber_bits_b))
        if min_len == 0: return 0.0
        errors = np.sum(qber_bits_a[:min_len] != qber_bits_b[:min_len])
        return errors / min_len
