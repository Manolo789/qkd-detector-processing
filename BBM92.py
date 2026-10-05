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
    """Motor do protocolo BBM92 para sifting, mapeamento e cálculo do QBER."""
    def __init__(self, coincidence_window_ps: int = 1000):
        self.window = coincidence_window_ps  # Janela de coincidência (ex: 1000 ps = 1 ns)
        
        # Mapeamento de canais para base (Z ou X) e valor de bit
        # Base Z = 0, Base X = 1
        self.channel_map = {
            1: ('Z', 0), 2: ('Z', 1), 3: ('X', 0), 4: ('X', 1),  # Alice
            5: ('Z', 0), 6: ('Z', 1), 7: ('X', 0), 8: ('X', 1)   # Bob
        }
        # Propriedades para salvar os estados pós-medição
        self.valid_ts = np.array([])
        self.bases = np.array([])



    def process_time_tags(self, timestamps, channels):
        """Extrai a chave bruta gerada e armazena bases e tempos locais para sifting."""
        raw_key = []
        bases = []
        valid_ts = []
        
        for t, c in zip(timestamps, channels):
            if c in self.channel_map:
                base, bit = self.channel_map[c]
                raw_key.append(bit)
                bases.append(base)
                valid_ts.append(t)
                
        self.valid_ts = np.array(valid_ts)
        self.bases = np.array(bases)
        
        key = np.array(raw_key)
        sifted_len = len(key)
        return key, sifted_len

    def perform_sifting(self, local_key, remote_ts, remote_bases, is_bob=False):
        """
        Alinha temporalmente os eventos locais e remotos.
        Gera a chave peneirada comparando as bases, sem que os bits transitem na rede.
        """
        sifted_key = []
        i, j = 0, 0
        len_local = len(self.valid_ts)
        len_remote = len(remote_ts)
        
        while i < len_local and j < len_remote:
            diff = self.valid_ts[i] - remote_ts[j]
            if abs(diff) <= self.window:
                if self.bases[i] == remote_bases[j]:
                    if is_bob:
                        sifted_key.append(1 - local_key[i])
                    else:
                        sifted_key.append(local_key[i])
                i += 1
                j += 1
            elif self.valid_ts[i] < remote_ts[j]:
                i += 1
            else:
                j += 1
                
        return np.array(sifted_key), len(sifted_key)

    def calculate_qber(self, qber_bits_a, qber_bits_b):
        """Calcula o QBER baseando-se apenas no subconjunto de bits revelados."""
        min_len = min(len(qber_bits_a), len(qber_bits_b))
        if min_len == 0: return 0.0
        errors = np.sum(qber_bits_a[:min_len] != qber_bits_b[:min_len])
        return errors / min_len