# Utilizar apenas no teste de hardware simulado
#import SIMLINK
#SIMLINK.install_simulated_hardware(pair_rate_hz=3e6, eve_intercept_resend=False)


import Swabian.TimeTagger as TT
import numpy as np

# Install Swabian with 'pip install Swabian-TimeTagger' in Linux and Windows

class BBM92HardwareManager:
    """Interface com o Swabian TimeTagger configurável por nó."""
    def __init__(self, node_type, serial_number=None):
        if serial_number:
            self.tagger = TT.createTimeTagger(serial_number)
        else:
            self.tagger = TT.createTimeTagger()
            
        # Define os canais baseado em qual nó está instanciando a classe
        if node_type.upper() == 'ALICE':
            self.channels = {'H': 1, 'V': 2, 'D': 3, 'A': 4}
        elif node_type.upper() == 'BOB':
            self.channels = {'H': 5, 'V': 6, 'D': 7, 'A': 8}
        else:
            raise ValueError("O nó deve ser 'Alice' ou 'Bob'")
        
    def calibrate_delays(self, delays_ps: dict):
        """
        Compensa diferenças nos comprimentos de fibra óptica/cabeamento.
        delays_ps: ex. {1: 0, 2: 120, 5: 3500, ...} em picossegundos.
        """
        for ch, delay in delays_ps.items():
            self.tagger.setInputDelay(ch, delay)
            
    def capture_stream(self, duration_s: float):
        """Coleta marcas de tempo brutas por um intervalo de tempo."""
        stream = TT.TimeTagStream(self.tagger, buffer_size=10_000_000, channels=list(self.channels.values()))

        
        stream.start()
        self.tagger.sync()
        import time
        time.sleep(duration_s)
        stream.stop()
        
        data = stream.getEvents()
        return data.getTimestamps(), data.getChannels()


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
