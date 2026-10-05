import json
import os
import socket
import numpy as np
from BBM92 import BBM92HardwareManager, BBM92ProtocolEngine
from CLASSIC_CHANNEL import AliceClassicalChannel
from AUXILIARY import AuthKeyPool, ClassicalLink, PostProcessingError
from POSTPROCESS import run_postprocessing

# --- Configuração do pós-processamento (pode ser sobrescrita por variáveis de ambiente) ---
EC_METHOD = os.environ.get("EC_METHOD", "cascade")            # 'cascade' (padrão) ou 'ldpc'
AUTH_KEY_FILE = os.environ.get("AUTH_KEY_FILE", "auth_key.json")  # chave pré-compartilhada (mesmo arquivo em Alice e Bob)
CAPTURE_S = float(os.environ.get("CAPTURE_S", "0.01"))       # duração da captura (s); IGUAL em Alice e Bob
CLASSIC_PORT = int(os.environ.get("CLASSIC_PORT", "65432"))   # ÚNICA porta TCP (sifting + pós-processamento); IGUAL em Alice e Bob

# Utilizar apenas no teste de hardware simulado
#import SIMLINK
#SIMLINK.install_simulated_hardware(pair_rate_hz=3e6, eve_intercept_resend=False, seed=42)

def _print_local_ip():
    """Descobre e imprime o IP desta máquina na rede local, para que o valor
    possa ser configurado em BOB.py (variável ALICE_HOST)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # não envia nada de fato, só resolve a rota/interface
        ip = s.getsockname()[0]
    except OSError:
        ip = "127.0.0.1"
    finally:
        s.close()
    print(f"[ALICE] IP desta máquina na rede (configure ALICE_HOST em BOB.py com este valor): {ip}")

def main():
    _print_local_ip()

    # 1. Canal clássico ÚNICO e persistente (porta única): é aberto aqui, usado
    #    no sifting e reaproveitado, pelo ClassicalLink, em todo o pós-processamento.
    with AliceClassicalChannel(port=CLASSIC_PORT) as channel:
        hw = BBM92HardwareManager(node_type='Alice')
        try:
            hw.calibrate_delays({1: 0, 2: 50, 3: 0, 4: 50})

            print("[ALICE] Realizando capturas do SPAD...")
            timestamps, channels = hw.capture_stream(duration_s=CAPTURE_S)
        finally:
            hw.close()  # libera o Time Tagger (freeTimeTagger); só a captura usa o hardware

        engine = BBM92ProtocolEngine(coincidence_window_ps=800)
        key, sifted_len = engine.process_time_tags(timestamps, channels)
        print(f"[ALICE] Chave gerada (não peneirada): {sifted_len} bits adquiridos.")

        dados_bases_alice = {
            "ts": engine.valid_ts.tolist(),
            "bases": engine.bases.tolist()
        }

        # Troca de bases (sifting) pela conexão persistente
        print("[ALICE] Trocando BASES (Sifting) via canal clássico...")
        resposta_bob = channel.exchange(json.dumps(dados_bases_alice))
        dados_bob = json.loads(resposta_bob)

        ts_bob = np.array(dados_bob["ts"])
        bases_bob = np.array(dados_bob["bases"])

        sifted_key, final_sifted_len = engine.perform_sifting(key, ts_bob, bases_bob, is_bob=False)
        if final_sifted_len == 0:
            print("[ALICE][ERRO] Nenhuma coincidência foi encontrada entre Alice e "
                  "Bob. Verifique a conectividade de rede (host/porta) e se os "
                  "dois processos usam a mesma seed/duração de captura. "
                  "Encerrando sem gerar chave.")
            return
        print(f"Chave bruta: {key} bits")
        print(f"[ALICE] Chave peneirada (sifting): {final_sifted_len} bits")

        # === PÓS-PROCESSAMENTO (independente de nó: POSTPROCESS.py) ===
        # PE (QBER) -> correção de erros -> privacy amplification -> autenticação,
        # tudo na MESMA conexão TCP do sifting.
        link = ClassicalLink(role="alice", channel=channel)
        # As bases/timestamps trocados no sifting também entram na transcrição autenticada:
        link.record(alice_msg=json.dumps(dados_bases_alice), bob_msg=resposta_bob)
        pool = AuthKeyPool.load(AUTH_KEY_FILE)
        try:
            res = run_postprocessing(sifted_key, link, pool, ec_method=EC_METHOD)
        except PostProcessingError as exc:
            print(f"[ALICE][ABORTADO] {type(exc).__name__}: {exc}")
            return

        print("\n=== RESULTADOS FINAIS EM ALICE ===")
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
            print("[SUCESSO] Chave segura e autenticada gerada em Alice.")

if __name__ == "__main__":
    main()