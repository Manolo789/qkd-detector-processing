import os
import socket
from QKD_NODE import SessionConfig, run_session, print_report

# --- Configuração do pós-processamento (pode ser sobrescrita por variáveis de ambiente) ---
EC_METHOD = os.environ.get("EC_METHOD", "cascade")            # 'cascade' (padrão) ou 'ldpc'
AUTH_KEY_FILE = os.environ.get("AUTH_KEY_FILE", "auth_key.json")  # chave pré-compartilhada (mesmo arquivo em Alice e Bob)
CAPTURE_S = float(os.environ.get("CAPTURE_S", "0.01"))       # duração da captura (s); IGUAL em Alice e Bob
CLASSIC_PORT = int(os.environ.get("CLASSIC_PORT", "65432"))   # ÚNICA porta TCP (sifting + pós-processamento); IGUAL em Alice e Bob
SLOT_PS = int(os.environ.get("SLOT_PS", "800"))              # largura do slot de coincidência (ps); IGUAL em Alice e Bob
SIMULATE = os.environ.get("SIMULATE", "0") == "1"            # 1 = hardware simulado (SIMLINK, seed=42)


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
    # Toda a sessão (captura -> sifting -> PE -> EC -> PA -> AUTH) está em QKD_NODE.run_session
    res = run_session(SessionConfig(role="alice", port=CLASSIC_PORT, capture_s=CAPTURE_S,
                                    slot_ps=SLOT_PS, ec_method=EC_METHOD,
                                    auth_key_file=AUTH_KEY_FILE, simulate=SIMULATE))
    print_report(res)


if __name__ == "__main__":
    main()
