import os
from QKD_NODE import SessionConfig, run_session, print_report

# --- Configuração do pós-processamento (deve ser IGUAL à de ALICE.py) ---
EC_METHOD = os.environ.get("EC_METHOD", "cascade")
AUTH_KEY_FILE = os.environ.get("AUTH_KEY_FILE", "auth_key.json")
CAPTURE_S = float(os.environ.get("CAPTURE_S", "0.01"))       # duração da captura (s); IGUAL em Alice e Bob
CLASSIC_PORT = int(os.environ.get("CLASSIC_PORT", "65432"))   # ÚNICA porta TCP (sifting + pós-processamento); IGUAL em Alice e Bob
SLOT_PS = int(os.environ.get("SLOT_PS", "800"))              # largura do slot de coincidência (ps); IGUAL em Alice e Bob
SIMULATE = os.environ.get("SIMULATE", "0") == "1"            # 1 = hardware simulado (SIMLINK, seed=42)

# ALICE_HOST=<IP DE ALICE> python BOB.py
ALICE_HOST = os.environ.get("ALICE_HOST", "").strip()


def main():
    if not ALICE_HOST:
        raise SystemExit(
            "Defina o IP de Alice na rede antes de rodar BOB.py, por "
            "exemplo:\n    ALICE_HOST=192.168.1.10 python BOB.py\n"
            "(rode ALICE.py primeiro -- ele imprime esse IP no início da execução)."
        )
    # Toda a sessão (captura -> sifting -> PE -> EC -> PA -> AUTH) está em QKD_NODE.run_session
    res = run_session(SessionConfig(role="bob", peer_host=ALICE_HOST, port=CLASSIC_PORT,
                                    capture_s=CAPTURE_S, slot_ps=SLOT_PS, ec_method=EC_METHOD,
                                    auth_key_file=AUTH_KEY_FILE, simulate=SIMULATE))
    print_report(res)


if __name__ == "__main__":
    main()
