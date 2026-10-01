import socket
import struct
import time

_HEADER = struct.Struct("!Q")  # cabeçalho de 8 bytes (tamanho da mensagem, big-endian)


def _send_message(sock: socket.socket, message: str) -> None:
    """Envia uma mensagem de tamanho arbitrário, prefixada pelo seu tamanho."""
    payload = message.encode("utf-8")
    sock.sendall(_HEADER.pack(len(payload)))
    sock.sendall(payload)


def _receive_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Lê exatamente num_bytes do socket, repetindo recv() quantas vezes
    forem necessárias."""
    chunks = []
    remaining = num_bytes
    while remaining > 0:
        chunk = sock.recv(min(remaining, 65536))
        if not chunk:
            raise ConnectionError(
                "Conexão encerrada pelo outro lado antes de receber a mensagem completa."
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _receive_message(sock: socket.socket) -> str:
    (length,) = _HEADER.unpack(_receive_exact(sock, _HEADER.size))
    return _receive_exact(sock, length).decode("utf-8")


class AliceClassicalChannel:
    """
    Canal clássico persistente para Alice (Servidor TCP).
    """
    def __init__(self, host='0.0.0.0', port=65432):
        self.host = host
        self.port = port
        self.server_socket = None
        self.conn = None

    def connect(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen()
        print(f"[Alice] Aguardando conexão de Bob em {self.host}:{self.port}...")
        self.conn, addr = self.server_socket.accept()
        print(f"[Alice] Conectado a Bob em {addr}")

    def exchange(self, msg: str) -> str:
        """Envia mensagem para Bob e aguarda a resposta dele na mesma conexão."""
        _send_message(self.conn, msg)
        print("[Alice] Mensagem enviada via canal clássico.")
        return _receive_message(self.conn)

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
        if self.server_socket:
            self.server_socket.close()
            self.server_socket = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class BobClassicalChannel:
    """
    Canal clássico persistente para Bob (Cliente TCP).
    """
    def __init__(self, host, port=65432, connect_timeout_s=30.0, retry_interval_s=0.5):
        if not host or host in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(
                "bob_classical_channel: 'host' precisa ser o endereço IP real de "
                "Alice na rede local (ex.: '192.168.1.10'), nunca localhost -- "
                "Alice e Bob rodam em computadores diferentes."
            )
        self.host = host
        self.port = port
        self.connect_timeout_s = connect_timeout_s
        self.retry_interval_s = retry_interval_s
        self.sock = None

    def connect(self):
        deadline = time.monotonic() + self.connect_timeout_s
        last_error = None
        while time.monotonic() < deadline:
            try:
                self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sock.connect((self.host, self.port))
                print(f"[Bob] Conectado a Alice em {self.host}:{self.port}.")
                return
            except (ConnectionRefusedError, OSError) as exc:
                if self.sock:
                    self.sock.close()
                    self.sock = None
                last_error = exc
                time.sleep(self.retry_interval_s)
        raise ConnectionError(
            f"[Bob] Não foi possível conectar a Alice em {self.host}:{self.port} após "
            f"{self.connect_timeout_s:.0f}s de tentativas: {last_error}"
        )

    def exchange(self, msg: str) -> str:
        """Recebe a mensagem de Alice primeiro e depois envia a resposta."""
        msg_recebida = _receive_message(self.sock)
        _send_message(self.sock, msg)
        print("[Bob] Mensagem enviada via canal clássico.")
        return msg_recebida

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
