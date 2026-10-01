"""
AUXILIARY.py
======

Funções auxiliares de USO RESTRITO ao POSTPROCESS.py (pós-processamento
clássico de QKD: estimação de parâmetros, correção de erros, amplificação de
privacidade e autenticação). Nada aqui é chamado diretamente por ALICE.py ou
BOB.py -- eles só falam com o POSTPROCESS.py (e com `ClassicalLink`, que é o
adaptador entre o POSTPROCESS e o canal clássico PERSISTENTE, de porta única,
definido em CLASSIC_CHANNEL.py: `AliceClassicalChannel` / `BobClassicalChannel`).

Convenções
----------
* Chaves são `np.ndarray` de `uint8` com valores 0/1.
* Alice é sempre a REFERÊNCIA: a chave dela nunca é alterada na correção de
  erros; Bob é quem corrige a dele ("reverse reconciliation" NÃO é usada).
* Todas as mensagens trocadas passam por `ClassicalLink`, que registra a
  transcrição completa (usada na autenticação final).
* Aleatoriedade que precisa ser imprevisível (amostra da QBER, sementes de
  hash, bits punçados) vem do módulo `secrets`. Aleatoriedade PÚBLICA e
  reprodutível (permutações do Cascade, matriz H do LDPC) usa
  `np.random.RandomState`, cujo fluxo é CONGELADO pelo NumPy (o `Generator`
  novo não garante o mesmo fluxo entre versões, e Alice e Bob podem ter
  versões diferentes do NumPy).

Índice
------
  1. Exceções e utilitários (entropia binária, empacotamento de bits)
  2. ClassicalLink  -- adaptador para o canal persistente de CLASSIC_CHANNEL.py + transcrição
  3. Estimação de parâmetros -- limite de Serfling
  4. Cascade  (Brassard-Salvail 1993; ver Martinez-Mateo et al., QIC 15, 2015)
  5. LDPC rate-adaptive por síndrome (Elkouss et al., ISIT 2009 / QIC 2011)
  6. Verificação da correção de erros (hash 2-universal)
  7. Hash de Toeplitz (2-universal) e comprimento da chave final
  8. Autenticação Wegman-Carter (hash polinomial + one-time pad)
"""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import queue
import secrets
import socket
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from CLASSIC_CHANNEL import AliceClassicalChannel, BobClassicalChannel


# =============================================================================
# 1. EXCEÇÕES E UTILITÁRIOS
# =============================================================================
class PostProcessingError(Exception):
    """Erro genérico do pós-processamento."""


class PostProcessingAbort(PostProcessingError):
    """O protocolo deve ser abortado (QBER alta, verificação falhou, ...).
    É levantada de forma CONSISTENTE nos dois nós."""


class AuthenticationError(PostProcessingError):
    """A transcrição do canal clássico não foi autenticada: a chave gerada
    NÃO deve ser usada."""


class AuthKeyExhausted(PostProcessingError):
    """Acabou a chave pré-compartilhada de autenticação."""


def binary_entropy(x) -> float:
    """h2(x) = -x log2 x - (1-x) log2 (1-x)."""
    x = float(min(max(x, 1e-12), 1 - 1e-12))
    return -x * math.log2(x) - (1 - x) * math.log2(1 - x)


def as_bits(key) -> np.ndarray:
    return np.asarray(key, dtype=np.uint8).ravel().copy()


def random_bits(n: int) -> np.ndarray:
    """n bits uniformes vindos do CSPRNG do sistema operacional."""
    if n <= 0:
        return np.zeros(0, dtype=np.uint8)
    raw = np.frombuffer(secrets.token_bytes((n + 7) // 8), dtype=np.uint8)
    return np.unpackbits(raw)[:n].copy()


def pack_bits(bits) -> str:
    """Vetor de bits -> string hexadecimal (8x mais compacto que JSON)."""
    bits = np.asarray(bits, dtype=np.uint8)
    return np.packbits(bits).tobytes().hex()


def unpack_bits(hex_str: str, n: int) -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex(hex_str), dtype=np.uint8)
    return np.unpackbits(raw)[:n].copy()


def bits_to_int(bits) -> int:
    v = 0
    for b in np.asarray(bits, dtype=np.uint8):
        v = (v << 1) | int(b)
    return v


def log(role: str, msg: str, verbose: bool = True) -> None:
    if verbose:
        print(f"[POS-{role.upper()}] {msg}")


# =============================================================================
# 2. CLASSICAL LINK  (adaptador para o canal persistente de CLASSIC_CHANNEL.py)
# =============================================================================
def _enable_tcp_nodelay(channel) -> None:
    """Desliga o algoritmo de Nagle no socket do canal (melhor esforço).

    O CLASSIC_CHANNEL envia cada mensagem em dois `sendall` (cabeçalho de 8
    bytes + corpo). Com Nagle ligado, o 2º pacote pode esperar o ACK atrasado
    (~40 ms) do 1º; como o Cascade faz centenas de trocas pequenas, isso
    dominaria o tempo de execução. Só tem efeito se o canal já estiver
    conectado (por isso o ClassicalLink deve ser criado DEPOIS do connect)."""
    sock = getattr(channel, "conn", None) or getattr(channel, "sock", None)
    if sock is not None:
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass


class ClassicalLink:
    """
    Adaptador entre o POSTPROCESS e o canal clássico persistente de
    CLASSIC_CHANNEL.py (`AliceClassicalChannel` / `BobClassicalChannel`).

    Há UMA única conexão TCP (porta única), aberta uma só vez por ALICE.py /
    BOB.py e compartilhada por TODAS as fases: sifting, estimação de
    parâmetros, correção de erros, privacy amplification e autenticação. O
    canal oferece uma primitiva: `exchange(msg)` -- troca simétrica na mesma
    conexão (Alice envia e depois recebe; Bob recebe e depois envia). Toda a
    comunicação do pós-processamento é construída sobre ela:

        exchange(msg)  -> troca simétrica (devolve a mensagem do outro nó)
        send(obj)      -> este nó fala, o outro só escuta   (o outro chama recv)
        recv()         -> este nó escuta, o outro fala      (o outro chama send)

    Como Alice e Bob executam o MESMO roteiro em lockstep, a k-ésima chamada
    de um lado sempre casa com a k-ésima chamada do outro.

    Além disso o link:
      * grava a transcrição de tudo que foi trocado (`transcript`), que é o
        que a etapa de autenticação protege. Trocas feitas FORA do link (ex.:
        sifting em ALICE.py/BOB.py) devem ser registradas com `record()`;
      * silencia (por padrão) o `print` do canal, que imprimiria uma linha
        por troca (o Cascade faz centenas delas);
      * não abre nem fecha a conexão: quem a possui é o código que criou o
        canal (`with AliceClassicalChannel(...) as channel:`). Crie o link
        DEPOIS de o canal estar conectado e use-o dentro do bloco `with`.

    Parâmetros
    ----------
    role         'alice' (servidor TCP) ou 'bob' (cliente TCP)
    channel      `AliceClassicalChannel` (role='alice') ou
                 `BobClassicalChannel` (role='bob') JÁ CONECTADO
    exchange_fn  (opcional) função `msg -> msg_remota` que substitui o canal;
                 usada por `make_local_link_pair()` para testes num só
                 processo (então `channel` pode ser omitido).
    quiet        se True, suprime os prints do canal a cada troca
    """

    def __init__(self, role: str, channel=None,
                 exchange_fn: Optional[Callable[[str], str]] = None,
                 quiet: bool = True):
        role = role.strip().lower()
        if role not in ("alice", "bob"):
            raise ValueError("role deve ser 'alice' ou 'bob'")
        if channel is None and exchange_fn is None:
            raise ValueError(
                "ClassicalLink precisa de um canal já conectado "
                "(channel=AliceClassicalChannel/BobClassicalChannel) ou de uma exchange_fn.")
        if channel is not None:
            expected = AliceClassicalChannel if role == "alice" else BobClassicalChannel
            if not isinstance(channel, expected):
                raise ValueError(
                    f"role='{role}' exige um {expected.__name__}, mas recebeu "
                    f"{type(channel).__name__}.")
            _enable_tcp_nodelay(channel)
        self.role = role
        self.channel = channel
        self._exchange_fn = exchange_fn
        self.quiet = quiet
        self.transcript: List[tuple] = []   # [(msg_alice, msg_bob), ...]
        self.n_exchanges = 0
        self.bytes_total = 0

    # ---- primitivas -------------------------------------------------------
    def exchange(self, msg: str = "") -> str:
        if self._exchange_fn is not None:
            remote = self._exchange_fn(msg)
        else:
            sink = io.StringIO() if self.quiet else None
            ctx = contextlib.redirect_stdout(sink) if sink else contextlib.nullcontext()
            with ctx:
                remote = self.channel.exchange(msg)
        if self.role == "alice":
            self.record(msg, remote)
        else:
            self.record(remote, msg)
        return remote

    def send(self, obj) -> None:
        self.exchange(json.dumps(obj))

    def recv(self):
        return json.loads(self.exchange(""))

    # ---- transcrição --------------------------------------------------------
    def record(self, alice_msg: str, bob_msg: str) -> None:
        """Registra uma troca feita fora do link (mesma ordem nos dois nós!)."""
        self.transcript.append((alice_msg, bob_msg))
        self.n_exchanges += 1
        self.bytes_total += len(alice_msg) + len(bob_msg)

    def transcript_bytes(self) -> bytes:
        out = bytearray()
        for i, (a, b) in enumerate(self.transcript):
            ab, bb = a.encode("utf-8"), b.encode("utf-8")
            out += (i.to_bytes(8, "big") + len(ab).to_bytes(8, "big") + ab
                    + len(bb).to_bytes(8, "big") + bb)
        return bytes(out)


def make_local_link_pair():
    """Dois ClassicalLink ligados por filas em memória (para testes com duas
    threads no mesmo processo, sem rede)."""
    a2b, b2a = queue.Queue(), queue.Queue()

    def alice_fn(msg):
        a2b.put(msg)
        return b2a.get()

    def bob_fn(msg):
        b2a.put(msg)
        return a2b.get()

    return (ClassicalLink("alice", exchange_fn=alice_fn),
            ClassicalLink("bob", exchange_fn=bob_fn))


# =============================================================================
# 3. ESTIMAÇÃO DE PARÂMETROS -- limite de Serfling (PDF, Teorema 4.5, eq. 4.34)
# =============================================================================
def serfling_mu(n_rest: int, k_sample: int, eps: float) -> float:
    """
    Menor mu tal que  Pr[ e_resto >= e_amostra + mu ] <= eps  (eq. 4.34):

        exp( -2 k^2 n mu^2 / ((k+1) N) ) <= eps ,   N = n + k

        =>  mu = sqrt( N (k+1) ln(1/eps) / (2 n k^2) )
    """
    if n_rest <= 0 or k_sample <= 0:
        return 1.0
    N = n_rest + k_sample
    return math.sqrt(N * (k_sample + 1) * math.log(1.0 / eps)
                     / (2.0 * n_rest * k_sample ** 2))


# =============================================================================
# 4. CASCADE
# =============================================================================
def cascade_block_sizes(n: int, qber: float, n_passes: int,
                        k1: Optional[int] = None) -> List[int]:
    """
    Tamanhos de bloco por passagem.

    Cascade original (Brassard & Salvail): k1 = 0.73/QBER e k_i = 2 k_{i-1}.
    Como mostram Martinez-Mateo et al. (QIC 15, 2015), passagens extras com
    blocos de tamanho ~n/2 reduzem a taxa de erro de quadro de forma
    aproximadamente exponencial (~2^-s); por isso o tamanho é limitado a
    ceil(n/2) a partir da 2ª passagem.
    """
    if n <= 1:
        return [max(n, 1)] * n_passes
    q = max(qber, 1e-4)
    k = int(k1) if k1 else max(2, int(math.ceil(0.73 / q)))
    sizes = []
    for p in range(n_passes):
        cap = n if p == 0 else int(math.ceil(n / 2))
        sizes.append(int(min(max(k, 1), cap)))
        k *= 2
    return sizes


def _prefix_parity(key: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """P[i] = XOR de key[perm[0..i-1]]; paridade de [a,b) = P[b]^P[a]."""
    out = np.zeros(len(perm) + 1, dtype=np.uint8)
    out[1:] = np.cumsum(key[perm].astype(np.int64)) & 1
    return out


def _cascade_layout(n: int, sizes: List[int], seed: int):
    rs = np.random.RandomState(seed)
    perms = [np.arange(n)] + [rs.permutation(n) for _ in sizes[1:]]
    starts = [np.arange(0, n, k) for k in sizes]
    ends = [np.minimum(s + k, n) for s, k in zip(starts, sizes)]
    return perms, starts, ends


def _block_parities(key, perm, starts, ends) -> np.ndarray:
    P = _prefix_parity(key, perm)
    return P[ends] ^ P[starts]


def cascade_alice(key: np.ndarray, link: ClassicalLink, qber: float,
                  n_passes: int = 4, k1: Optional[int] = None,
                  seed: Optional[int] = None) -> Dict:
    """Lado de Alice: só ANUNCIA paridades (a chave dela nunca muda)."""
    n = len(key)
    sizes = cascade_block_sizes(n, qber, n_passes, k1)
    seed = secrets.randbits(31) if seed is None else int(seed)
    perms, starts, ends = _cascade_layout(n, sizes, seed)

    par = [_block_parities(key, perms[q], starts[q], ends[q])
           for q in range(n_passes)]
    leaked = int(sum(len(p) for p in par))
    link.send({"seed": seed, "sizes": sizes,
               "par": [pack_bits(p) for p in par]})

    pref = np.stack([_prefix_parity(key, perms[q]) for q in range(n_passes)])
    rounds = 0
    while True:
        req = link.recv()                      # pedido de Bob
        if req.get("done"):
            break
        q = np.asarray(req["q"], dtype=np.int64)
        a = np.asarray(req["a"], dtype=np.int64)
        b = np.asarray(req["b"], dtype=np.int64)
        bits = pref[q, b] ^ pref[q, a]         # paridade de perm_q[a:b]
        link.send({"bits": pack_bits(bits)})
        leaked += len(bits)
        rounds += 1
    return {"key": key, "leaked": leaked, "rounds": rounds,
            "n_corrected": None, "sizes": sizes}


def cascade_bob(key: np.ndarray, link: ClassicalLink) -> Dict:
    """
    Lado de Bob: corrige a própria chave.

    Variante "em lote" do Cascade: em cada rodada TODAS as buscas binárias
    (uma por bloco de paridade ímpar) andam em paralelo, o que reduz o número
    de idas-e-voltas na rede de (#erros x log2 k) para (~log2 k) por rodada de
    cascata. A lógica é a do Cascade original:

      passagem p:  (i) blocos ímpares da passagem p -> BINARY (acha 1 erro);
                   (ii) cada bit corrigido inverte a paridade do bloco que o
                   contém em TODAS as passagens anteriores; blocos que
                   viraram ímpares são tratados no mesmo laço (cascata);
                   repete até não restar bloco ímpar em nenhuma passagem <= p.

    Como Alice já mandou de antemão as paridades de TODOS os blocos de TODAS
    as passagens (a quantidade revelada é a mesma do Cascade sequencial), Bob
    detecta sozinho, sem tráfego, quais blocos ficaram ímpares.
    """
    key = key.copy()
    n = len(key)
    init = link.recv()
    sizes, seed = init["sizes"], init["seed"]
    P = len(sizes)
    perms, starts, ends = _cascade_layout(n, sizes, seed)
    inv = [np.argsort(pm) for pm in perms]
    nb = [len(s) for s in starts]
    alice_par = [unpack_bits(h, nb[q]) for q, h in enumerate(init["par"])]
    bob_par = [_block_parities(key, perms[q], starts[q], ends[q]) for q in range(P)]
    leaked = int(sum(nb))
    rounds = 0
    n_corrected = 0

    for p in range(P):
        while True:
            oq, ob = [], []
            for q in range(p + 1):
                bad = np.nonzero(bob_par[q] != alice_par[q])[0]
                oq.extend([q] * len(bad))
                ob.extend(bad.tolist())
            if not oq:
                break
            qa = np.asarray(oq, dtype=np.int64)
            lo = np.asarray([starts[q][b] for q, b in zip(oq, ob)], dtype=np.int64)
            hi = np.asarray([ends[q][b] for q, b in zip(oq, ob)], dtype=np.int64)
            pref = np.stack([_prefix_parity(key, perms[q]) for q in range(p + 1)])

            # BINARY em paralelo: invariante = paridade de [lo,hi) difere de Alice
            while True:
                act = np.nonzero(hi - lo > 1)[0]
                if len(act) == 0:
                    break
                mid = (lo[act] + hi[act]) // 2
                link.send({"q": qa[act].tolist(), "a": lo[act].tolist(),
                           "b": mid.tolist()})
                resp = link.recv()
                a_bits = unpack_bits(resp["bits"], len(act))
                b_bits = pref[qa[act], mid] ^ pref[qa[act], lo[act]]
                left_bad = a_bits != b_bits          # erro (ímpar) na metade esquerda
                hi[act] = np.where(left_bad, mid, hi[act])
                lo[act] = np.where(left_bad, lo[act], mid)
                leaked += len(act)
                rounds += 1

            pos = np.unique([perms[q][l] for q, l in zip(qa.tolist(), lo.tolist())])
            key[pos] ^= 1
            n_corrected += len(pos)
            for q in range(P):                       # atualiza paridades de todas as passagens
                blk = inv[q][pos] // sizes[q]
                bob_par[q] ^= (np.bincount(blk, minlength=nb[q]) & 1).astype(np.uint8)

    link.send({"done": True})
    return {"key": key, "leaked": leaked, "rounds": rounds,
            "n_corrected": n_corrected, "sizes": sizes}


# =============================================================================
# 5. LDPC RATE-ADAPTIVE (reconciliação por síndrome, Bob decodifica por BP)
# =============================================================================
# Escada de eficiências. A 1ª tentativa revela ~f0*n*h(q) bits; cada falha
# "des-punça" bits (Alice revela seus valores) até chegar à eficiência
# seguinte. Só o estado FINAL importa para o vazamento (m - p_final).
LDPC_EFF_LADDER = (1.20, 1.25, 1.30, 1.36, 1.45)
# Distribuição de graus dos nós de variável (fração de nós por grau).
LDPC_VAR_DEGREES = {3: 0.80, 15: 0.20}
LDPC_BIG_LLR = 25.0


def build_ldpc(n: int, m: int, seed: int, degrees: Optional[Dict[int, float]] = None,
               n_tail: int = 0):
    """
    Matriz de verificação de paridade H (m x n) esparsa, irregular, pseudo-
    aleatória (modelo de configuração), determinística em `seed`. Devolve as
    listas de arestas (var_idx, chk_idx). Alice e Bob a reconstroem a partir de
    (n, m, seed), sem trafegar a matriz.
    """
    degrees = degrees or LDPC_VAR_DEGREES
    rs = np.random.RandomState(seed)
    fr = np.array(list(degrees.values()), dtype=float)
    dv = np.array(list(degrees.keys()), dtype=np.int64)
    cnt = np.floor(fr / fr.sum() * n).astype(np.int64)
    cnt[0] += n - cnt.sum()
    deg = np.sort(np.repeat(dv, cnt))
    # os `n_tail` últimos nós (bits punçados) recebem os MAIORES graus: um bit
    # sem informação de canal só é recuperado bem se tiver muitas verificações.
    deg = np.concatenate([rs.permutation(deg[:n - n_tail]), deg[n - n_tail:]])
    var_stubs = rs.permutation(np.repeat(np.arange(n), deg))
    chk_stubs = np.arange(len(var_stubs)) % m
    code = np.unique(chk_stubs.astype(np.int64) * n + var_stubs)   # remove arestas duplicadas
    return (code % n).astype(np.int64), (code // n).astype(np.int64)


def ldpc_syndrome(var_idx, chk_idx, m: int, x: np.ndarray) -> np.ndarray:
    return (np.bincount(chk_idx, weights=x[var_idx], minlength=m)
            .astype(np.int64) & 1).astype(np.uint8)


def bp_decode(var_idx, chk_idx, n: int, m: int, llr_ch: np.ndarray,
              syndrome: np.ndarray, max_iter: int = 100):
    """
    Propagação de crenças (soma-produto) no domínio tanh, vetorizada, para
    decodificação POR SÍNDROME: acha c tal que H c = syndrome, com prioris
    dadas por llr_ch (LLR>0 => bit 0). Devolve (bits, convergiu).
    """
    s_sign = 1.0 - 2.0 * syndrome[chk_idx].astype(np.float64)
    v2c = llr_ch[var_idx].astype(np.float64)
    hard = (llr_ch < 0).astype(np.uint8)
    for _ in range(max_iter):
        t = np.tanh(np.clip(v2c, -30.0, 30.0) / 2.0)
        neg = (t < 0).astype(np.float64)
        la = np.log(np.maximum(np.abs(t), 1e-15))
        S = np.bincount(chk_idx, weights=la, minlength=m)
        Nn = np.bincount(chk_idx, weights=neg, minlength=m)
        mag = np.minimum(np.exp(S[chk_idx] - la), 1.0 - 1e-12)
        par_neg = (Nn[chk_idx] - neg).astype(np.int64) & 1
        c2v = 2.0 * np.arctanh((1.0 - 2.0 * par_neg) * s_sign * mag)
        total = llr_ch + np.bincount(var_idx, weights=c2v, minlength=n)
        hard = (total < 0).astype(np.uint8)
        if not np.any(ldpc_syndrome(var_idx, chk_idx, m, hard) ^ syndrome):
            return hard, True
        v2c = total[var_idx] - c2v
    return hard, False


def _split_frames(N: int, frame_size: int) -> List[int]:
    F = max(1, int(math.ceil(N / frame_size)))
    return [N // F + (1 if i < N % F else 0) for i in range(F)]


def _ldpc_schedule(n_k: int, q: float, ladder) -> Dict:
    base = n_k * binary_entropy(q)
    leaks = [max(1, int(round(f * base))) for f in ladder]
    m = leaks[-1]
    p_list = [m - L for L in leaks]          # p_0 = d (maior) ... p_last = 0
    return {"m": m, "d": p_list[0], "p": p_list}


def ldpc_alice(key: np.ndarray, link: ClassicalLink, qber: float,
               frame_size: int = 8192, ladder=LDPC_EFF_LADDER,
               seed: Optional[int] = None) -> Dict:
    N = len(key)
    q_eff = min(max(qber, 0.005), 0.5)
    sizes = _split_frames(N, frame_size)
    seed = secrets.randbits(31) if seed is None else int(seed)
    frames, punct, H, syn = [], [], [], []
    off = 0
    for i, n_k in enumerate(sizes):
        sc = _ldpc_schedule(n_k, q_eff, ladder)
        m = sc["m"]
        n = n_k + sc["d"]
        if m >= n:
            raise PostProcessingAbort("QBER alta demais para reconciliação LDPC (m >= n).")
        vi, ci = build_ldpc(n, m, seed + i, n_tail=sc["d"])
        pb = random_bits(sc["d"])                       # bits punçados: aleatórios e secretos
        cw = np.concatenate([key[off:off + n_k], pb])
        s = ldpc_syndrome(vi, ci, m, cw)
        frames.append({"n_k": n_k, "m": m, "d": sc["d"], "seed": seed + i,
                       "p": sc["p"], "syn": pack_bits(s)})
        punct.append(pb); H.append((vi, ci, m, n)); syn.append(s)
        off += n_k
    link.send({"frames": frames, "q": q_eff})

    att = [None] * len(sizes)          # índice da tentativa em que cada quadro convergiu
    L = len(ladder)
    pending = list(range(len(sizes)))
    dropped: List[int] = []
    cur = [0] * len(sizes)
    for a in range(L):
        st = link.recv()
        failed = st["failed"]
        for i in pending:
            if i not in failed:
                att[i] = a
        if not failed:
            pending = []
            break
        if a == L - 1:
            dropped = failed
            pending = []
            break
        reveal = {}
        for i in failed:                # revela os bits punçados no intervalo [p_{a+1}, p_a)
            lo_, hi_ = frames[i]["p"][a + 1], frames[i]["p"][a]
            reveal[str(i)] = pack_bits(punct[i][lo_:hi_])
        link.send({"reveal": reveal})
        pending = failed

    kept, leak, off = [], 0, 0
    for i, n_k in enumerate(sizes):
        if i not in dropped:
            kept.append(key[off:off + n_k])
            leak += frames[i]["m"] - frames[i]["p"][att[i]]
        off += n_k
    out = np.concatenate(kept) if kept else np.zeros(0, dtype=np.uint8)
    return {"key": out, "leaked": int(leak), "rounds": len(sizes),
            "dropped_frames": len(dropped), "n_frames": len(sizes)}


def ldpc_bob(key: np.ndarray, link: ClassicalLink, ladder=LDPC_EFF_LADDER,
             max_iter: int = 100) -> Dict:
    init = link.recv()
    frames, q_eff = init["frames"], init["q"]
    L0 = math.log((1 - q_eff) / q_eff)
    st, off = [], 0
    for f in frames:
        n_k, m, d = f["n_k"], f["m"], f["d"]
        n = n_k + d
        vi, ci = build_ldpc(n, m, f["seed"], n_tail=d)
        llr = np.zeros(n)
        llr[:n_k] = L0 * (1.0 - 2.0 * key[off:off + n_k].astype(np.float64))
        st.append({"vi": vi, "ci": ci, "m": m, "n": n, "n_k": n_k, "llr": llr,
                   "syn": unpack_bits(f["syn"], m), "p": f["p"], "out": None})
        off += n_k

    L = len(ladder)
    pending = list(range(len(frames)))
    att = [None] * len(frames)
    dropped: List[int] = []
    for a in range(L):
        failed = []
        for i in pending:
            s = st[i]
            hard, ok = bp_decode(s["vi"], s["ci"], s["n"], s["m"], s["llr"],
                                 s["syn"], max_iter)
            if ok:
                s["out"] = hard[:s["n_k"]]
                att[i] = a
            else:
                failed.append(i)
        link.send({"failed": failed})
        if not failed:
            break
        if a == L - 1:
            dropped = failed
            break
        rev = link.recv()["reveal"]
        for i in failed:
            s = st[i]
            lo_, hi_ = s["p"][a + 1], s["p"][a]
            vals = unpack_bits(rev[str(i)], hi_ - lo_)
            s["llr"][s["n_k"] + lo_: s["n_k"] + hi_] = \
                LDPC_BIG_LLR * (1.0 - 2.0 * vals.astype(np.float64))
        pending = failed

    kept, leak = [], 0
    for i, s in enumerate(st):
        if i not in dropped:
            kept.append(s["out"])
            leak += s["m"] - s["p"][att[i]]
    out = np.concatenate(kept).astype(np.uint8) if kept else np.zeros(0, dtype=np.uint8)
    return {"key": out, "leaked": int(leak), "rounds": len(frames),
            "dropped_frames": len(dropped), "n_frames": len(frames)}


# =============================================================================
# 6/7. HASH DE TOEPLITZ (2-universal) -- verificação de EC e privacy amplification
# =============================================================================
def toeplitz_hash(seed_bits: np.ndarray, x: np.ndarray, m: int) -> np.ndarray:
    """
    y = T x (mod 2), T de dimensão m x n, T[i,j] = seed[i - j + n - 1]
    (n + m - 1 bits de semente). Uma matriz de Toeplitz aleatória forma uma
    família 2-universal. Calculado de forma direta e vetorizada via indexação
    2D (broadcasting) e redução XOR em GF(2).
    """
    n = len(x)
    seed_bits = np.asarray(seed_bits, dtype=np.uint8)
    x = np.asarray(x, dtype=np.uint8)
    
    # 1. Cria a matriz de índices 2D de dimensão (m, n)
    # Linha i contém: [i + n - 1, i + n - 2, ..., i]
    idx = np.arange(m)[:, None] + np.arange(n - 1, -1, -1)
    
    # 2. Extrai toda a matriz de Toeplitz T (m x n)
    T = seed_bits[idx]
    
    # 3. Operação AND bit a bit com o vetor x (broadcasted) e redução XOR ao longo do eixo 1
    return np.bitwise_xor.reduce(T & x, axis=1)


def verify_alice(key: np.ndarray, link: ClassicalLink, eps_cor: float) -> Dict:
    """Confirma que as chaves ficaram idênticas (PDF, seção 4.2.2): Alice envia
    a função de hash (semente) e a saída de ceil(log2(1/eps_cor)) bits."""
    m = int(math.ceil(math.log2(1.0 / eps_cor)))
    seed = random_bits(len(key) + m - 1)
    tag = toeplitz_hash(seed, key, m)
    link.send({"m": m, "seed": pack_bits(seed), "tag": pack_bits(tag)})
    ok = bool(link.recv()["ok"])
    return {"ok": ok, "leaked": m}


def verify_bob(key: np.ndarray, link: ClassicalLink) -> Dict:
    msg = link.recv()
    m = msg["m"]
    seed = unpack_bits(msg["seed"], len(key) + m - 1)
    tag = unpack_bits(msg["tag"], m)
    ok = bool(np.array_equal(toeplitz_hash(seed, key, m), tag))
    link.send({"ok": ok})
    return {"ok": ok, "leaked": m}


def secret_key_length(n: int, e_phase: float, leaked_bits: int, eps_pa: float) -> int:
    """
    Comprimento seguro da chave final (PDF, Lema 4.9 / eq. 4.44):

        l = floor( H_min - leak_EC - leak_verif  + 2 - 2 log2(1/eps_pa) )

    com H_min ~ n (1 - h2(e_phase)), e_phase = limite superior do erro de
    fase. Em BBM92/BB84 o erro de fase é limitado pelo erro de bit
    (estimado na amostra) somado ao mu de Serfling.
    """
    e = min(max(e_phase, 0.0), 0.5)
    h_min = n * (1.0 - binary_entropy(e)) if e > 0 else float(n)
    l = math.floor(h_min - leaked_bits + 2.0 - 2.0 * math.log2(1.0 / eps_pa))
    return max(0, int(l))


# =============================================================================
# 8. AUTENTICAÇÃO WEGMAN-CARTER
# =============================================================================
_P127 = (1 << 127) - 1
_TAG_BITS = 127
AUTH_BITS_PER_SESSION = 3 * _TAG_BITS      # r (chave do hash) + OTP de Alice + OTP de Bob


def _poly_hash(data: bytes, r: int) -> int:
    """Hash polinomial sobre GF(2^127-1): h = sum c_i r^(L-i+1) (Horner).
    Família epsilon-quase-universal, eps <= L/p (L = nº de blocos); o
    comprimento entra como 1º bloco e cada bloco leva um byte marcador
    0x01, o que torna a codificação injetiva."""
    data = len(data).to_bytes(8, "big") + data
    acc = 0
    for i in range(0, len(data), 14):
        c = int.from_bytes(data[i:i + 14] + b"\x01", "little")   # < 2^120 < p
        acc = ((acc + c) * r) % _P127
    return acc


def wc_tag(data: bytes, r_bits: np.ndarray, otp_bits: np.ndarray) -> str:
    """Tag Wegman-Carter: (hash polinomial de `data` com chave r) XOR (OTP)."""
    r = bits_to_int(r_bits) % (_P127 - 1) + 1
    return format(_poly_hash(data, r) ^ bits_to_int(otp_bits), "032x")


class AuthKeyPool:
    """
    Reservatório de chave SECRETA PRÉ-COMPARTILHADA para autenticação.
    Cada bit é usado UMA única vez (one-time pad); a posição já consumida é
    persistida em disco (autosave) para nunca reutilizar bits após reiniciar.
    Alice e Bob precisam começar com o MESMO arquivo, distribuído
    previamente por canal seguro (não pela rede). Ao fim de cada execução
    bem-sucedida, parte da chave gerada é devolvida ao reservatório
    (`add`), que é como o QKD "se autossustenta".
    """

    def __init__(self, bits, path: Optional[str] = None):
        self._bits = as_bits(bits)
        self._pos = 0
        self.path = path

    @classmethod
    def generate(cls, n_bits: int, path: Optional[str] = None) -> "AuthKeyPool":
        pool = cls(random_bits(n_bits), path)
        if path:
            pool.save()
        return pool

    @classmethod
    def load(cls, path: str) -> "AuthKeyPool":
        with open(path, "r") as fh:
            d = json.load(fh)
        pool = cls(unpack_bits(d["bits"], d["n"]), path)
        pool._pos = int(d["pos"])
        return pool

    def save(self, path: Optional[str] = None) -> None:
        path = path or self.path
        if not path:
            return
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"n": len(self._bits), "pos": self._pos,
                       "bits": pack_bits(self._bits)}, fh)
        os.replace(tmp, path)

    def remaining(self) -> int:
        return len(self._bits) - self._pos

    def take(self, n: int) -> np.ndarray:
        if n > self.remaining():
            raise AuthKeyExhausted(
                f"Chave de autenticação insuficiente: pedidos {n} bits, "
                f"restam {self.remaining()}. Recarregue o reservatório.")
        out = self._bits[self._pos:self._pos + n].copy()
        self._pos += n
        self.save()
        return out

    def add(self, bits) -> None:
        self._bits = np.concatenate([self._bits[self._pos:], as_bits(bits)])
        self._pos = 0
        self.save()


# =============================================================================
# Resultados
# =============================================================================
@dataclass
class PEResult:
    key: np.ndarray
    qber: float
    qber_upper: float
    mu: float
    n_sample: int
    n_errors: int
    n_remaining: int


@dataclass
class ECResult:
    key: np.ndarray
    leaked_bits: int
    method: str
    n_in: int
    n_out: int
    efficiency: float          # leak_EC / (n h2(qber)); 1.0 = limite de Shannon
    rounds: int
    verified: bool
    info: Dict = field(default_factory=dict)


@dataclass
class PAResult:
    key: np.ndarray
    n_in: int
    n_out: int
    qber_upper: float
    leaked_bits: int


@dataclass
class AuthResult:
    verified: bool
    key_bits_consumed: int
    n_messages: int
    transcript_bytes: int


@dataclass
class PostProcessResult:
    final_key: np.ndarray
    qber: float
    qber_upper: float
    authenticated: bool
    pe: PEResult
    ec: ECResult
    pa: PAResult
    auth: AuthResult
    replenished_bits: int = 0


# =============================================================================
# CLI: gera o arquivo de chave pré-compartilhada de autenticação
#   python AUXILIARY.py --gen-auth-key auth_key.json 8192
# (copie o MESMO arquivo, por meio seguro, para as máquinas de Alice e Bob)
# =============================================================================
if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--gen-auth-key":
        AuthKeyPool.generate(int(sys.argv[3]), sys.argv[2])
        print(f"Chave de autenticação de {sys.argv[3]} bits gravada em {sys.argv[2]}. "
              f"Copie o MESMO arquivo para Alice e Bob por um meio seguro.")
    else:
        print(__doc__)
