"""
POSTPROCESS.py
=============

Pós-processamento clássico de QKD, INDEPENDENTE DE NÓ: Alice e Bob chamam as
mesmas funções, com o mesmo roteiro, e o papel de cada um vem de
`link.role` ('alice' ou 'bob'), onde `link` é um `ClassicalLink` (AUXILIARY.py) que
encapsula o canal clássico PERSISTENTE (porta única) de CLASSIC_CHANNEL.py.

Canal único: ALICE.py / BOB.py abrem UMA conexão TCP
(`AliceClassicalChannel` / `BobClassicalChannel`), a usam no sifting e a
entregam, via `ClassicalLink(role, channel=...)`, a este módulo, que continua
na MESMA conexão em todas as etapas abaixo. Este módulo nunca abre nem fecha
a conexão -- isso é responsabilidade de quem criou o canal.

A etapa de SIFTING é específica de cada protocolo QKD (aqui, BBM92) e por isso
fica em ALICE.py / BOB.py. Este módulo recebe a chave PENEIRADA (bits 0/1 já
alinhados entre Alice e Bob) e entrega a chave secreta final:

    chave peneirada
        |  parameter_estimation      amostra aleatória -> QBER + limite de Serfling
        v
    chave "limpa" (sem a amostra)
        |  error_correction_cascade | error_correction_LDPC     (padrão: Cascade)
        v                          + verificação por hash 2-universal
    chaves idênticas (com `leaked_bits` bits revelados ao público)
        |  privacy_amp               hash de Toeplitz (Leftover Hash Lemma)
        v
    chave final secreta
        |  authentication            tag Wegman-Carter sobre TODA a transcrição
        v
    chave aceita (ou AuthenticationError -> descartar)

Funções principais
------------------
    parameter_estimation(key, link, ...)              -> PEResult
    error_correction_cascade(key, qber, link, ...)    -> ECResult
    error_correction_LDPC(key, qber, link, ...)       -> ECResult
    privacy_amp(key, link, qber_upper, leaked_bits)   -> PAResult
    authentication(link, auth_pool)                   -> AuthResult
    run_postprocessing(key, link, auth_pool, ...)     -> PostProcessResult   (orquestrador)

Regras de uso
-------------
* Alice e Bob DEVEM chamar as mesmas funções, na mesma ordem: cada função
  faz trocas em lockstep pelo canal clássico (uma chamada de um lado casa com
  a chamada correspondente do outro). Como a conexão é única e persistente,
  qualquer divergência de roteiro faz as mensagens ficarem defasadas ou
  trava a execução.
* O `link` deve ser usado ENQUANTO o canal estiver aberto (dentro do bloco
  `with ...ClassicalChannel(...) as channel:` de ALICE.py/BOB.py).
* Exceções `PostProcessingAbort` e `AuthenticationError` são levantadas de
  forma consistente nos dois nós; nesse caso NENHUMA chave deve ser usada.
"""

from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np

from AUXILIARY import (
    AUTH_BITS_PER_SESSION, LDPC_EFF_LADDER,
    AuthenticationError, AuthKeyPool, AuthResult, ClassicalLink, ECResult,
    PAResult, PEResult, PostProcessResult, PostProcessingAbort,
    as_bits, binary_entropy, cascade_alice, cascade_bob, ldpc_alice, ldpc_bob,
    log, pack_bits, random_bits, secret_key_length, serfling_mu,
    toeplitz_hash, unpack_bits, verify_alice, verify_bob, wc_tag,
)

import secrets

_DEFAULT_QBER_MAX = 0.11      # mesmo limiar de ALICE.py/BOB.py (limite do BB84/BBM92 ~ 11%)


# =============================================================================
# 1. ESTIMAÇÃO DE PARÂMETROS
# =============================================================================
def parameter_estimation(key, link: ClassicalLink, sample_fraction: float = 0.20,
                         qber_threshold: float = _DEFAULT_QBER_MAX,
                         eps_pe: float = 2.5e-10, verbose: bool = True) -> PEResult:
    """
    Estima a QBER numa amostra aleatória da chave peneirada (PDF, seção 4.2.1).

    Alice sorteia `sample_fraction` das posições (com `secrets`, sem
    reposição -- requisito do teorema de Serfling), revela posições e bits;
    Bob conta os erros e devolve o número. A amostra é DESCARTADA da chave
    (bits revelados não são mais secretos).

    Diferente de usar "os primeiros 20%", a amostra é aleatória: é isso que
    permite extrapolar a taxa de erro da amostra para o restante da chave.

    Saída
    -----
    PEResult.qber        taxa de erro observada na amostra (e_k)
    PEResult.qber_upper  e_k + mu, com  Pr[e_resto >= e_k + mu] <= eps_pe
                         (Serfling, eq. 4.34).
    PEResult.key         chave sem a amostra

    Aborta (PostProcessingAbort) se e_k > qber_threshold.
    """
    key = as_bits(key)
    N = len(key)
    role = link.role
    if role == "alice":
        k = int(round(N * sample_fraction))
        if k < 1 or N - k < 1:
            link.send({"n": N, "idx": [], "bits": ""})
            link.recv()
            raise PostProcessingAbort(
                f"Chave peneirada pequena demais ({N} bits) para estimar a QBER.")
        rng = np.random.RandomState(secrets.randbits(32))
        idx = np.sort(rng.choice(N, size=k, replace=False))
        link.send({"n": N, "idx": idx.tolist(), "bits": pack_bits(key[idx])})
        reply = link.recv()
        if "abort" in reply:
            raise PostProcessingAbort(reply["abort"])
        errors = int(reply["errors"])
    else:
        msg = link.recv()
        idx = np.asarray(msg["idx"], dtype=np.int64)
        k = len(idx)
        if msg["n"] != N:
            link.send({"abort": f"Comprimentos diferentes: Alice={msg['n']}, Bob={N}"})
            raise PostProcessingAbort(
                f"Chaves peneiradas com comprimentos diferentes (Alice={msg['n']}, Bob={N}).")
        if k < 1 or N - k < 1:
            link.send({"abort": "amostra vazia"})
            raise PostProcessingAbort(
                f"Chave peneirada pequena demais ({N} bits) para estimar a QBER.")
        errors = int(np.sum(key[idx] != unpack_bits(msg["bits"], k)))
        link.send({"errors": errors})

    n_rest = N - k
    qber = errors / k
    mu = serfling_mu(n_rest, k, eps_pe)
    qber_upper = min(0.5, qber + mu)
    mask = np.ones(N, dtype=bool)
    mask[idx] = False
    rest = key[mask]
    log(role, f"[PE] amostra={k} bits, erros={errors}, QBER={qber*100:.2f}% "
              f"(limite superior {qber_upper*100:.2f}%, mu={mu:.4f}); "
              f"restam {n_rest} bits.", verbose)
    if qber > qber_threshold:
        raise PostProcessingAbort(
            f"QBER={qber*100:.2f}% acima do limite de {qber_threshold*100:.1f}%: "
            f"possível espionagem/ruído excessivo. Protocolo abortado.")
    return PEResult(key=rest, qber=qber, qber_upper=qber_upper, mu=mu,
                    n_sample=k, n_errors=errors, n_remaining=n_rest)


# =============================================================================
# 2. CORREÇÃO DE ERROS
# =============================================================================
def _finish_ec(key_in, res, qber, link, method, eps_cor, verbose) -> ECResult:
    """Verificação por hash 2-universal (PDF, seção 4.2.2) + estatísticas."""
    key_out = as_bits(res["key"])
    leak = int(res["leaked"])
    n_in = len(key_in)
    if len(key_out) > 0:
        v = verify_alice(key_out, link, eps_cor) if link.role == "alice" \
            else verify_bob(key_out, link)
        leak += v["leaked"]
        if not v["ok"]:
            raise PostProcessingAbort(
                "Verificação da correção de erros falhou: as chaves continuam "
                "diferentes após a reconciliação. Protocolo abortado.")
    # eficiência f = vazamento / (n h2(QBER)); indefinida se a QBER observada é 0
    eff = (res["leaked"] / (n_in * binary_entropy(qber))) if (n_in and qber > 0) else float("nan")
    out = ECResult(key=key_out, leaked_bits=leak, method=method, n_in=n_in,
                   n_out=len(key_out), efficiency=eff, rounds=res["rounds"],
                   verified=True,
                   info={k: v for k, v in res.items() if k not in ("key",)})
    log(link.role, f"[EC-{method}] {n_in} -> {len(key_out)} bits | vazados={leak} "
                   f"| eficiência f={eff:.3f} | rodadas={res['rounds']} "
                   f"| verificação OK", verbose)
    return out


def error_correction_cascade(key, qber: float, link: ClassicalLink,
                             n_passes: int = 4, k1: Optional[int] = None,
                             eps_cor: float = 1e-10, seed: Optional[int] = None,
                             verbose: bool = True) -> ECResult:
    """
    Correção de erros pelo protocolo CASCADE (Brassard & Salvail, 1993).

    Alice (referência) nunca altera sua chave; Bob a altera até coincidir.
    Em cada passagem as chaves são embaralhadas (permutação pública), divididas
    em blocos e as paridades comparadas; blocos com paridade diferente sofrem
    busca binária (BINARY) que localiza e corrige um erro; cada correção
    "cascateia" para os blocos de passagens anteriores cuja paridade mudou.

    Parâmetros
    ----------
    qber      QBER estimada em `parameter_estimation` (define o bloco inicial
              k1 = 0.73/QBER e k_i = 2 k_{i-1})
    n_passes  nº de passagens (4 = Cascade original; mais passagens reduzem a
              taxa de erro residual ao custo de mais paridades reveladas)
    k1        força o tamanho do 1º bloco
    eps_cor   probabilidade máxima de as chaves ainda diferirem após a
              verificação (hash de ceil(log2(1/eps_cor)) bits)

    Devolve ECResult com `leaked_bits` = paridades reveladas + bits do hash de
    verificação -- esse valor é subtraído pela privacy_amp.
    """
    key = as_bits(key)
    if link.role == "alice":
        res = cascade_alice(key, link, qber, n_passes, k1, seed)
    else:
        res = cascade_bob(key, link)
    return _finish_ec(key, res, qber, link, "cascade", eps_cor, verbose)


def error_correction_LDPC(key, qber: float, link: ClassicalLink,
                          frame_size: int = 8192, ladder=LDPC_EFF_LADDER,
                          max_iter: int = 100, eps_cor: float = 1e-10,
                          seed: Optional[int] = None,
                          verbose: bool = True) -> ECResult:
    """
    Correção de erros por LDPC rate-adaptive com reconciliação por SÍNDROME
    (Elkouss, Leverrier, Alléaume, Boutros, ISIT 2009; Elkouss, Martinez,
    Martin, QIC 2011).

    * A chave é dividida em quadros de ~`frame_size` bits.
    * Para cada quadro, Alice anexa `d` bits ALEATÓRIOS "punçados" (secretos,
      desconhecidos de Bob), calcula a síndrome s = H c e a envia; Bob
      decodifica por propagação de crenças (LLR=0 nos bits punçados) e recupera
      a palavra de Alice.
    * Adaptação de taxa: a 1ª tentativa usa o menor vazamento
      (eficiência f0 = ladder[0]); se o BP não converge, Alice REVELA parte dos
      bits punçados (que passam a ser "encurtados"), subindo para a próxima
      eficiência da escada. O vazamento final é m - p_final (p = punçados
      restantes).
    * Quadros que falham mesmo na última eficiência são DESCARTADOS (nos dois
      nós) -- a chave sai menor, mas nunca inconsistente.
    * Ao final, a verificação por hash 2-universal confirma a igualdade.

    A matriz H (irregular, pseudo-aleatória) é regenerada dos dois lados a
    partir de (n, m, semente); nenhuma matriz trafega pela rede.
    """
    key = as_bits(key)
    if link.role == "alice":
        res = ldpc_alice(key, link, qber, frame_size, ladder, seed)
    else:
        res = ldpc_bob(key, link, ladder, max_iter)
    return _finish_ec(key, res, qber, link, "ldpc", eps_cor, verbose)


def error_correction(key, qber: float, link: ClassicalLink, method: str = "cascade",
                     **kwargs) -> ECResult:
    """Despacha para Cascade (padrão) ou LDPC."""
    method = method.strip().lower()
    if method == "cascade":
        return error_correction_cascade(key, qber, link, **kwargs)
    if method == "ldpc":
        return error_correction_LDPC(key, qber, link, **kwargs)
    raise ValueError("method deve ser 'cascade' ou 'ldpc'")


# =============================================================================
# 3. AMPLIFICAÇÃO DE PRIVACIDADE
# =============================================================================
def privacy_amp(key, link: ClassicalLink, qber_upper: float, leaked_bits: int,
                eps_sec: float = 1e-9, verbose: bool = True) -> PAResult:
    """
    Amplificação de privacidade por hash de Toeplitz (família 2-universal),
    com o comprimento dado pelo Quantum Leftover Hash Lemma (PDF, Lema 4.9):

        l = floor( n (1 - h2(e_fase)) - leaked_bits + 2 - 2 log2(1/eps_pa) )

    * e_fase = `qber_upper` (QBER da amostra + mu de Serfling). Em BBM92 o erro
      de fase nas bases X/Z é estimado pelo erro de bit.
    * leaked_bits = tudo que o canal público revelou na correção de erros
      (paridades/síndromes + hash de verificação).
    * eps_sec = parâmetro de segurança total; usa-se eps_pa = eps_sec/2 (termo
      do Leftover Hash Lemma) e 2*eps_pe = eps_sec/2 (suavização/estimação).

    Alice sorteia a semente (bits do CSPRNG), envia a Bob junto com l, e os dois
    aplicam a mesma matriz de Toeplitz `l x n` à chave corrigida. Se l = 0 não
    há chave segura a extrair (típico com chaves curtas: o custo de tamanho
    finito -2 log2(1/eps) e o mu de Serfling consomem tudo).
    """
    key = as_bits(key)
    n = len(key)
    eps_pa = eps_sec / 2.0
    l = secret_key_length(n, qber_upper, int(leaked_bits), eps_pa) if n > 0 else 0
    if link.role == "alice":
        if l > 0:
            seed = random_bits(n + l - 1)
            link.send({"l": l, "seed": pack_bits(seed)})
            out = toeplitz_hash(seed, key, l)
        else:
            link.send({"l": 0})
            out = np.zeros(0, dtype=np.uint8)
    else:
        msg = link.recv()
        if int(msg["l"]) != l:
            raise PostProcessingAbort(
                f"Comprimento final divergente entre nós (Alice={msg['l']}, Bob={l}).")
        out = (toeplitz_hash(unpack_bits(msg["seed"], n + l - 1), key, l)
               if l > 0 else np.zeros(0, dtype=np.uint8))
    log(link.role, f"[PA] n={n}, e_fase<={qber_upper*100:.2f}%, vazado={leaked_bits} "
                   f"-> chave final = {l} bits (eps_sec={eps_sec:g})", verbose)
    return PAResult(key=out, n_in=n, n_out=l, qber_upper=qber_upper,
                    leaked_bits=int(leaked_bits))


# =============================================================================
# 4. AUTENTICAÇÃO
# =============================================================================
def authentication(link: ClassicalLink, auth_pool: AuthKeyPool,
                   raise_on_failure: bool = True, verbose: bool = True) -> AuthResult:
    """
    Autentica TODO o tráfego clássico da sessão (sifting, estimação, correção
    de erros, privacy amplification) com um MAC de Wegman-Carter, seguro em
    sentido teórico-informacional:

        tag = PolyHash_r(transcrição) XOR OTP

    * PolyHash: hash polinomial sobre GF(2^127-1), probabilidade de forjar
      <= L/p (L = nº de blocos de 14 bytes da transcrição; ~2^-100 na prática);
    * r, OTP_Alice e OTP_Bob (127 bits cada) vêm da chave pré-compartilhada
      (`auth_pool`) e são USADOS UMA ÚNICA VEZ; cada direção tem seu próprio
      OTP e um rótulo de domínio ('A'/'B') para impedir ataques de reflexão.
    * Alice e Bob trocam suas tags numa única troca e cada um verifica a do
      outro contra a transcrição que ELE registrou. Qualquer mensagem
      alterada, removida, inserida ou reordenada muda a transcrição e falha.

    Deve ser chamada por último: só depois dela a chave final pode ser usada.
    Consome AUTH_BITS_PER_SESSION (=381) bits do reservatório.
    """
    bits = auth_pool.take(AUTH_BITS_PER_SESSION)
    r_bits, otp_a, otp_b = bits[:127], bits[127:254], bits[254:]
    data = link.transcript_bytes()
    n_msgs = len(link.transcript)

    mine = wc_tag(b"A" + data, r_bits, otp_a) if link.role == "alice" \
        else wc_tag(b"B" + data, r_bits, otp_b)
    remote = json.loads(link.exchange(json.dumps({"tag": mine})))["tag"]
    expected = wc_tag(b"B" + data, r_bits, otp_b) if link.role == "alice" \
        else wc_tag(b"A" + data, r_bits, otp_a)
    ok = secrets.compare_digest(remote, expected)
    log(link.role, f"[AUTH] {n_msgs} mensagens / {len(data)} bytes autenticados: "
                   f"{'OK' if ok else 'FALHOU'} (consumiu {AUTH_BITS_PER_SESSION} bits de chave; "
                   f"restam {auth_pool.remaining()})", verbose)
    if not ok and raise_on_failure:
        raise AuthenticationError(
            "Autenticação da transcrição falhou: o canal clássico foi adulterado "
            "(ou as chaves de autenticação de Alice e Bob estão dessincronizadas). "
            "Descarte a chave.")
    return AuthResult(verified=ok, key_bits_consumed=AUTH_BITS_PER_SESSION,
                      n_messages=n_msgs, transcript_bytes=len(data))


# =============================================================================
# 5. ORQUESTRADOR
# =============================================================================
def run_postprocessing(sifted_key, link: ClassicalLink, auth_pool: AuthKeyPool,
                       ec_method: str = "cascade", sample_fraction: float = 0.20,
                       qber_threshold: float = _DEFAULT_QBER_MAX,
                       eps_sec: float = 1e-9, eps_cor: float = 1e-10,
                       replenish_auth: bool = True,
                       ec_kwargs: Optional[dict] = None,
                       verbose: bool = True) -> PostProcessResult:
    """
    Executa a cadeia completa (PE -> EC -> PA -> AUTH) -- é o que ALICE.py e
    BOB.py chamam logo após o sifting. `ec_method`: 'cascade' (padrão) ou
    'ldpc'.

    Se `replenish_auth=True`, ao final (e só se a autenticação passou) uma
    parte da chave final, do tamanho do que foi consumido, volta ao
    reservatório de autenticação.
    """
    ec_kwargs = dict(ec_kwargs or {})
    pe = parameter_estimation(sifted_key, link, sample_fraction, qber_threshold,
                              eps_pe=eps_sec / 4.0,
                              verbose=verbose)
    ec = error_correction(pe.key, pe.qber, link, method=ec_method,
                          eps_cor=eps_cor, verbose=verbose, **ec_kwargs)
    pa = privacy_amp(ec.key, link, pe.qber_upper, ec.leaked_bits,
                     eps_sec=eps_sec, verbose=verbose)
    auth = authentication(link, auth_pool, raise_on_failure=True, verbose=verbose)

    final_key = pa.key
    replenished = 0
    if replenish_auth and len(final_key) > AUTH_BITS_PER_SESSION:
        replenished = AUTH_BITS_PER_SESSION
        auth_pool.add(final_key[-replenished:])
        final_key = final_key[:-replenished]
        log(link.role, f"[AUTH] {replenished} bits da chave final devolvidos ao "
                       f"reservatório de autenticação.", verbose)
    return PostProcessResult(final_key=final_key, qber=pe.qber,
                             qber_upper=pe.qber_upper, authenticated=auth.verified,
                             pe=pe, ec=ec, pa=pa, auth=auth,
                             replenished_bits=replenished)
