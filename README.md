# qkd-detector-processing
[![DOI](https://zenodo.org/badge/1400179561.svg)](https://doi.org/10.5281/zenodo.23243500)


Interface de processamento de dados dos detectores para enlace de **Distribuição Quântica de Chaves (QKD)** baseado no protocolo BBM92

Mais detalhes em ```documentation.txt```

## Integração com gerenciadores de chave (qpass)

`QKD_NODE.py` concentra a sessão de um nó (`run_session`) e expõe uma CLI com
saída JSON para gerenciadores de chave externos. `ALICE.py` e `BOB.py` são
invólucros finos sobre ela (mesmas variáveis de ambiente; `SIMULATE=1` liga o
hardware simulado).

```bash
python QKD_NODE.py auth-init --auth-key-file auth_key.json --bits 8192
python QKD_NODE.py session --role alice --capture-s 2 --json | gerenciador
python QKD_NODE.py session --role bob --peer-host 192.168.1.10 --capture-s 2 --json | gerenciador
python QKD_NODE.py auth-status --auth-key-file auth_key.json
```

stdout recebe uma única linha JSON (`status`, `key_id`, `key_bits`, `key_hex`,
QBER, EC, autenticação...); os logs vão para stderr. O `key_id` é derivado da
transcrição autenticada e é igual nos dois nós. Códigos de saída: 0 chave,
3 chave vazia, 2 abortado, 1 erro.
