"""Carrega as stopwords usadas no treino (mesmas listas da Camada 1a, DEC-022).

Três listas, unidas:
  classicas_universais.txt   — stopwords clássicas (pt/en/es + símbolos);
  academicas_v2.txt          — unigramas acadêmicos genéricos (ex.: 'abordagem', 'objetivo');
  academicas_v2_ngramas.txt  — n-gramas genéricos cross-área, unidos por '_'.
"""
from pathlib import Path

STOPWORDS_DIR = Path(__file__).resolve().parent / "stopwords"
ARQUIVOS = ["classicas_universais.txt", "academicas_v2.txt", "academicas_v2_ngramas.txt"]


def carregar_stopwords() -> set[str]:
    sw: set[str] = set()
    for nome in ARQUIVOS:
        p = STOPWORDS_DIR / nome
        sw |= {w.strip().lower() for w in p.read_text(encoding="utf-8").splitlines()
               if w.strip() and not w.startswith("#")}
    return sw
