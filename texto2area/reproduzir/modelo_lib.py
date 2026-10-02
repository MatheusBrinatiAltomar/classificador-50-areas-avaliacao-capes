"""
Tokenizador usado no TREINO (analyzer do TfidfVectorizer).

A entrada é texto já pré-processado no formato do corpus (`lemmas_ext`:
lemas + n-gramas unidos por '_', separados por espaço, minúsculas).
Split por espaço + remoção de stopwords.

Este módulo precisa estar importável ao carregar o `vetorizador.joblib` gerado
por `treinar_e_salvar.py` (o joblib guarda a referência à função). O pacote
final NÃO depende dele: `_construir_artefatos.py` troca o analyzer por
`texto2area._vectorizer_analyzer.split_tokens` (equivalente, pois as stopwords
não entram no vocabulário).
"""
from stopwords_lib import carregar_stopwords

_SW = None


def tokenizar(texto):
    global _SW
    if _SW is None:
        _SW = carregar_stopwords()
    return [t for t in texto.split() if t not in _SW]
