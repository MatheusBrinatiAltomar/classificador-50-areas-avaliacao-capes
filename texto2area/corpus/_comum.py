"""Configuração e utilitários compartilhados pela construção do corpus.

Caminhos: tudo fica em <repo>/dados/ (interim/ para intermediários, modelos/ para
modelos baixados). Constantes replicam o pipeline de origem (Tarrafa / Camada 1a).
"""
from __future__ import annotations

import os
import re
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DADOS = Path(os.environ.get("TEXTO2AREA_DADOS", REPO / "dados"))  # override p/ testes
INTERIM = DADOS / "interim"
MODELOS = DADOS / "modelos"
STOPWORDS_NGRAMAS = REPO / "reproduzir" / "stopwords" / "academicas_v2_ngramas.txt"

# artefatos intermediários (ordem do pipeline)
BRUTO = INTERIM / "01_bruto.parquet"              # extração do banco
NORMALIZADO = INTERIM / "02_normalizado.parquet"  # + idioma (langid), texto NFC, n_palavras
LEMAS_PARTS = INTERIM / "_lemas_parts"            # partes da lematização (retomável)
LEMAS_PT = INTERIM / "03_lemas_pt.parquet"        # docs em pt lematizados
LEMAS_CAUDA = INTERIM / "03_lemas_cauda.parquet"  # docs en/es traduzidos e lematizados
BIGRAMAS = INTERIM / "04_bigramas.csv"
TRIGRAMAS = INTERIM / "04_trigramas.csv"
VOCAB = INTERIM / "04_vocabulario.csv"
FINAL_PADRAO = DADOS / "corpus_td_lemas.parquet"

# parâmetros do pipeline (não mudar sem re-treinar tudo)
ANO_INI, ANO_FIM = 2013, 2024
POS_MANTER = {"NOUN", "PROPN", "ADJ", "VERB"}
SPACY_MODELO = "pt_core_news_lg"
IDIOMAS = ["pt", "en", "es"]          # langid restrito (DEC-006 da origem)
NGRAM_FREQ_MIN = 50                   # piso de frequência de bi/trigramas (TD)
LLR_QUANTIL = 0.50                    # corte de LLR para bigramas
CHUNK = int(os.environ.get("CHUNK", "40000"))
BATCH = int(os.environ.get("BATCH", "64"))
N_PROCESS = int(os.environ.get("N_PROCESS", "4"))

META_COLS = ["id_producao", "id_programa", "chave_canonica", "grande_area", "area_avaliacao",
             "ano_publicacao", "quadrienio", "idioma_sucupira"]

_RX_CONTROLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RX_ESPACOS = re.compile(r"\s+")


def normalizar_nfc(texto: str) -> str:
    """NFC + remove controles + colapsa espaços + strip (etapa 2 da origem)."""
    if not texto:
        return ""
    t = unicodedata.normalize("NFC", texto)
    t = _RX_CONTROLE.sub(" ", t)
    t = _RX_ESPACOS.sub(" ", t)
    return t.strip()


def texto_para_idioma(texto: str) -> str:
    """Forma usada na detecção de idioma e na tradução: espaços colapsados + minúsculas."""
    return _RX_ESPACOS.sub(" ", texto or "").strip().lower()


def filtrar_lemas(doc) -> list[str]:
    """Lemas de conteúdo: POS em POS_MANTER, minúsculos, len>=2, com letra."""
    out = []
    for t in doc:
        if t.pos_ not in POS_MANTER:
            continue
        l = t.lemma_.lower().strip()
        if len(l) >= 2 and any(c.isalpha() for c in l):
            out.append(l)
    return out


def carregar_spacy():
    import spacy
    try:
        return spacy.load(SPACY_MODELO, exclude=["parser", "ner", "senter"])
    except OSError:
        sys.exit(f"Modelo spaCy '{SPACY_MODELO}' ausente. Rode: python -m spacy download {SPACY_MODELO}")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def exigir(path: Path, dica: str) -> None:
    if not path.exists():
        sys.exit(f"Arquivo ausente: {path}\n  -> {dica}")


def iter_parquet(path: Path, columns=None, batch_size=CHUNK):
    """Itera um parquet em DataFrames de batch_size linhas (nunca carrega tudo)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield b.to_pandas()
