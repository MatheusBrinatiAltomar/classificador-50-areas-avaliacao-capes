"""Passo 5 — injeção dos n-gramas retidos e montagem do corpus final.

Réplica de injetar_ngramas.py da origem (DEC-022):
  retidos = bigramas não-lixo, freq >= 50, LLR >= mediana desses, fora da lista
            de n-gramas-stopword  +  trigramas não-lixo, freq >= 50, fora da lista;
  lemmas_ext = unigramas originais + token unido ('_') de cada n-grama retido casado
               (anexados ao fim, em ordem de ocorrência: primeiro bigramas, depois trigramas).

Monta o parquet final com o esquema documentado em dados/README.md (11 colunas).
Ordem das linhas: docs pt (ordem do banco) e depois a cauda traduzida.

Entradas: 03_lemas_pt.parquet, 03_lemas_cauda.parquet (opcional), 04_bigramas.csv, 04_trigramas.csv
Saída:    --out (padrão dados/corpus_td_lemas.parquet; recusa sobrescrever sem --sobrescrever)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from _comum import (LEMAS_PT, LEMAS_CAUDA, BIGRAMAS, TRIGRAMAS, VOCAB, FINAL_PADRAO,
                    STOPWORDS_NGRAMAS, NGRAM_FREQ_MIN, LLR_QUANTIL, log, exigir)
from ngramas_lib import gerar_ngramas, juntar_token

FINAL_COLS = ["id_producao", "id_programa", "chave_canonica", "grande_area", "area_avaliacao",
              "ano_publicacao", "quadrienio", "idioma_original", "n_palavras", "lemmas_ext", "n_lemas_ext"]
SCHEMA = pa.schema([
    ("id_producao", pa.string()), ("id_programa", pa.string()), ("chave_canonica", pa.string()),
    ("grande_area", pa.string()), ("area_avaliacao", pa.string()),
    ("ano_publicacao", pa.int16()), ("quadrienio", pa.string()), ("idioma_original", pa.string()),
    ("n_palavras", pa.int32()), ("lemmas_ext", pa.large_string()), ("n_lemas_ext", pa.int32()),
])


def carregar_retidos() -> set[tuple]:
    sw = {l.strip() for l in STOPWORDS_NGRAMAS.read_text(encoding="utf-8").splitlines()
          if l.strip() and not l.startswith("#")}
    retidos = set()
    db = pd.read_csv(BIGRAMAS)
    db = db[(db.e_lixo == 0) & (db.freq >= NGRAM_FREQ_MIN)]
    llr_thr = db.llr.quantile(LLR_QUANTIL)
    db = db[db.llr >= llr_thr]
    log(f"  bigramas: corte LLR(q{LLR_QUANTIL})={llr_thr:.1f}; candidatos {len(db):,}")
    for ng in db.ngrama.astype(str):
        if juntar_token(ng) not in sw:
            retidos.add(tuple(ng.split()))
    dt = pd.read_csv(TRIGRAMAS)
    dt = dt[(dt.e_lixo == 0) & (dt.freq >= NGRAM_FREQ_MIN)]
    log(f"  trigramas: candidatos {len(dt):,}")
    for ng in dt.ngrama.astype(str):
        if juntar_token(ng) not in sw:
            retidos.add(tuple(ng.split()))
    return retidos


def injetar(tokens: list[str], retidos: set[tuple]) -> list[str]:
    extra = []
    for n in (2, 3):
        for ng in gerar_ngramas(tokens, n):
            if ng in retidos:
                extra.append(juntar_token(ng))
    return tokens + extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=FINAL_PADRAO)
    ap.add_argument("--sobrescrever", action="store_true")
    args = ap.parse_args()
    if args.out.exists() and not args.sobrescrever:
        sys.exit(f"{args.out} já existe. Use --out outro caminho ou --sobrescrever.")
    for p, d in ((LEMAS_PT, "corpus/lematizar.py"), (BIGRAMAS, "corpus/ngramas.py"), (TRIGRAMAS, "corpus/ngramas.py")):
        exigir(p, f"rode {d}")

    retidos = carregar_retidos()
    log(f"n-gramas retidos para injeção: {len(retidos):,}")

    dfs = [pd.read_parquet(LEMAS_PT)]
    dfs[0]["idioma_original"] = dfs[0]["idioma"]
    if LEMAS_CAUDA.exists():
        dfs.append(pd.read_parquet(LEMAS_CAUDA))
    else:
        log("AVISO: cauda traduzida ausente; corpus final só com docs em pt")
    df = pd.concat(dfs, ignore_index=True)

    novos = []
    for lem in df["lemmas"].fillna("").tolist():
        toks = lem.split()
        novos.append(" ".join(injetar(toks, retidos)) if toks else "")
    df["lemmas_ext"] = novos
    df["n_lemas_ext"] = [len(s.split()) for s in novos]
    log(f"docs com >=1 n-grama injetado: {sum(1 for s in novos if '_' in s) / len(novos) * 100:.1f}%")

    voc = Counter()
    for s in novos:
        voc.update(s.split())
    vd = pd.DataFrame(sorted(voc.items()), columns=["token", "freq"])
    vd["e_ngrama"] = vd.token.str.contains("_").astype(int)
    vd.to_csv(VOCAB, index=False)
    log(f"vocabulário: {len(vd):,} tokens ({int(vd.e_ngrama.sum()):,} n-gramas) -> {VOCAB.name}")

    df["area_avaliacao"] = df["area_avaliacao"].str.strip()
    t = pa.Table.from_pandas(df[FINAL_COLS], schema=SCHEMA, preserve_index=False)
    t = t.replace_schema_metadata({
        b"dataset": b"texto2area/corpus_td_lemas",
        b"origem": b"Catalogo de Teses e Dissertacoes CAPES (Sucupira), 2013-2024; construido por texto2area/corpus/",
        b"texto": b"lemmas_ext = lemas (NOUN/PROPN/ADJ/VERB, spaCy pt_core_news_lg) de titulo+resumo + n-gramas retidos unidos por '_'",
        b"documentacao": b"README.md na pasta dados/ do repositorio texto2area",
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(t, args.out, compression="zstd", compression_level=9, row_group_size=100_000)
    h = hashlib.sha256()
    with args.out.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    args.out.with_suffix(".sha256").write_text(f"{h.hexdigest()}  {args.out.name}\n")
    log(f"OK {len(df):,} docs -> {args.out} ({args.out.stat().st_size/1e6:,.0f} MB) sha256={h.hexdigest()[:16]}...")


if __name__ == "__main__":
    main()
