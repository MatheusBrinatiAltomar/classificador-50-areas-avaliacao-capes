"""Passo 3 — lematização (spaCy pt_core_news_lg) dos documentos em português.

Replica a etapa 3 da origem (lematizar_chunked.py, trilha traduzir=False):
  - só docs com idioma == 'pt' (os demais vão ao passo da cauda);
  - texto em minúsculas; lemas de conteúdo (POS NOUN/PROPN/ADJ/VERB, len>=2, com letra);
  - processamento em chunks com partes incrementais em dados/interim/_lemas_parts/,
    RETOMÁVEL (partes prontas são puladas se o processo cair).

Custo: ~1h para 1M docs com 4 processos (N_PROCESS) em máquina de 32 GB.
Entrada: dados/interim/02_normalizado.parquet
Saída:   dados/interim/03_lemas_pt.parquet  (metadados + lemmas + n_lemas)
"""
from __future__ import annotations

import time

import pandas as pd

from _comum import (NORMALIZADO, LEMAS_PARTS, LEMAS_PT, META_COLS, CHUNK, BATCH, N_PROCESS,
                    filtrar_lemas, carregar_spacy, iter_parquet, log, exigir)

OUT_COLS = META_COLS + ["idioma", "n_palavras", "n_lemas", "lemmas"]


def main() -> None:
    exigir(NORMALIZADO, "rode corpus/idioma.py")
    LEMAS_PARTS.mkdir(parents=True, exist_ok=True)
    nlp = carregar_spacy()
    log(f"chunk={CHUNK} batch={BATCH} n_process={N_PROCESS}")
    t0 = time.time(); lidos = 0
    for ci, chunk in enumerate(iter_parquet(NORMALIZADO, batch_size=CHUNK)):
        part = LEMAS_PARTS / f"part_{ci:04d}.parquet"
        lidos += len(chunk)
        if part.exists():
            continue
        sub = chunk[chunk["idioma"] == "pt"].copy()
        if sub.empty:
            pd.DataFrame(columns=OUT_COLS).to_parquet(part, index=False); continue
        textos = sub["texto"].fillna("").str.lower().tolist()
        lemas = [filtrar_lemas(d) for d in nlp.pipe(textos, batch_size=BATCH, n_process=N_PROCESS)]
        sub["lemmas"] = [" ".join(l) for l in lemas]
        sub["n_lemas"] = [len(l) for l in lemas]
        sub[OUT_COLS].to_parquet(part, index=False, compression="snappy")
        log(f"  chunk {ci} ({len(sub):,} pt) | {lidos:,} lidos | {lidos/(time.time()-t0):,.0f} lin/s")
    log("concatenando partes...")
    full = pd.concat([pd.read_parquet(p) for p in sorted(LEMAS_PARTS.glob("part_*.parquet"))],
                     ignore_index=True)
    full.to_parquet(LEMAS_PT, index=False, compression="zstd")
    log(f"OK {len(full):,} docs pt | lemas {int(full.n_lemas.sum()):,} | média {full.n_lemas.mean():.1f} -> {LEMAS_PT}")


if __name__ == "__main__":
    main()
