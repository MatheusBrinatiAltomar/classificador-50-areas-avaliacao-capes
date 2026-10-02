"""Passo 4 — contagem de bigramas e trigramas sobre o corpus lematizado (2 passadas).

Réplica de ngramas_corpus_pt.py da origem (Bloco 0 / DEC-022), memory-safe:
  passada 1: contagens globais (unigramas, bigramas, trigramas) e retidos (freq >= 50);
  passada 2: contagem por grande área só dos retidos.
Métricas por n-grama: freq, PMI, NPMI, LLR (bigramas), qtd_areas, area_dominante,
share_dominante, e_lixo. Ordenação: bigramas por LLR desc; trigramas por PMI desc.

Entrada: dados/interim/03_lemas_pt.parquet (+ 03_lemas_cauda.parquet, se existir)
Saídas:  dados/interim/04_bigramas.csv, 04_trigramas.csv
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from _comum import LEMAS_PT, LEMAS_CAUDA, BIGRAMAS, TRIGRAMAS, NGRAM_FREQ_MIN, log, exigir
from ngramas_lib import gerar_ngramas, ngrama_e_lixo, calcular_llr_bigrama

COLS = ["ngrama", "freq", "pmi", "npmi", "llr", "qtd_areas", "area_dominante", "share_dominante", "e_lixo"]


def carregar_lemas() -> pd.DataFrame:
    """pt + cauda (nessa ordem), só lemmas e grande_area."""
    exigir(LEMAS_PT, "rode corpus/lematizar.py")
    dfs = [pd.read_parquet(LEMAS_PT, columns=["lemmas", "grande_area"])]
    if LEMAS_CAUDA.exists():
        dfs.append(pd.read_parquet(LEMAS_CAUDA, columns=["lemmas", "grande_area"]))
    else:
        log("AVISO: cauda traduzida ausente; n-gramas só sobre os docs em pt")
    df = pd.concat(dfs, ignore_index=True)
    return df[df.lemmas.notna() & (df.lemmas.str.len() > 0)]


def _write(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)


def main() -> None:
    freq_min = NGRAM_FREQ_MIN
    df = carregar_lemas()
    lemas = df.lemmas.tolist(); areas = df.grande_area.tolist(); del df
    intern = sys.intern
    log(f"{len(lemas):,} docs | freq_min={freq_min} | passada 1")
    unig = Counter(); big = Counter(); trig = Counter()
    N_unig = N_big = N_trig = 0
    for lem in lemas:
        toks = [intern(t) for t in lem.split()]
        if not toks:
            continue
        unig.update(toks); N_unig += len(toks)
        for b in gerar_ngramas(toks, 2):
            big[b] += 1; N_big += 1
        for t in gerar_ngramas(toks, 3):
            trig[t] += 1; N_trig += 1
    log(f"  unig {len(unig):,} | big {len(big):,} | trig {len(trig):,}")
    ret_big = {k for k, c in big.items() if c >= freq_min}
    ret_trig = {k for k, c in trig.items() if c >= freq_min}
    big = {k: big[k] for k in ret_big}; trig = {k: trig[k] for k in ret_trig}
    log(f"  retidos freq>={freq_min}: big {len(ret_big):,} | trig {len(ret_trig):,} | passada 2")

    big_areas = defaultdict(Counter); trig_areas = defaultdict(Counter)
    for lem, area in zip(lemas, areas):
        toks = lem.split()
        if not toks:
            continue
        for b in gerar_ngramas(toks, 2):
            if b in ret_big:
                big_areas[b][area] += 1
        for t in gerar_ngramas(toks, 3):
            if t in ret_trig:
                trig_areas[t][area] += 1

    log2 = math.log2
    rows_b = []
    for (x, y), c in big.items():
        cx, cy = unig[x], unig[y]
        if not cx or not cy:
            continue
        pxy = c / N_big; px = cx / N_unig; py = cy / N_unig
        if pxy <= 0 or px * py <= 0:
            continue
        ng = f"{x} {y}"; ac = big_areas[(x, y)]; total_a = sum(ac.values())
        dom_area, dom_cnt = ac.most_common(1)[0]
        rows_b.append({"ngrama": ng, "freq": c,
                       "pmi": round(log2(pxy / (px * py)), 4),
                       "npmi": round(log2(pxy / (px * py)) / -log2(pxy), 4) if pxy < 1 else 0,
                       "llr": round(calcular_llr_bigrama(c, cx, cy, N_unig), 2),
                       "qtd_areas": len(ac), "area_dominante": dom_area,
                       "share_dominante": round(dom_cnt / total_a, 4),
                       "e_lixo": int(ngrama_e_lixo(ng))})
    rows_b.sort(key=lambda r: (-r["llr"], r["ngrama"]))
    _write(BIGRAMAS, rows_b)
    log(f"  bigramas {len(rows_b):,} (lixo {sum(r['e_lixo'] for r in rows_b):,}) -> {BIGRAMAS.name}")

    rows_t = []
    for (x, y, z), c in trig.items():
        cx, cy, cz = unig[x], unig[y], unig[z]
        if not (cx and cy and cz):
            continue
        pxyz = c / N_trig; px = cx / N_unig; py = cy / N_unig; pz = cz / N_unig
        if pxyz <= 0 or px * py * pz <= 0:
            continue
        ng = f"{x} {y} {z}"; ac = trig_areas[(x, y, z)]; total_a = sum(ac.values())
        dom_area, dom_cnt = ac.most_common(1)[0]
        rows_t.append({"ngrama": ng, "freq": c,
                       "pmi": round(log2(pxyz / (px * py * pz)), 4),
                       "npmi": round(log2(pxyz / (px * py * pz)) / -log2(pxyz), 4) if pxyz < 1 else 0,
                       "llr": "", "qtd_areas": len(ac), "area_dominante": dom_area,
                       "share_dominante": round(dom_cnt / total_a, 4),
                       "e_lixo": int(ngrama_e_lixo(ng))})
    rows_t.sort(key=lambda r: (-r["pmi"], r["ngrama"]))
    _write(TRIGRAMAS, rows_t)
    log(f"  trigramas {len(rows_t):,} (lixo {sum(r['e_lixo'] for r in rows_t):,}) -> {TRIGRAMAS.name}")


if __name__ == "__main__":
    main()
