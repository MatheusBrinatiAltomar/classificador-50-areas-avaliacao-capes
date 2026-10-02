"""Compara um corpus reconstruído com o de referência (mesmo esquema), por id_producao.

Relata: documentos em comum e exclusivos de cada lado; igualdade por coluna;
para lemmas_ext, igualdade exata, igualdade como multiconjunto (mesmos tokens em
outra ordem) e exemplos das diferenças. É o teste de aceitação de uma reconstrução.

Uso: python corpus/comparar.py NOVO.parquet [REFERENCIA.parquet] [--so-unigramas]
  --so-unigramas: ignora os tokens com "_" (útil ao comparar uma amostra, cujos n-gramas
                  retidos diferem dos do corpus inteiro).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from _comum import FINAL_PADRAO

COLS_META = ["id_programa", "chave_canonica", "grande_area", "area_avaliacao",
             "ano_publicacao", "quadrienio", "idioma_original", "n_palavras", "n_lemas_ext"]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("novo", type=Path); ap.add_argument("referencia", type=Path, nargs="?", default=FINAL_PADRAO)
    ap.add_argument("--so-unigramas", action="store_true")
    a_ = ap.parse_args(); novo, ref, so_uni = a_.novo, a_.referencia, a_.so_unigramas
    a = pd.read_parquet(novo).set_index("id_producao")
    b = pd.read_parquet(ref).set_index("id_producao")
    comum = a.index.intersection(b.index)
    print(f"novo: {len(a):,} docs | referência: {len(b):,} | em comum: {len(comum):,}")
    print(f"só no novo: {len(a.index.difference(b.index)):,} | só na referência: {len(b.index.difference(a.index)):,}")
    a, b = a.loc[comum].copy(), b.loc[comum].copy()
    if so_uni:
        for d in (a, b):
            d["lemmas_ext"] = d["lemmas_ext"].map(lambda s: " ".join(t for t in s.split() if "_" not in t))
        print("(comparando só unigramas de lemmas_ext)")
    print("\nigualdade por coluna (docs em comum):")
    for c in COLS_META:
        ig = (a[c].astype(str).values == b[c].astype(str).values).mean()
        print(f"  {c:16s} {ig*100:7.3f}%")
    ex = (a["lemmas_ext"].values == b["lemmas_ext"].values)
    print(f"  {'lemmas_ext':16s} {ex.mean()*100:7.3f}%  (exata)")
    dif = [i for i, e in enumerate(ex) if not e]
    if dif:
        mult = sum(Counter(a["lemmas_ext"].values[i].split()) == Counter(b["lemmas_ext"].values[i].split()) for i in dif)
        print(f"  {'':16s} {(ex.sum()+mult)/len(ex)*100:7.3f}%  (como multiconjunto de tokens)")
        print(f"\n{len(dif):,} docs diferentes em lemmas_ext; exemplos:")
        for i in dif[:5]:
            ta, tb = Counter(a["lemmas_ext"].values[i].split()), Counter(b["lemmas_ext"].values[i].split())
            so_a = list((ta - tb).elements())[:8]; so_b = list((tb - ta).elements())[:8]
            print(f"  id={comum[i]} idioma_ref={b['idioma_original'].values[i]} | só novo: {so_a} | só ref: {so_b}")
        idi = Counter(b["idioma_original"].values[i] for i in dif)
        print(f"  diferenças por idioma_original (ref): {dict(idi)}")


if __name__ == "__main__":
    main()
