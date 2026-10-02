"""
Treina o classificador no corpus completo e salva os artefatos do modelo.

Entrada : dados/corpus_td_lemas.parquet   (ver dados/README.md)
Saída   : reproduzir/modelo_treinado/<rotulo>/
            vetorizador.joblib   TF-IDF (analyzer = modelo_lib.tokenizar)
            modelo.joblib        LinearSVC one-vs-rest, class_weight='balanced'
            classes.json         ordem das classes
            treino.json          parâmetros e contagens do treino

Determinístico (SEED=42). O modelo publicado no pacote foi gerado com
`--rotulo grande_area` (padrão) sobre 100% dos dados.

Uso:
  python reproduzir/treinar_e_salvar.py                          # 9 grandes áreas
  python reproduzir/treinar_e_salvar.py --rotulo area_avaliacao  # áreas de avaliação
  python reproduzir/treinar_e_salvar.py --amostra 200000         # subamostra p/ testes rápidos
  python reproduzir/treinar_e_salvar.py --mapa-rotulos dados/de_para_area_avaliacao.csv  # aplica um de-para
  python reproduzir/treinar_e_salvar.py --rotulo area_avaliacao --vetorizador texto2area/data/vetorizador.joblib
      # reaproveita o TF-IDF do modelo publicado (só transform); útil p/ compartilhar o vetorizador entre cabeças
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modelo_lib import tokenizar  # noqa: E402

REPO = HERE.parent
CORPUS = REPO / "dados" / "corpus_td_lemas.parquet"
SEED, MIN_DF, COL = 42, 50, "lemmas_ext"


def carregar_corpus(rotulo: str, amostra: int | None, mapa: Path | None,
                    extra: list[str] | None = None) -> pd.DataFrame:
    if not CORPUS.exists():
        sys.exit(f"Corpus não encontrado: {CORPUS}\nVeja dados/README.md para obtê-lo.")
    df = pd.read_parquet(CORPUS, columns=[rotulo, COL] + (extra or [])).dropna(subset=[rotulo, COL])
    df = df[df[COL].str.strip() != ""]
    if mapa is not None:
        m = pd.read_csv(mapa)                       # colunas: de, para
        de_para = dict(zip(m["de"], m["para"]))
        faltam = sorted(set(df[rotulo]) - set(de_para))
        if faltam:
            sys.exit(f"Rótulos sem entrada no de-para ({len(faltam)}): {faltam[:5]} ...")
        df[rotulo] = df[rotulo].map(de_para)
        df = df.dropna(subset=[rotulo])             # 'para' vazio = descartar
    if amostra:
        df = df.sample(n=min(amostra, len(df)), random_state=SEED)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotulo", default="grande_area", help="coluna de rótulo (grande_area | area_avaliacao)")
    ap.add_argument("--amostra", type=int, default=None, help="treina numa subamostra aleatória (p/ testes)")
    ap.add_argument("--min-df", type=int, default=MIN_DF)
    ap.add_argument("--mapa-rotulos", type=Path, default=None, help="CSV com colunas de,para (de-para de rótulos)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--vetorizador", type=Path, default=None,
                    help="vetorizador .joblib já ajustado: não refaz o fit do TF-IDF (ignora --min-df)")
    args = ap.parse_args()
    out = args.out or (HERE / "modelo_treinado" / args.rotulo)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = carregar_corpus(args.rotulo, args.amostra, args.mapa_rotulos)
    print(f"{len(df):,} documentos | rótulo={args.rotulo} | classes={df[args.rotulo].nunique()}", flush=True)

    if args.vetorizador:
        sys.path.insert(0, str(REPO))          # texto2area._vectorizer_analyzer (analyzer do pacote)
        vec = joblib.load(args.vetorizador)
        print(f"Vetorizando com {args.vetorizador} (sem fit) ...", flush=True)
        X = vec.transform(df[COL].values)
    else:
        vec = TfidfVectorizer(analyzer=tokenizar, min_df=args.min_df, sublinear_tf=True)
        print(f"Vetorizando (min_df={args.min_df}) ...", flush=True)
        X = vec.fit_transform(df[COL].values)
    y = df[args.rotulo].values
    print(f"  features: {X.shape[1]:,}; treinando LinearSVC ...", flush=True)
    clf = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED)
    clf.fit(X, y)

    joblib.dump(vec, out / "vetorizador.joblib")
    joblib.dump(clf, out / "modelo.joblib")
    (out / "classes.json").write_text(json.dumps(list(clf.classes_), ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "treino.json").write_text(json.dumps({
        "rotulo": args.rotulo, "n_docs": int(len(df)), "n_classes": int(len(clf.classes_)),
        "n_features": int(X.shape[1]), "min_df": None if args.vetorizador else args.min_df, "seed": SEED,
        "amostra": args.amostra, "mapa_rotulos": str(args.mapa_rotulos) if args.mapa_rotulos else None,
        "vetorizador_reaproveitado": str(args.vetorizador) if args.vetorizador else None,
        "corpus": CORPUS.name, "segundos": round(time.time() - t0, 1),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Salvo em {out}  ({time.time()-t0:,.0f}s)", flush=True)


if __name__ == "__main__":
    main()
