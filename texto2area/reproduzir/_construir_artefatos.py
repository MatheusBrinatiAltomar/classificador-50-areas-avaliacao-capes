"""
Converte um modelo treinado (reproduzir/modelo_treinado/<rotulo>/) nos artefatos
do pacote (texto2area/data/):

  - troca o analyzer do vetorizador (modelo_lib.tokenizar -> texto2area split),
    tornando o .joblib autocontido, sem depender de reproduzir/;
  - copia modelo.joblib e classes.json.

Verifica que a troca de analyzer NÃO altera o transform (as stopwords não estão
no vocabulário, logo split == split+stopwords para os termos do vocabulário).

Tamanho: os coeficientes do LinearSVC são convertidos para float32 (metade do tamanho;
um modelo de 50 classes cai de ~144 MB para ~72 MB, abaixo do limite de 100 MB do PyPI).
A conversão é verificada em uma amostra do corpus: as previsões precisam ser idênticas
e a maior diferença na decision_function é impressa. Use --float64 para desligar.

Uso:  python reproduzir/_construir_artefatos.py [--rotulo grande_area] [--origem PASTA] [--dest texto2area/data] [--float64]
      --origem: pasta do modelo treinado (padrão reproduzir/modelo_treinado/<rotulo>; use se treinou com --out)
"""
import argparse
import shutil
import sys
from pathlib import Path

import joblib

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))   # modelo_lib (analyzer do vetorizador treinado)
sys.path.insert(0, str(REPO))   # texto2area._vectorizer_analyzer

import texto2area._vectorizer_analyzer as va  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--rotulo", default="grande_area")
ap.add_argument("--origem", type=Path, default=None, help="pasta com vetorizador.joblib/modelo.joblib/classes.json")
ap.add_argument("--dest", type=Path, default=REPO / "texto2area" / "data")
ap.add_argument("--float64", action="store_true", help="mantém coef_/intercept_ em float64 (padrão: float32)")
ap.add_argument("--n-verif", type=int, default=20000, help="docs do corpus usados para verificar a conversão")
args = ap.parse_args()
origem = args.origem or (HERE / "modelo_treinado" / args.rotulo)
args.dest.mkdir(parents=True, exist_ok=True)

print(f"Carregando {origem / 'vetorizador.joblib'} ...", flush=True)
vec = joblib.load(origem / "vetorizador.joblib")
amostra = "elemento finito simulacao trinca concreto_armado estrutura conclusao trabalho"
X_old = vec.transform([amostra])
vec.analyzer = va.split_tokens
X_new = vec.transform([amostra])
assert (X_old != X_new).nnz == 0, "troca de analyzer alterou o transform!"
print("  OK: troca de analyzer preserva o transform.", flush=True)

import json
import numpy as np
treino = json.loads((origem / "treino.json").read_text()) if (origem / "treino.json").exists() else {}

# ---- float32: metade do tamanho, mesmas previsões (verificado abaixo) ----
clf = joblib.load(origem / "modelo.joblib")
if not args.float64 and clf.coef_.dtype != np.float32:
    corpus = REPO / "dados" / "corpus_td_lemas.parquet"
    if not corpus.exists():
        raise SystemExit(f"{corpus} ausente: necessário para verificar a conversão (ou use --float64)")
    import pandas as pd
    amostra = pd.read_parquet(corpus, columns=["lemmas_ext"]).sample(n=args.n_verif, random_state=42)["lemmas_ext"].values
    Xv = vec.transform(amostra)
    pred64, dec64 = clf.predict(Xv), clf.decision_function(Xv)
    clf.coef_ = clf.coef_.astype(np.float32)
    clf.intercept_ = clf.intercept_.astype(np.float32)
    pred32, dec32 = clf.predict(Xv), clf.decision_function(Xv)
    iguais = float((pred64 == pred32).mean())
    print(f"  float32: previsões idênticas em {iguais*100:.3f}% de {len(amostra):,} docs; "
          f"maior diferença na decision_function = {np.abs(dec64 - dec32).max():.2e}")
    if iguais < 0.9999:
        raise SystemExit("conversão para float32 alterou previsões acima do tolerável; use --float64")
else:
    print(f"  coeficientes mantidos em {clf.coef_.dtype}")
reaproveitado = treino.get("vetorizador_reaproveitado")
if reaproveitado:
    # modelo treinado sobre um vetorizador já existente (ex.: o do pacote): não duplicar 12 MB no wheel
    print(f"  vetorizador reaproveitado de {reaproveitado}: NÃO copiado (use o já existente em texto2area/data/)")
else:
    joblib.dump(vec, args.dest / "vetorizador.joblib")
joblib.dump(clf, args.dest / "modelo.joblib")
shutil.copy(origem / "classes.json", args.dest / "classes.json")
print(f"  modelo.joblib: {(args.dest / 'modelo.joblib').stat().st_size/1e6:,.1f} MB")
print(f"Artefatos em {args.dest}: {'' if reaproveitado else 'vetorizador.joblib, '}modelo.joblib, classes.json", flush=True)
