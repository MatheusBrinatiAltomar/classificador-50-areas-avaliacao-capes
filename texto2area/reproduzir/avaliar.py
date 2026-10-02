"""
Avaliação do classificador (mesmos protocolos usados nos artigos).

  (A) A_aleatorio    split estratificado 80/20                — desempenho in-distribution
  (B) B_out_of_time  treino ano <= 2023, teste ano == 2024    — validação preditiva real

Métricas: acurácia, F1-macro, F1-weighted, baseline (classe majoritária);
por classe: precision/recall/F1/support; matriz de confusão normalizada por
linha; termos mais discriminativos por classe (coeficientes do SVM).

Entrada : dados/corpus_td_lemas.parquet
Saída   : reproduzir/avaliacao/<rotulo>/avaliacao_{resumo,report_X,confusao_X,top_features_X}.csv

Uso:
  python reproduzir/avaliar.py                                  # 9 grandes áreas, corpus completo
  python reproduzir/avaliar.py --rotulo area_avaliacao --amostra 200000
  python reproduzir/avaliar.py --rotulo area_avaliacao --mapa-rotulos de_para.csv

Valores de referência (grande_area, corpus completo, min_df=50):
  A_aleatorio    acurácia 0,792  F1-macro 0,791
  B_out_of_time  acurácia 0,729  F1-macro 0,732
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modelo_lib import tokenizar          # noqa: E402
from treinar_e_salvar import CORPUS, COL, SEED, MIN_DF, carregar_corpus  # noqa: E402

N_TOP_FEAT = 15


def salvar_report(report: dict, caminho: Path) -> None:
    linhas = [{"classe": c, "precision": round(m["precision"], 4), "recall": round(m["recall"], 4),
               "f1": round(m["f1-score"], 4), "support": int(m["support"])}
              for c, m in report.items() if isinstance(m, dict)]
    pd.DataFrame(linhas).to_csv(caminho, index=False, encoding="utf-8")


def salvar_confusao(y_true, y_pred, labels, caminho: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    df = pd.DataFrame(cm_norm, index=labels, columns=labels).round(4)
    df.index.name = "verdadeiro\\previsto"
    df.to_csv(caminho, encoding="utf-8")


def top_features_por_classe(clf, vec) -> pd.DataFrame:
    """Termos de maior coeficiente por classe. Usa clf.classes_ (as classes VISTAS no treino):
    no protocolo B uma classe pode existir só no teste e não ter linha em coef_."""
    termos = np.array(vec.get_feature_names_out())
    classes = list(clf.classes_)
    coef = clf.coef_ if len(classes) > 2 else np.vstack([-clf.coef_[0], clf.coef_[0]])
    return pd.DataFrame([{"classe": c, "top_termos": " | ".join(termos[np.argsort(coef[i])[::-1][:N_TOP_FEAT]])}
                         for i, c in enumerate(classes)])


def avaliar(nome, X_tr, y_tr, X_te, y_te, vec, labels, out: Path) -> dict:
    print(f"\n===== PROTOCOLO {nome} =====")
    print(f"  treino: {X_tr.shape[0]:,} | teste: {X_te.shape[0]:,} | features: {X_tr.shape[1]:,}")
    clf = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED)
    clf.fit(X_tr, y_tr)
    sem_treino = sorted(set(labels) - set(clf.classes_))
    if sem_treino:
        print(f"  AVISO: {len(sem_treino)} classe(s) sem documento no treino deste protocolo "
              f"(F1 = 0 por construção): {sem_treino}")
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    f1w = f1_score(y_te, y_pred, average="weighted")
    maj = pd.Series(y_tr).value_counts().idxmax()
    base = accuracy_score(y_te, [maj] * len(y_te))
    print(f"  acurácia={acc:.4f} | F1-macro={f1m:.4f} | F1-weighted={f1w:.4f} | baseline(maj)={base:.4f}")

    report = classification_report(y_te, y_pred, labels=labels, output_dict=True, zero_division=0)
    salvar_report(report, out / f"avaliacao_report_{nome}.csv")
    salvar_confusao(y_te, y_pred, labels, out / f"avaliacao_confusao_{nome}.csv")
    top_features_por_classe(clf, vec).to_csv(out / f"avaliacao_top_features_{nome}.csv",
                                                     index=False, encoding="utf-8")
    return {"protocolo": nome, "n_treino": X_tr.shape[0], "n_teste": X_te.shape[0],
            "n_features": X_tr.shape[1], "n_classes": len(labels), "acuracia": round(acc, 4),
            "f1_macro": round(f1m, 4), "f1_weighted": round(f1w, 4), "baseline_majoritaria": round(base, 4)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rotulo", default="grande_area")
    ap.add_argument("--amostra", type=int, default=None)
    ap.add_argument("--min-df", type=int, default=MIN_DF)
    ap.add_argument("--mapa-rotulos", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or (HERE / "avaliacao" / args.rotulo)
    out.mkdir(parents=True, exist_ok=True)

    df = carregar_corpus(args.rotulo, args.amostra, args.mapa_rotulos, extra=["ano_publicacao"])
    # classes com menos de 2 docs inviabilizam o split estratificado: ficam fora da avaliação
    cont = df[args.rotulo].value_counts()
    raras = sorted(cont[cont < 2].index)
    if raras:
        print(f"  AVISO: {len(raras)} classe(s) com <2 documentos excluída(s) da avaliação: {raras}")
        df = df[~df[args.rotulo].isin(raras)].reset_index(drop=True)
    y_all = df[args.rotulo].values
    labels = sorted(pd.unique(y_all))
    print(f"{len(df):,} documentos | rótulo={args.rotulo} | classes={len(labels)} | saída: {out}")
    resumo = []

    # (A) split estratificado 80/20
    idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20, random_state=SEED, stratify=y_all)
    vecA = TfidfVectorizer(analyzer=tokenizar, min_df=args.min_df, sublinear_tf=True)
    XA_tr = vecA.fit_transform(df[COL].values[idx_tr]); XA_te = vecA.transform(df[COL].values[idx_te])
    resumo.append(avaliar("A_aleatorio", XA_tr, y_all[idx_tr], XA_te, y_all[idx_te], vecA, labels, out))

    # (B) out-of-time
    ano = df["ano_publicacao"].astype(int).values
    m_tr, m_te = ano <= 2023, ano == 2024
    vecB = TfidfVectorizer(analyzer=tokenizar, min_df=args.min_df, sublinear_tf=True)
    XB_tr = vecB.fit_transform(df[COL].values[m_tr]); XB_te = vecB.transform(df[COL].values[m_te])
    resumo.append(avaliar("B_out_of_time", XB_tr, y_all[m_tr], XB_te, y_all[m_te], vecB, labels, out))

    pd.DataFrame(resumo).to_csv(out / "avaliacao_resumo.csv", index=False, encoding="utf-8")
    print(f"\nSalvos em {out}/avaliacao_*.csv")
    print(pd.DataFrame(resumo).to_string(index=False))


if __name__ == "__main__":
    main()
