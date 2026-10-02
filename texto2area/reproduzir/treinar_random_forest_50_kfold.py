"""
Treina Random Forest para classificar teses/dissertações nas 50 áreas de
avaliação CAPES usando EXATAMENTE os mesmos folds canônicos do experimento
XGBoost.

A ideia deste script é permitir uma comparação direta entre modelos.

Arquivos reutilizados do experimento anterior:
    reproduzir/folds_capes_5.csv
    reproduzir/folds_capes_5_meta.json
    modelo_treinado/xgboost_50_kfold/
        features_xgboost.float32.dat
        features_meta.json
        y.npy
        indices_parquet.npy

Os folds são definidos pelo arquivo canônico:
    indice_parquet, fold, id_classe, classe

Para cada fold:
    4 folds -> treinamento
    1 fold  -> validação

Ao final:
    - métricas por fold
    - métricas agregadas
    - avaliação OOF (Out-of-Fold)
    - matriz de confusão OOF
    - classification report OOF
    - predições OOF
    - modelo Random Forest separado para cada fold
    - importância média das features

IMPORTANTE:
- Este script NÃO recria os folds.
- O mesmo documento terá exatamente o mesmo fold usado no XGBoost.
- A entrada X é a mesma representação já produzida para o XGBoost:
      [hash_texto | margens_9 | grande_area_9 | estatisticas_6]
- O Random Forest é executado com sklearn e CPU.
- RandomForestClassifier não possui um callback público estável para
  tqdm por árvore; por isso o progresso é mostrado por fold e por predição.
  O parâmetro verbose do sklearn pode ser ativado.

Uso:
    python reproduzir/treinar_random_forest_50_kfold.py

Com outra quantidade de árvores:
    python reproduzir/treinar_random_forest_50_kfold.py --n-estimators 500

Sem balanceamento:
    python reproduzir/treinar_random_forest_50_kfold.py --sem-pesos

Usando outro diretório de features:
    python reproduzir/treinar_random_forest_50_kfold.py \
        --features-dir modelo_treinado/xgboost_50_kfold
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm


# =============================================================================
# CAMINHOS
# =============================================================================

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

DEFAULT_FOLDS_FILE = HERE / "folds_capes_5.csv"
DEFAULT_FOLDS_META = HERE / "folds_capes_5_meta.json"

# Diretório produzido pelo script do XGBoost.
DEFAULT_FEATURES_DIR = (
    HERE
    / "modelo_treinado"
    / "xgboost_50_kfold"
)

DEFAULT_OUT = (
    HERE
    / "modelo_treinado"
    / "random_forest_50_kfold"
)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

SEED = 42
N_FOLDS = 5

BATCH_SIZE = 10_000

# Hiperparâmetros iniciais adequados para uma RF grande.
N_ESTIMATORS = 400
MAX_DEPTH = 18
MIN_SAMPLES_SPLIT = 2
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"
BOOTSTRAP = True

# "balanced" mantém um comportamento mais próximo do balanceamento
# utilizado no experimento XGBoost.
CLASS_WEIGHT = "balanced"

# verbose=0 evita o log muito extenso do sklearn.
# Use 1 para ver o progresso interno do joblib/sklearn.
RF_VERBOSE = 0


# =============================================================================
# LOG
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# VERIFICAÇÃO
# =============================================================================

def verificar_arquivos(
    folds_file: Path,
    folds_meta_file: Path,
    features_dir: Path,
) -> None:

    arquivos = {
        "Folds canônicos": folds_file,
        "Metadados dos folds": folds_meta_file,
        "Features": features_dir / "features_xgboost.float32.dat",
        "Metadados das features": features_dir / "features_meta.json",
        "y": features_dir / "y.npy",
        "Índices parquet": features_dir / "indices_parquet.npy",
    }

    faltantes = [
        f"{nome}: {path}"
        for nome, path in arquivos.items()
        if not path.exists()
    ]

    if faltantes:
        raise FileNotFoundError(
            "Arquivos necessários não encontrados:\n  "
            + "\n  ".join(faltantes)
            + "\n\n"
            "Execute primeiro o experimento XGBoost para gerar as "
            "features compartilhadas, ou informe --features-dir."
        )


# =============================================================================
# CARREGA FOLDS
# =============================================================================

def carregar_folds(
    folds_file: Path,
    folds_meta_file: Path,
    indices_parquet: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """
    Carrega os folds canônicos e os converte para índices locais de X/y.

    O arquivo canônico usa `indice_parquet`.
    X/y usam a posição dentro do conjunto de documentos válidos.

    Retorna:
        folds_local
        classes
        folds_df
    """

    log("[1/7] Carregando folds canônicos...")

    folds_df = pd.read_csv(
        folds_file,
        dtype={
            "indice_parquet": np.int64,
            "fold": np.int16,
            "id_classe": np.int32,
            "classe": str,
        },
    )

    required = {
        "indice_parquet",
        "fold",
        "id_classe",
        "classe",
    }

    if not required.issubset(folds_df.columns):
        raise ValueError(
            "O arquivo de folds precisa conter as colunas: "
            f"{sorted(required)}"
        )

    if len(folds_df) != len(indices_parquet):
        raise ValueError(
            "O número de linhas do arquivo de folds não corresponde "
            "ao número de documentos em indices_parquet.npy.\n"
            f"Folds: {len(folds_df):,}\n"
            f"Documentos: {len(indices_parquet):,}"
        )

    if folds_df["indice_parquet"].duplicated().any():
        raise ValueError(
            "O arquivo de folds contém indices_parquet duplicados."
        )

    folds_ids = sorted(
        folds_df["fold"].unique().tolist()
    )

    if folds_ids != list(range(1, N_FOLDS + 1)):
        raise ValueError(
            "Os folds precisam ser exatamente "
            f"{list(range(1, N_FOLDS + 1))}; "
            f"encontrados: {folds_ids}"
        )

    # -------------------------------------------------------------------------
    # Recupera a ordem das classes pelo id_classe.
    # -------------------------------------------------------------------------

    classes_por_id = (
        folds_df[
            ["id_classe", "classe"]
        ]
        .drop_duplicates()
        .sort_values("id_classe")
    )

    if classes_por_id["id_classe"].min() != 0:
        raise ValueError(
            "Os id_classe precisam começar em 0."
        )

    classes = classes_por_id["classe"].astype(str).tolist()

    if len(classes) != 50:
        raise ValueError(
            "Esperavam-se 50 classes; "
            f"foram encontradas {len(classes)}."
        )

    # -------------------------------------------------------------------------
    # Valida consistência classe -> id.
    # -------------------------------------------------------------------------

    mapa_classe_id = {
        str(row.classe): int(row.id_classe)
        for row in classes_por_id.itertuples()
    }

    for row in folds_df.itertuples():
        esperado = mapa_classe_id[str(row.classe)]
        if int(row.id_classe) != esperado:
            raise ValueError(
                "Inconsistência entre classe e id_classe no arquivo de folds."
            )

    # -------------------------------------------------------------------------
    # Como indices_parquet normalmente está ordenado, usamos searchsorted.
    # Isso evita um dicionário gigante com centenas de milhares de entradas.
    # -------------------------------------------------------------------------

    indices_parquet = np.asarray(
        indices_parquet,
        dtype=np.int64,
    )

    if np.any(
        indices_parquet[1:] < indices_parquet[:-1]
    ):
        raise ValueError(
            "indices_parquet.npy não está ordenado em ordem crescente."
        )

    indices_folds = folds_df[
        "indice_parquet"
    ].to_numpy(
        dtype=np.int64
    )

    pos = np.searchsorted(
        indices_parquet,
        indices_folds,
    )

    if np.any(pos >= len(indices_parquet)):
        raise ValueError(
            "Há índices do arquivo de folds que não existem "
            "em indices_parquet.npy."
        )

    if not np.array_equal(
        indices_parquet[pos],
        indices_folds,
    ):
        raise ValueError(
            "O arquivo de folds e indices_parquet.npy não correspondem "
            "ao mesmo conjunto de documentos."
        )

    # fold por posição local.
    folds_local = np.empty(
        len(indices_parquet),
        dtype=np.int16,
    )

    folds_local[pos] = folds_df[
        "fold"
    ].to_numpy(
        dtype=np.int16
    )

    # -------------------------------------------------------------------------
    # Valida y contra o arquivo de folds.
    # -------------------------------------------------------------------------

    ids_classe_folds = np.empty(
        len(indices_parquet),
        dtype=np.int32,
    )

    ids_classe_folds[pos] = folds_df[
        "id_classe"
    ].to_numpy(
        dtype=np.int32
    )

    if not np.array_equal(
        ids_classe_folds,
        y.astype(np.int32),
    ):
        raise ValueError(
            "Os rótulos de y.npy não correspondem aos id_classe "
            "do folds_capes_5.csv."
        )

    # -------------------------------------------------------------------------
    # Metadados
    # -------------------------------------------------------------------------

    if folds_meta_file.exists():

        meta = json.loads(
            folds_meta_file.read_text(
                encoding="utf-8"
            )
        )

        meta_k = int(
            meta.get("k", -1)
        )

        if meta_k != N_FOLDS:
            raise ValueError(
                "O arquivo de metadados dos folds informa "
                f"K={meta_k}, mas este script exige K={N_FOLDS}."
            )

    log(
        f"  documentos: {len(folds_local):,}"
    )

    log(
        f"  classes:    {len(classes)}"
    )

    for fold_id in range(
        1,
        N_FOLDS + 1,
    ):

        n = int(
            np.sum(
                folds_local == fold_id
            )
        )

        log(
            f"  fold {fold_id}: {n:,} documentos"
        )

    return (
        folds_local,
        classes,
        folds_df,
    )


# =============================================================================
# CARREGA FEATURES
# =============================================================================

def carregar_features(
    features_dir: Path,
) -> tuple[np.memmap, np.ndarray, np.ndarray, dict]:
    """
    Abre as features compartilhadas sem carregá-las integralmente na RAM.
    """

    log("[2/7] Carregando features compartilhadas...")

    meta_path = (
        features_dir
        / "features_meta.json"
    )

    feature_path = (
        features_dir
        / "features_xgboost.float32.dat"
    )

    y_path = (
        features_dir
        / "y.npy"
    )

    indices_path = (
        features_dir
        / "indices_parquet.npy"
    )

    meta = json.loads(
        meta_path.read_text(
            encoding="utf-8"
        )
    )

    n_docs = int(
        meta["n_docs"]
    )

    n_features = int(
        meta["n_features"]
    )

    y = np.load(
        y_path,
        mmap_mode="r",
    )

    indices_parquet = np.load(
        indices_path,
        mmap_mode="r",
    )

    if len(y) != n_docs:
        raise ValueError(
            "y.npy não possui a quantidade de documentos informada "
            "em features_meta.json."
        )

    if len(indices_parquet) != n_docs:
        raise ValueError(
            "indices_parquet.npy não possui a quantidade de documentos "
            "informada em features_meta.json."
        )

    X = np.memmap(
        feature_path,
        dtype=np.float32,
        mode="r",
        shape=(n_docs, n_features),
    )

    log(
        f"  documentos: {n_docs:,}"
    )

    log(
        f"  features:   {n_features:,}"
    )

    log(
        f"  arquivo:    {feature_path}"
    )

    return (
        X,
        np.asarray(y),
        np.asarray(indices_parquet),
        meta,
    )


# =============================================================================
# AMOSTRA
# =============================================================================

def aplicar_amostra(
    y: np.ndarray,
    indices_parquet: np.ndarray,
    folds: np.ndarray,
    amostra: int | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Faz amostragem somente após carregar os folds canônicos.

    Retorna também as posições originais escolhidas dentro de X/y.
    """

    if amostra is None:
        idx = np.arange(
            len(y),
            dtype=np.int64,
        )

        return (
            y,
            indices_parquet,
            folds,
            idx,
        )

    rng = np.random.default_rng(
        SEED
    )

    n = min(
        int(amostra),
        len(y),
    )

    idx = np.sort(
        rng.choice(
            len(y),
            size=n,
            replace=False,
        )
    )

    return (
        y[idx],
        indices_parquet[idx],
        folds[idx],
        idx,
    )


# =============================================================================
# TREINAMENTO
# =============================================================================

def criar_modelo_random_forest(
    fold_id: int,
    sem_pesos: bool,
    n_estimators: int,
    max_depth: int | None,
    min_samples_split: int,
    min_samples_leaf: int,
    max_features: str | int | float,
) -> RandomForestClassifier:

    class_weight = (
        None
        if sem_pesos
        else CLASS_WEIGHT
    )

    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=BOOTSTRAP,
        class_weight=class_weight,
        random_state=SEED + fold_id,
        n_jobs=-1,
        verbose=RF_VERBOSE,
        warm_start=False,
    )


# =============================================================================
# PREDIÇÃO EM BATCH
# =============================================================================

def prever_em_batches(
    model: RandomForestClassifier,
    X: np.memmap,
    indices: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera classe prevista e probabilidade máxima em batches.

    A probabilidade máxima é útil para análises posteriores de confiança.
    """

    pred_ids = np.empty(
        len(indices),
        dtype=np.int32,
    )

    confidence = np.empty(
        len(indices),
        dtype=np.float32,
    )

    pbar = tqdm(
        total=len(indices),
        desc="Predição",
        unit="doc",
        dynamic_ncols=True,
    )

    for inicio in range(
        0,
        len(indices),
        batch_size,
    ):

        fim = min(
            inicio + batch_size,
            len(indices),
        )

        xb = np.asarray(
            X[
                indices[inicio:fim]
            ]
        )

        prob = model.predict_proba(
            xb
        )

        pred_local = np.argmax(
            prob,
            axis=1,
        )

        pred_ids[
            inicio:fim
        ] = pred_local.astype(
            np.int32
        )

        confidence[
            inicio:fim
        ] = prob[
            np.arange(
                len(pred_local)
            ),
            pred_local,
        ].astype(
            np.float32
        )

        pbar.update(
            fim - inicio
        )

        del xb
        del prob

    pbar.close()

    return (
        pred_ids,
        confidence,
    )


# =============================================================================
# AVALIAÇÃO
# =============================================================================

def avaliar_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    fold_id: int,
    fold_dir: Path,
) -> dict:

    nomes_true = [
        classes[int(i)]
        for i in y_true
    ]

    nomes_pred = [
        classes[int(i)]
        for i in y_pred
    ]

    labels_ids = np.arange(
        len(classes),
        dtype=np.int32,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=labels_ids,
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels_ids,
        zero_division=0,
    )

    report = classification_report(
        nomes_true,
        nomes_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        fold_dir
        / "classification_report.csv",
        encoding="utf-8",
    )

    cm = confusion_matrix(
        nomes_true,
        nomes_pred,
        labels=classes,
    )

    cm_norm = (
        cm
        / np.maximum(
            cm.sum(
                axis=1,
                keepdims=True,
            ),
            1,
        )
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    ).to_csv(
        fold_dir
        / "matriz_confusao.csv",
        encoding="utf-8",
    )

    pd.DataFrame(
        cm_norm,
        index=classes,
        columns=classes,
    ).round(6).to_csv(
        fold_dir
        / "matriz_confusao_normalizada.csv",
        encoding="utf-8",
    )

    log(
        f"Fold {fold_id}: "
        f"accuracy={accuracy:.4f} | "
        f"F1-macro={f1_macro:.4f} | "
        f"F1-weighted={f1_weighted:.4f}"
    )

    return {
        "fold": int(fold_id),
        "n_valid": int(len(y_true)),
        "accuracy": round(
            float(accuracy),
            6,
        ),
        "f1_macro": round(
            float(f1_macro),
            6,
        ),
        "f1_weighted": round(
            float(f1_weighted),
            6,
        ),
    }


# =============================================================================
# AVALIAÇÃO OOF
# =============================================================================

def avaliar_oof(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    folds: np.ndarray,
    indices_parquet: np.ndarray,
    confidences: np.ndarray,
    out_dir: Path,
) -> dict:

    log("[6/7] Gerando avaliação agregada OOF...")

    nomes_true = [
        classes[int(i)]
        for i in y_true
    ]

    nomes_pred = [
        classes[int(i)]
        for i in y_pred
    ]

    labels_ids = np.arange(
        len(classes),
        dtype=np.int32,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    f1_macro = f1_score(
        y_true,
        y_pred,
        average="macro",
        labels=labels_ids,
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_true,
        y_pred,
        average="weighted",
        labels=labels_ids,
        zero_division=0,
    )

    log(
        f"OOF Accuracy    = {accuracy:.4f}"
    )

    log(
        f"OOF F1-macro    = {f1_macro:.4f}"
    )

    log(
        f"OOF F1-weighted = {f1_weighted:.4f}"
    )

    # -------------------------------------------------------------------------
    # Classification report
    # -------------------------------------------------------------------------

    report = classification_report(
        nomes_true,
        nomes_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        out_dir
        / "avaliacao_kfold_classification_report.csv",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Matrizes
    # -------------------------------------------------------------------------

    cm = confusion_matrix(
        nomes_true,
        nomes_pred,
        labels=classes,
    )

    cm_norm = (
        cm
        / np.maximum(
            cm.sum(
                axis=1,
                keepdims=True,
            ),
            1,
        )
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    ).to_csv(
        out_dir
        / "matriz_confusao_oof.csv",
        encoding="utf-8",
    )

    pd.DataFrame(
        cm_norm,
        index=classes,
        columns=classes,
    ).round(6).to_csv(
        out_dir
        / "matriz_confusao_oof_normalizada.csv",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Predições OOF
    # -------------------------------------------------------------------------

    predicoes = pd.DataFrame(
        {
            "indice_parquet": indices_parquet,
            "fold": folds,
            "y_true_id": y_true,
            "y_pred_id": y_pred,
            "y_true": nomes_true,
            "y_pred": nomes_pred,
            "confidence": confidences,
            "correto": (
                y_true == y_pred
            ),
        }
    )

    predicoes.to_csv(
        out_dir
        / "predicoes_oof.csv",
        index=False,
        encoding="utf-8",
    )

    return {
        "n_documentos": int(
            len(y_true)
        ),
        "n_classes": int(
            len(classes)
        ),
        "accuracy": round(
            float(accuracy),
            6,
        ),
        "f1_macro": round(
            float(f1_macro),
            6,
        ),
        "f1_weighted": round(
            float(f1_weighted),
            6,
        ),
    }


# =============================================================================
# IMPORTÂNCIA DAS FEATURES
# =============================================================================

def salvar_importancias(
    importancias_folds: list[np.ndarray],
    feature_names: list[str],
    out_dir: Path,
) -> None:
    """
    Salva:
        - importância por fold
        - média entre folds
        - desvio-padrão entre folds
    """

    if not importancias_folds:
        return

    arr = np.vstack(
        importancias_folds
    )

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": arr.mean(
                axis=0
            ),
            "importance_std": arr.std(
                axis=0
            ),
            "importance_min": arr.min(
                axis=0
            ),
            "importance_max": arr.max(
                axis=0
            ),
        }
    )

    df = df.sort_values(
        "importance_mean",
        ascending=False,
    )

    df.to_csv(
        out_dir
        / "feature_importances_media.csv",
        index=False,
        encoding="utf-8",
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--folds-file",
        type=Path,
        default=DEFAULT_FOLDS_FILE,
        help=(
            "Arquivo canônico K=5 gerado anteriormente."
        ),
    )

    parser.add_argument(
        "--folds-meta",
        type=Path,
        default=DEFAULT_FOLDS_META,
        help=(
            "Metadados do arquivo de folds."
        ),
    )

    parser.add_argument(
        "--features-dir",
        type=Path,
        default=DEFAULT_FEATURES_DIR,
        help=(
            "Diretório que contém as features compartilhadas, "
            "y.npy e indices_parquet.npy."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=(
            f"Diretório de saída "
            f"(padrão: {DEFAULT_OUT})."
        ),
    )

    parser.add_argument(
        "--amostra",
        type=int,
        default=None,
        help=(
            "Subamostra para testes rápidos. "
            "Os folds continuam sendo os mesmos folds canônicos."
        ),
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help=(
            f"Número de árvores "
            f"(padrão: {N_ESTIMATORS})."
        ),
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=MAX_DEPTH,
        help=(
            f"Profundidade máxima das árvores "
            f"(padrão: {MAX_DEPTH})."
        ),
    )

    parser.add_argument(
        "--min-samples-split",
        type=int,
        default=MIN_SAMPLES_SPLIT,
        help=(
            f"Mínimo de amostras para dividir um nó "
            f"(padrão: {MIN_SAMPLES_SPLIT})."
        ),
    )

    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=MIN_SAMPLES_LEAF,
        help=(
            f"Mínimo de amostras em uma folha "
            f"(padrão: {MIN_SAMPLES_LEAF})."
        ),
    )

    parser.add_argument(
        "--max-features",
        type=str,
        default=MAX_FEATURES,
        help=(
            "Número/proporção de features consideradas "
            "por split. Exemplos: sqrt, log2, 0.5, 100."
        ),
    )

    parser.add_argument(
        "--sem-pesos",
        action="store_true",
        help=(
            "Não usa balanceamento de classes."
        ),
    )

    args = parser.parse_args()

    if args.n_estimators < 1:
        raise ValueError(
            "--n-estimators precisa ser >= 1."
        )

    if args.min_samples_split < 2:
        raise ValueError(
            "--min-samples-split precisa ser >= 2."
        )

    if args.min_samples_leaf < 1:
        raise ValueError(
            "--min-samples-leaf precisa ser >= 1."
        )

    if args.max_depth is not None and args.max_depth < 1:
        raise ValueError(
            "--max-depth precisa ser >= 1."
        )

    # -------------------------------------------------------------------------
    # Converte max_features.
    # -------------------------------------------------------------------------

    max_features_raw = str(
        args.max_features
    ).strip()

    if max_features_raw in {
        "sqrt",
        "log2",
        "None",
    }:
        if max_features_raw == "None":
            max_features = None
        else:
            max_features = max_features_raw
    else:
        try:
            if "." in max_features_raw:
                max_features = float(
                    max_features_raw
                )
            else:
                max_features = int(
                    max_features_raw
                )
        except ValueError as exc:
            raise ValueError(
                "--max-features deve ser sqrt, log2, None, "
                "um inteiro ou um float."
            ) from exc

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    inicio_total = time.time()

    log("=" * 78)
    log(
        "RANDOM FOREST | 50 ÁREAS CAPES | "
        "5-FOLD STRATIFIED CROSS-VALIDATION"
    )
    log("=" * 78)

    log(
        f"Folds:        {args.folds_file}"
    )

    log(
        f"Features:     {args.features_dir}"
    )

    log(
        f"Saída:        {args.out}"
    )

    log(
        f"Árvores:      {args.n_estimators}"
    )

    log(
        f"Max depth:    {args.max_depth}"
    )

    log(
        f"Max features: {max_features}"
    )

    log(
        f"Class weight: "
        + (
            "None"
            if args.sem_pesos
            else CLASS_WEIGHT
        )
    )

    # -------------------------------------------------------------------------
    # Verificação
    # -------------------------------------------------------------------------

    verificar_arquivos(
        folds_file=args.folds_file,
        folds_meta_file=args.folds_meta,
        features_dir=args.features_dir,
    )

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

    (
        X,
        y_full,
        indices_parquet_full,
        features_meta,
    ) = carregar_features(
        args.features_dir
    )

    # -------------------------------------------------------------------------
    # Folds
    # -------------------------------------------------------------------------

    (
        folds_full,
        classes,
        folds_df,
    ) = carregar_folds(
        folds_file=args.folds_file,
        folds_meta_file=args.folds_meta,
        indices_parquet=indices_parquet_full,
        y=y_full,
    )

    # -------------------------------------------------------------------------
    # Amostra
    # -------------------------------------------------------------------------

    (
        y,
        indices_parquet,
        folds,
        indices_base,
    ) = aplicar_amostra(
        y=y_full,
        indices_parquet=indices_parquet_full,
        folds=folds_full,
        amostra=args.amostra,
    )

    if args.amostra is not None:
        log(
            f"[3/7] Usando amostra de {len(y):,} documentos."
        )
    else:
        log(
            "[3/7] Usando o corpus completo."
        )

    # -------------------------------------------------------------------------
    # Arquivo de folds usado nesta execução.
    # -------------------------------------------------------------------------

    folds_execucao = pd.DataFrame(
        {
            "indice_parquet": indices_parquet,
            "fold": folds,
            "id_classe": y,
            "classe": [
                classes[int(v)]
                for v in y
            ],
        }
    )

    folds_execucao = folds_execucao.sort_values(
        "indice_parquet"
    ).reset_index(drop=True)

    folds_execucao.to_csv(
        args.out
        / "folds_usados_nesta_execucao.csv",
        index=False,
        encoding="utf-8",
    )

    np.save(
        args.out / "indices_base.npy",
        indices_base,
    )

    np.save(
        args.out / "folds.npy",
        folds,
    )

    np.save(
        args.out / "y.npy",
        y,
    )

    np.save(
        args.out / "indices_parquet.npy",
        indices_parquet,
    )

    # -------------------------------------------------------------------------
    # Feature names
    # -------------------------------------------------------------------------

    feature_names = features_meta.get(
        "ordem_features",
        [],
    )

    if len(feature_names) != X.shape[1]:

        feature_names = [
            f"feature_{i:04d}"
            for i in range(X.shape[1])
        ]

    # -------------------------------------------------------------------------
    # K-fold
    # -------------------------------------------------------------------------

    log(
        "[4/7] Iniciando os 5 folds..."
    )

    predicoes_oof = np.full(
        len(y),
        -1,
        dtype=np.int32,
    )

    confidence_oof = np.full(
        len(y),
        np.nan,
        dtype=np.float32,
    )

    metricas_folds = []

    importancias_folds = []

    modelos_info = []

    pbar_folds = tqdm(
        total=N_FOLDS,
        desc="Folds",
        unit="fold",
        dynamic_ncols=True,
    )

    for fold_id in range(
        1,
        N_FOLDS + 1,
    ):

        fold_dir = (
            args.out
            / f"fold_{fold_id}"
        )

        fold_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ---------------------------------------------------------------------
        # Índices locais
        # ---------------------------------------------------------------------

        idx_train = np.flatnonzero(
            folds != fold_id
        ).astype(
            np.int64
        )

        idx_valid = np.flatnonzero(
            folds == fold_id
        ).astype(
            np.int64
        )

        if len(idx_train) == 0:
            raise RuntimeError(
                f"Fold {fold_id}: treino vazio."
            )

        if len(idx_valid) == 0:
            raise RuntimeError(
                f"Fold {fold_id}: validação vazia."
            )

        # ---------------------------------------------------------------------
        # Salva índices locais
        # ---------------------------------------------------------------------

        np.save(
            fold_dir
            / f"idx_train_fold_{fold_id}.npy",
            idx_train,
        )

        np.save(
            fold_dir
            / f"idx_valid_fold_{fold_id}.npy",
            idx_valid,
        )

        # Índices originais do parquet.
        np.save(
            fold_dir
            / f"idx_train_fold_{fold_id}_parquet.npy",
            indices_parquet[idx_train],
        )

        np.save(
            fold_dir
            / f"idx_valid_fold_{fold_id}_parquet.npy",
            indices_parquet[idx_valid],
        )

        log("=" * 78)
        log(
            f"FOLD {fold_id}/{N_FOLDS}"
        )
        log("=" * 78)

        log(
            f"Treino:     {len(idx_train):,}"
        )

        log(
            f"Validação:  {len(idx_valid):,}"
        )

        # ---------------------------------------------------------------------
        # Modelo
        # ---------------------------------------------------------------------

        model = criar_modelo_random_forest(
            fold_id=fold_id,
            sem_pesos=args.sem_pesos,
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=max_features,
        )

        # ---------------------------------------------------------------------
        # Treino
        # ---------------------------------------------------------------------

        t0 = time.time()

        # A fatia do memmap vira ndarray para o sklearn.
        X_train = np.asarray(
            X[idx_train]
        )

        y_train = np.asarray(
            y[idx_train]
        )

        log(
            "Treinando Random Forest..."
        )

        model.fit(
            X_train,
            y_train,
        )

        segundos = (
            time.time()
            - t0
        )

        log(
            f"Treinamento concluído em "
            f"{segundos:,.1f}s"
        )

        # ---------------------------------------------------------------------
        # Salva modelo
        # ---------------------------------------------------------------------

        model_path = (
            fold_dir
            / f"modelo_random_forest_50_fold_{fold_id}.joblib"
        )

        joblib.dump(
            model,
            model_path,
            compress=3,
        )

        # ---------------------------------------------------------------------
        # Importância
        # ---------------------------------------------------------------------

        importancias = np.asarray(
            model.feature_importances_,
            dtype=np.float64,
        )

        importancias_folds.append(
            importancias
        )

        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importancias,
            }
        ).sort_values(
            "importance",
            ascending=False,
        ).to_csv(
            fold_dir
            / "feature_importances.csv",
            index=False,
            encoding="utf-8",
        )

        # ---------------------------------------------------------------------
        # Predição
        # ---------------------------------------------------------------------

        log(
            "Predizendo fold..."
        )

        pred_fold, confidence_fold = prever_em_batches(
            model=model,
            X=X,
            indices=idx_valid,
            batch_size=BATCH_SIZE,
        )

        predicoes_oof[
            idx_valid
        ] = pred_fold

        confidence_oof[
            idx_valid
        ] = confidence_fold

        # ---------------------------------------------------------------------
        # Avaliação
        # ---------------------------------------------------------------------

        metricas = avaliar_fold(
            y_true=y[idx_valid],
            y_pred=pred_fold,
            classes=classes,
            fold_id=fold_id,
            fold_dir=fold_dir,
        )

        metricas.update(
            {
                "n_train": int(
                    len(idx_train)
                ),
                "n_estimators": int(
                    args.n_estimators
                ),
                "max_depth": (
                    None
                    if args.max_depth is None
                    else int(args.max_depth)
                ),
                "min_samples_split": int(
                    args.min_samples_split
                ),
                "min_samples_leaf": int(
                    args.min_samples_leaf
                ),
                "max_features": str(
                    max_features
                ),
                "class_weight": (
                    None
                    if args.sem_pesos
                    else CLASS_WEIGHT
                ),
                "bootstrap": bool(
                    BOOTSTRAP
                ),
                "segundos_treinamento": round(
                    segundos,
                    3,
                ),
            }
        )

        metricas_folds.append(
            metricas
        )

        modelos_info.append(
            {
                "fold": int(fold_id),
                "modelo": str(
                    model_path
                ),
                "n_train": int(
                    len(idx_train)
                ),
                "n_valid": int(
                    len(idx_valid)
                ),
                "segundos_treinamento": round(
                    segundos,
                    3,
                ),
            }
        )

        # ---------------------------------------------------------------------
        # Salva resumo do fold
        # ---------------------------------------------------------------------

        (
            fold_dir
            / "treino.json"
        ).write_text(
            json.dumps(
                {
                    "fold": int(fold_id),
                    "n_train": int(
                        len(idx_train)
                    ),
                    "n_valid": int(
                        len(idx_valid)
                    ),
                    "n_estimators": int(
                        args.n_estimators
                    ),
                    "max_depth": (
                        None
                        if args.max_depth is None
                        else int(args.max_depth)
                    ),
                    "min_samples_split": int(
                        args.min_samples_split
                    ),
                    "min_samples_leaf": int(
                        args.min_samples_leaf
                    ),
                    "max_features": str(
                        max_features
                    ),
                    "bootstrap": bool(
                        BOOTSTRAP
                    ),
                    "class_weight": (
                        None
                        if args.sem_pesos
                        else CLASS_WEIGHT
                    ),
                    "random_state": int(
                        SEED + fold_id
                    ),
                    "segundos_treinamento": round(
                        segundos,
                        3,
                    ),
                    "modelo": str(
                        model_path
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # ---------------------------------------------------------------------
        # Libera memória antes do próximo fold
        # ---------------------------------------------------------------------

        del X_train
        del y_train
        del model
        del pred_fold
        del confidence_fold

        gc.collect()

        pbar_folds.update(1)

    pbar_folds.close()

    # -------------------------------------------------------------------------
    # Verificação OOF
    # -------------------------------------------------------------------------

    if np.any(
        predicoes_oof < 0
    ):
        raise RuntimeError(
            "Nem todos os documentos receberam uma predição OOF."
        )

    # -------------------------------------------------------------------------
    # Métricas por fold
    # -------------------------------------------------------------------------

    log(
        "[5/7] Salvando métricas por fold..."
    )

    df_metricas = pd.DataFrame(
        metricas_folds
    )

    df_metricas.to_csv(
        args.out
        / "metricas_por_fold.csv",
        index=False,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Estatísticas agregadas
    # -------------------------------------------------------------------------

    resumo_metricas = {}

    for coluna in [
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "segundos_treinamento",
    ]:

        valores = df_metricas[
            coluna
        ].astype(float)

        resumo_metricas[coluna] = {
            "media": float(
                valores.mean()
            ),
            "desvio_padrao": float(
                valores.std(
                    ddof=1
                )
            ),
            "minimo": float(
                valores.min()
            ),
            "maximo": float(
                valores.max()
            ),
        }

    # -------------------------------------------------------------------------
    # OOF
    # -------------------------------------------------------------------------

    avaliacao_oof = avaliar_oof(
        y_true=y,
        y_pred=predicoes_oof,
        classes=classes,
        folds=folds,
        indices_parquet=indices_parquet,
        confidences=confidence_oof,
        out_dir=args.out,
    )

    # -------------------------------------------------------------------------
    # Importâncias agregadas
    # -------------------------------------------------------------------------

    salvar_importancias(
        importancias_folds=importancias_folds,
        feature_names=feature_names,
        out_dir=args.out,
    )

    # -------------------------------------------------------------------------
    # Distribuição das classes
    # -------------------------------------------------------------------------

    contagens = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    distribuicao = [
        {
            "id": int(i),
            "classe": classes[i],
            "n_docs": int(
                contagens.get(
                    i,
                    0,
                )
            ),
        }
        for i in range(
            len(classes)
        )
    ]

    pd.DataFrame(
        distribuicao
    ).to_csv(
        args.out
        / "distribuicao_classes.csv",
        index=False,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Informação dos modelos
    # -------------------------------------------------------------------------

    pd.DataFrame(
        modelos_info
    ).to_csv(
        args.out
        / "modelos_folds.csv",
        index=False,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Metadados gerais
    # -------------------------------------------------------------------------

    metadata = {
        "algoritmo": "RandomForestClassifier",
        "seed": SEED,
        "k_folds": N_FOLDS,
        "estrategia_validacao": "StratifiedKFold",
        "folds_canonicos": str(
            args.folds_file
        ),
        "folds_meta": str(
            args.folds_meta
        ),
        "features_dir": str(
            args.features_dir
        ),
        "feature_dim": int(
            X.shape[1]
        ),
        "n_docs_usados": int(
            len(y)
        ),
        "n_classes": int(
            len(classes)
        ),
        "classes_50": classes,
        "amostra": (
            None
            if args.amostra is None
            else int(args.amostra)
        ),
        "hiperparametros": {
            "n_estimators": int(
                args.n_estimators
            ),
            "max_depth": (
                None
                if args.max_depth is None
                else int(args.max_depth)
            ),
            "min_samples_split": int(
                args.min_samples_split
            ),
            "min_samples_leaf": int(
                args.min_samples_leaf
            ),
            "max_features": str(
                max_features
            ),
            "bootstrap": bool(
                BOOTSTRAP
            ),
            "class_weight": (
                None
                if args.sem_pesos
                else CLASS_WEIGHT
            ),
            "n_jobs": -1,
        },
        "features": {
            "fonte": str(
                args.features_dir
            ),
            "meta": features_meta,
        },
        "metricas_por_fold": metricas_folds,
        "resumo_metricas_folds": resumo_metricas,
        "avaliacao_oof": avaliacao_oof,
        "modelos": modelos_info,
        "segundos_total": round(
            time.time()
            - inicio_total,
            3,
        ),
    }

    (
        args.out
        / "treino_kfold.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Relatório final
    # -------------------------------------------------------------------------

    log("=" * 78)
    log(
        "RANDOM FOREST K-FOLD FINALIZADO"
    )
    log("=" * 78)

    log(
        f"OOF Accuracy:    "
        f"{avaliacao_oof['accuracy']:.4f}"
    )

    log(
        f"OOF F1-macro:    "
        f"{avaliacao_oof['f1_macro']:.4f}"
    )

    log(
        f"OOF F1-weighted: "
        f"{avaliacao_oof['f1_weighted']:.4f}"
    )

    log(
        "Média F1-macro:  "
        f"{resumo_metricas['f1_macro']['media']:.4f}"
        " ± "
        f"{resumo_metricas['f1_macro']['desvio_padrao']:.4f}"
    )

    log(
        "F1-macro mínimo:  "
        f"{resumo_metricas['f1_macro']['minimo']:.4f}"
    )

    log(
        "F1-macro máximo:  "
        f"{resumo_metricas['f1_macro']['maximo']:.4f}"
    )

    log(
        f"Saída: {args.out}"
    )


if __name__ == "__main__":
    main()
