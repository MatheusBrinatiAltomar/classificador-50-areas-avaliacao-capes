"""
Treina um segundo estágio Linear SVC para classificar
teses/dissertações nas 50 áreas de avaliação CAPES.

Mantém a mesma ideia do experimento XGBoost:

    lemmas_ext
        |
        +--> HashingVectorizer -> 256 features
        |
        +--> texto2area
        |       -> 9 margens (decision_function)
        |       -> grande área prevista (one-hot)
        |
        +--> 6 estatísticas textuais
        |
        +--> LinearSVC -> 50 áreas de avaliação

A divisão de treino/validação NÃO é recriada:
o script reutiliza folds_capes_5.csv, permitindo comparação direta
com o experimento XGBoost.

O segundo estágio usa LinearSVC com class_weight='balanced'.

Uso:
    python reproduzir/treinar_svc_50_kfold.py

Teste rápido:
    python reproduzir/treinar_svc_50_kfold.py --amostra 100000
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.svm import LinearSVC
from tqdm import tqdm


# =============================================================================
# CAMINHOS
# =============================================================================

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CORPUS = REPO / "dados" / "corpus_td_lemas.parquet"
MAPA_ROTULOS = REPO / "dados" / "de_para_area_avaliacao.csv"
AREAS_CAPES = REPO / "dados" / "areas_avaliacao_capes.csv"

MODELO_PRIMEIRO = REPO / "texto2area" / "data" / "modelo.joblib"
VETORIZADOR_PRIMEIRO = REPO / "texto2area" / "data" / "vetorizador.joblib"

DEFAULT_OUT = HERE / "modelo_treinado" / "svc_50_kfold"

DEFAULT_FOLDS_FILE = HERE / "folds_capes_5.csv"
DEFAULT_FOLDS_META = HERE / "folds_capes_5_meta.json"


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

SEED = 42
N_FOLDS = 5

COL_TEXTO = "lemmas_ext"
COL_ROTULO = "area_avaliacao"
COL_FOLD = "fold"

HASH_FEATURES = 256
BATCH_SIZE = 10_000

# Hiperparâmetros do LinearSVC.
SVC_C = 1.0
SVC_TOL = 1e-4
SVC_MAX_ITER = 5000

def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# ARQUIVOS / RÓTULOS
# =============================================================================

def verificar_arquivos() -> None:
    arquivos = {
        "Corpus": CORPUS,
        "Mapa 60 -> 50": MAPA_ROTULOS,
        "Lista oficial 50": AREAS_CAPES,
        "Modelo texto2area": MODELO_PRIMEIRO,
        "Vetorizador texto2area": VETORIZADOR_PRIMEIRO,
        "Folds canônicos": DEFAULT_FOLDS_FILE,
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
        )


def carregar_mapeamento() -> tuple[dict[str, str], list[str], pd.DataFrame]:
    mapa = pd.read_csv(MAPA_ROTULOS, dtype=str).fillna("")
    areas = pd.read_csv(AREAS_CAPES, dtype=str).fillna("")

    required_mapa = {"de", "para"}
    required_areas = {"area_avaliacao", "area_avaliacao_maiusculas"}

    if not required_mapa.issubset(mapa.columns):
        raise ValueError(
            f"{MAPA_ROTULOS} precisa conter colunas {sorted(required_mapa)}"
        )

    if not required_areas.issubset(areas.columns):
        raise ValueError(
            f"{AREAS_CAPES} precisa conter colunas {sorted(required_areas)}"
        )

    oficiais = set(areas["area_avaliacao_maiusculas"].str.strip())
    oficiais.discard("")

    if len(oficiais) != 50:
        raise ValueError(
            f"A lista oficial deveria conter 50 áreas; foram encontradas {len(oficiais)}."
        )

    if mapa["de"].duplicated().any():
        dup = mapa.loc[mapa["de"].duplicated(), "de"].tolist()
        raise ValueError(f"De-para contém rótulos históricos duplicados: {dup}")

    de_para = dict(
        zip(
            mapa["de"].str.strip(),
            mapa["para"].str.strip(),
        )
    )

    vazios = [de for de, para in de_para.items() if not para]
    invalidos = sorted(
        {
            para
            for para in de_para.values()
            if para and para not in oficiais
        }
    )

    if len(de_para) != 60:
        raise ValueError(
            f"O de-para deve possuir 60 rótulos históricos; encontrados {len(de_para)}."
        )

    if vazios:
        raise ValueError(
            "O de-para possui rótulos sem destino:\n"
            + "\n".join(vazios)
        )

    if invalidos:
        raise ValueError(
            "Destinos fora das 50 áreas oficiais:\n"
            + "\n".join(invalidos)
        )

    return de_para, sorted(oficiais), areas


def carregar_rotulos() -> tuple[np.ndarray, np.ndarray, int, list[str]]:
    log("[1/7] Lendo rótulos do corpus completo...")

    tabela = pq.read_table(CORPUS, columns=[COL_ROTULO])
    serie = tabela.to_pandas()[COL_ROTULO].astype("string")

    mascara = serie.notna() & serie.str.strip().ne("")
    serie = serie[mascara].reset_index(drop=True)

    indices_originais = np.flatnonzero(mascara.to_numpy())

    de_para, oficiais, _ = carregar_mapeamento()

    y_str = serie.map(
        lambda x: de_para.get(str(x).strip(), "")
    )

    valid = y_str.ne("").to_numpy()
    y_str = y_str[valid].reset_index(drop=True)
    indices_originais = indices_originais[valid]

    class_to_id = {
        classe: i
        for i, classe in enumerate(oficiais)
    }

    y = np.asarray(
        [class_to_id[str(v)] for v in y_str],
        dtype=np.int32,
    )

    if len(np.unique(y)) < 50:
        log(
            f"AVISO: o corpus contém {len(np.unique(y))} das 50 classes."
        )

    return (
        y,
        indices_originais.astype(np.int64),
        int(len(tabela)),
        oficiais,
    )


# =============================================================================
# FOLDS CANÔNICOS
# =============================================================================

def carregar_folds_canonicos(
    y: np.ndarray,
    indices_originais: np.ndarray,
    classes: list[str],
    folds_file: Path,
) -> np.ndarray:

    log(f"[2/7] Reutilizando folds canônicos: {folds_file}")

    if not folds_file.exists():
        raise FileNotFoundError(
            "O arquivo de folds canônicos não existe. "
            "Execute primeiro o pipeline que cria folds_capes_5.csv."
        )

    folds_df = pd.read_csv(folds_file)

    required = {
        "indice_parquet",
        "fold",
        "id_classe",
        "classe",
    }

    if not required.issubset(folds_df.columns):
        faltam = sorted(required - set(folds_df.columns))
        raise ValueError(
            f"O arquivo de folds não contém as colunas necessárias: {faltam}"
        )

    folds_df["indice_parquet"] = folds_df["indice_parquet"].astype(np.int64)
    folds_df["fold"] = folds_df["fold"].astype(np.int16)
    folds_df["id_classe"] = folds_df["id_classe"].astype(np.int32)

    if folds_df["indice_parquet"].duplicated().any():
        raise ValueError("O arquivo de folds contém índices duplicados.")

    if set(folds_df["indice_parquet"]) != set(indices_originais.tolist()):
        raise ValueError(
            "O arquivo de folds não corresponde exatamente ao corpus atual."
        )

    mapa_fold = dict(
        zip(folds_df["indice_parquet"], folds_df["fold"])
    )
    mapa_id = dict(
        zip(folds_df["indice_parquet"], folds_df["id_classe"])
    )
    mapa_classe = dict(
        zip(
            folds_df["indice_parquet"],
            folds_df["classe"].astype(str),
        )
    )

    folds = np.asarray(
        [mapa_fold[int(idx)] for idx in indices_originais],
        dtype=np.int16,
    )

    ids_esperados = np.asarray(
        [mapa_id[int(idx)] for idx in indices_originais],
        dtype=np.int32,
    )

    classes_esperadas = np.asarray(
        [mapa_classe[int(idx)] for idx in indices_originais],
        dtype=object,
    )

    classes_atuais = np.asarray(
        [classes[int(v)] for v in y],
        dtype=object,
    )

    if not np.array_equal(ids_esperados, y.astype(np.int32)):
        raise ValueError(
            "Os IDs de classe dos folds não correspondem aos rótulos atuais."
        )

    if not np.array_equal(classes_esperadas, classes_atuais):
        raise ValueError(
            "Os nomes das classes dos folds não correspondem aos rótulos atuais."
        )

    folds_validos = sorted(np.unique(folds).tolist())
    esperados = list(range(1, N_FOLDS + 1))

    if folds_validos != esperados:
        raise ValueError(
            f"Os folds devem ser {esperados}; encontrados {folds_validos}."
        )

    return folds


def aplicar_amostra(
    y: np.ndarray,
    indices_originais: np.ndarray,
    folds: np.ndarray,
    amostra: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    if amostra is None:
        return y, indices_originais, folds

    rng = np.random.default_rng(SEED)
    n = min(int(amostra), len(y))

    escolha = np.sort(
        rng.choice(len(y), size=n, replace=False)
    )

    return (
        y[escolha],
        indices_originais[escolha],
        folds[escolha],
    )


# =============================================================================
# PRIMEIRO ESTÁGIO: TEXTO2AREA
# =============================================================================

def carregar_primeiro_estagio():
    log("[3/7] Carregando artefatos do texto2area...")

    vec = joblib.load(VETORIZADOR_PRIMEIRO)
    clf = joblib.load(MODELO_PRIMEIRO)

    if not hasattr(clf, "decision_function"):
        raise TypeError(
            "O modelo do primeiro estágio não possui decision_function()."
        )

    classes = [str(x) for x in clf.classes_]

    if len(classes) != 9:
        raise ValueError(
            f"O texto2area deveria possuir 9 classes; encontrou {len(classes)}."
        )

    log(f"  modelo: {type(clf).__name__}")
    log(f"  classes: {len(classes)}")

    return vec, clf, classes


def normalizar_texto_batch(values) -> list[str]:
    return [
        "" if v is None else str(v)
        for v in values
    ]


def criar_hash_vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=HASH_FEATURES,
        analyzer="word",
        token_pattern=r"(?u)\S+",
        lowercase=False,
        strip_accents=None,
        alternate_sign=False,
        norm=None,
        binary=False,
        dtype=np.float32,
    )


def estatisticas_texto(textos: list[str]) -> np.ndarray:
    out = np.zeros((len(textos), 6), dtype=np.float32)

    for i, texto in enumerate(textos):
        tokens = texto.split()
        n_tokens = len(tokens)

        if n_tokens:
            lengths = np.fromiter(
                (len(t) for t in tokens),
                dtype=np.float32,
                count=n_tokens,
            )

            n_ngrams = sum("_" in t for t in tokens)
            n_unigramas = n_tokens - n_ngrams
            n_unique = len(set(tokens))

            out[i, 0] = n_tokens
            out[i, 1] = n_unique
            out[i, 2] = n_unique / n_tokens
            out[i, 3] = n_ngrams
            out[i, 4] = n_unigramas
            out[i, 5] = float(lengths.mean())

    return out


def codificar_grande_area(
    preditas: np.ndarray,
    classes_1: list[str],
) -> np.ndarray:

    mapa = {
        classe: i
        for i, classe in enumerate(classes_1)
    }

    ids = np.asarray(
        [mapa[str(v)] for v in preditas],
        dtype=np.int32,
    )

    one_hot = np.zeros(
        (len(ids), len(classes_1)),
        dtype=np.float32,
    )

    one_hot[np.arange(len(ids)), ids] = 1.0

    return one_hot


# =============================================================================
# FEATURES: EXATAMENTE A MESMA ESTRUTURA DO XGBOOST
# =============================================================================

def gerar_features_em_memmap(
    y: np.ndarray,
    indices_originais: np.ndarray,
    n_total_original: int,
    vec_1,
    clf_1,
    classes_1: list[str],
    out_dir: Path,
    feature_dim: int,
) -> Path:

    feature_path = (
        out_dir / "features_svc.float32.dat"
    )

    meta_path = out_dir / "features_meta.json"

    log("[4/7] Gerando features do segundo estágio...")
    log(f"  dimensão: {feature_dim:,}")
    log(f"  documentos: {len(y):,}")
    log(f"  batch: {BATCH_SIZE:,}")

    mmap = np.memmap(
        feature_path,
        dtype=np.float32,
        mode="w+",
        shape=(len(y), feature_dim),
    )

    posicao = np.full(
        n_total_original,
        -1,
        dtype=np.int64,
    )

    posicao[indices_originais] = np.arange(
        len(indices_originais),
        dtype=np.int64,
    )

    hv = criar_hash_vectorizer()
    parquet = pq.ParquetFile(CORPUS)

    cursor_original = 0
    gravados = 0

    pbar = tqdm(
        total=len(y),
        desc="Features",
        unit="doc",
        dynamic_ncols=True,
    )

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=[COL_TEXTO],
    ):
        valores = normalizar_texto_batch(
            batch.column(0).to_pylist()
        )

        n = len(valores)

        indices_batch = np.arange(
            cursor_original,
            cursor_original + n,
            dtype=np.int64,
        )

        destinos = posicao[indices_batch]
        cursor_original += n

        mask = destinos >= 0

        if not np.any(mask):
            continue

        posicoes_selecionadas = np.flatnonzero(mask)

        textos_sel = [
            valores[i]
            for i in posicoes_selecionadas
        ]

        destinos_sel = destinos[mask]

        # texto2area
        X1 = vec_1.transform(textos_sel)

        margem = np.asarray(
            clf_1.decision_function(X1),
            dtype=np.float32,
        )

        if margem.ndim == 1:
            margem = np.column_stack(
                [-margem, margem]
            ).astype(np.float32, copy=False)

        preditas = np.asarray(
            clf_1.predict(X1),
            dtype=object,
        )

        area_one_hot = codificar_grande_area(
            preditas,
            classes_1,
        )

        # hashing textual
        Xhash_dense = (
            hv.transform(textos_sel)
            .astype(np.float32)
            .toarray()
        )

        # estatísticas
        stats = estatisticas_texto(textos_sel)

        # [hash 256] + [margens 9] + [one-hot 9] + [stats 6]
        bloco = np.empty(
            (len(textos_sel), feature_dim),
            dtype=np.float32,
        )

        a = 0

        b = a + HASH_FEATURES
        bloco[:, a:b] = Xhash_dense
        a = b

        b = a + len(classes_1)
        bloco[:, a:b] = margem
        a = b

        b = a + len(classes_1)
        bloco[:, a:b] = area_one_hot
        a = b

        bloco[:, a:] = stats

        mmap[destinos_sel] = bloco

        gravados += len(destinos_sel)
        pbar.update(len(destinos_sel))

    pbar.close()
    mmap.flush()

    del mmap
    gc.collect()

    if gravados != len(y):
        raise RuntimeError(
            f"Features geradas para {gravados:,}/{len(y):,} documentos."
        )

    meta = {
        "n_docs": int(len(y)),
        "n_features": int(feature_dim),
        "hash_features": HASH_FEATURES,
        "primeiro_estagio_classes": classes_1,
        "classificador_segundo_estagio": "LinearSVC",
        "ordem_features": (
            [f"hash_{i:04d}" for i in range(HASH_FEATURES)]
            + [
                f"margem_{i}_{classes_1[i]}"
                for i in range(len(classes_1))
            ]
            + [
                f"grande_area_{i}_{classes_1[i]}"
                for i in range(len(classes_1))
            ]
            + [
                "texto_n_tokens",
                "texto_n_unicos",
                "texto_razao_unicos_tokens",
                "texto_n_ngrams",
                "texto_n_unigramas",
                "texto_tamanho_medio_token",
            ]
        ),
    }

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return feature_path


# =============================================================================
# LINEAR SVC
# =============================================================================

def treinar_um_fold(
    X: np.memmap,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_valid: np.ndarray,
    classes: list[str],
    fold_id: int,
    fold_dir: Path,
    c: float,
    tol: float,
    max_iter: int,
    use_balanced: bool = True,
) -> tuple[LinearSVC, dict]:

    log("=" * 78)
    log(f"FOLD {fold_id}/{N_FOLDS}")
    log("=" * 78)
    log(f"Treinamento: {len(idx_train):,}")
    log(f"Validação:   {len(idx_valid):,}")

    fold_dir.mkdir(parents=True, exist_ok=True)

    model = LinearSVC(
        C=c,
        tol=tol,
        max_iter=max_iter,
        class_weight="balanced" if use_balanced else None,
        dual="auto",
        random_state=SEED,
    )

    # LinearSVC não possui partial_fit. Carregamos somente o fold de treino.
    # Com 280 features e ~814 mil documentos, isto exige RAM considerável.
    log("Carregando features do fold de treino para RAM...")
    t0 = time.time()

    X_train = np.asarray(X[idx_train], dtype=np.float32)
    y_train = y[idx_train]

    log("Ajustando LinearSVC...")
    model.fit(X_train, y_train)

    segundos = time.time() - t0

    model_path = (
        fold_dir / f"modelo_svc_50_fold_{fold_id}.joblib"
    )
    joblib.dump(model, model_path)

    parametros = {
        "modelo": "LinearSVC",
        "C": float(c),
        "tol": float(tol),
        "max_iter": int(max_iter),
        "class_weight": "balanced" if use_balanced else None,
        "dual": "auto",
        "random_state": SEED,
    }

    (fold_dir / "parametros_svc.json").write_text(
        json.dumps(parametros, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    info = {
        "fold": int(fold_id),
        "n_train": int(len(idx_train)),
        "n_valid": int(len(idx_valid)),
        "segundos_treinamento": round(segundos, 3),
        "n_iter_max": int(np.max(np.atleast_1d(model.n_iter_))),
    }

    log(
        f"Fold {fold_id} treinado em {segundos:,.1f}s "
        f"| iterações={info['n_iter_max']}"
    )

    del X_train
    del y_train
    gc.collect()

    return model, info


def prever_fold(
    model: LinearSVC,
    X: np.memmap,
    idx_valid: np.ndarray,
    batch_size: int,
) -> np.ndarray:

    pred_ids = np.empty(
        len(idx_valid),
        dtype=np.int32,
    )

    pbar = tqdm(
        total=len(idx_valid),
        desc="Predição",
        unit="doc",
        dynamic_ncols=True,
    )

    for inicio in range(0, len(idx_valid), batch_size):
        fim = min(inicio + batch_size, len(idx_valid))
        ids = idx_valid[inicio:fim]

        Xb = np.asarray(X[ids], dtype=np.float32)

        pred_ids[inicio:fim] = (
            model.predict(Xb).astype(np.int32)
        )

        pbar.update(len(ids))
        del Xb
        del ids

    pbar.close()
    return pred_ids


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

    acc = accuracy_score(
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

    report_dict = classification_report(
        nomes_true,
        nomes_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report_dict
    ).T.to_csv(
        fold_dir / "classification_report.csv",
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
            cm.sum(axis=1, keepdims=True),
            1,
        )
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    ).to_csv(
        fold_dir / "matriz_confusao.csv",
        encoding="utf-8",
    )

    pd.DataFrame(
        cm_norm,
        index=classes,
        columns=classes,
    ).round(6).to_csv(
        fold_dir / "matriz_confusao_normalizada.csv",
        encoding="utf-8",
    )

    resumo = {
        "fold": int(fold_id),
        "n_valid": int(len(y_true)),
        "accuracy": round(float(acc), 6),
        "f1_macro": round(float(f1_macro), 6),
        "f1_weighted": round(float(f1_weighted), 6),
    }

    log(
        f"Fold {fold_id}: "
        f"accuracy={acc:.4f} | "
        f"F1-macro={f1_macro:.4f} | "
        f"F1-weighted={f1_weighted:.4f}"
    )

    return resumo


def executar_kfold(
    X: np.memmap,
    y: np.ndarray,
    folds_por_documento: np.ndarray,
    indices_originais: np.ndarray,
    classes: list[str],
    out_dir: Path,
    batch_size: int,
    c: float,
    tol: float,
    max_iter: int,
    use_balanced: bool,
) -> tuple[list[dict], np.ndarray]:

    log("[5/7] Treinando LinearSVC nos 5 folds...")

    predicoes_oof = np.full(
        len(y),
        -1,
        dtype=np.int32,
    )

    metricas = []

    for fold_id in range(
        1,
        N_FOLDS + 1,
    ):
        idx_valid = np.flatnonzero(
            folds_por_documento == fold_id
        ).astype(np.int64)

        idx_train = np.flatnonzero(
            folds_por_documento != fold_id
        ).astype(np.int64)

        if len(idx_valid) == 0:
            raise RuntimeError(
                f"Fold {fold_id} ficou sem documentos."
            )

        np.save(
            out_dir / f"idx_train_fold_{fold_id}.npy",
            idx_train,
        )

        np.save(
            out_dir / f"idx_valid_fold_{fold_id}.npy",
            idx_valid,
        )

        # Aqui são realmente os índices ORIGINAIS do parquet.
        np.save(
            out_dir / f"idx_train_fold_{fold_id}_parquet.npy",
            indices_originais[idx_train],
        )

        np.save(
            out_dir / f"idx_valid_fold_{fold_id}_parquet.npy",
            indices_originais[idx_valid],
        )

        fold_dir = (
            out_dir / f"fold_{fold_id}"
        )

        model, treino_info = treinar_um_fold(
            X=X,
            y=y,
            idx_train=idx_train,
            idx_valid=idx_valid,
            classes=classes,
            fold_id=fold_id,
            fold_dir=fold_dir,
            c=c,
            tol=tol,
            max_iter=max_iter,
            use_balanced=use_balanced,
        )

        pred_fold = prever_fold(
            model=model,
            X=X,
            idx_valid=idx_valid,
            batch_size=batch_size,
        )

        predicoes_oof[idx_valid] = pred_fold

        metricas_fold = avaliar_fold(
            y_true=y[idx_valid],
            y_pred=pred_fold,
            classes=classes,
            fold_id=fold_id,
            fold_dir=fold_dir,
        )

        metricas_fold.update(
            {
                "n_train": int(len(idx_train)),
                "segundos_treinamento": treino_info[
                    "segundos_treinamento"
                ],
            }
        )

        metricas.append(metricas_fold)

        del model
        del pred_fold
        gc.collect()

    if np.any(predicoes_oof < 0):
        raise RuntimeError(
            "Existem documentos sem predição OOF."
        )

    return metricas, predicoes_oof


def avaliar_oof(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    indices_originais: np.ndarray,
    folds: np.ndarray,
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

    acc = accuracy_score(y_true, y_pred)

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

    log(f"OOF accuracy    = {acc:.4f}")
    log(f"OOF F1-macro    = {f1_macro:.4f}")
    log(f"OOF F1-weighted = {f1_weighted:.4f}")

    report_dict = classification_report(
        nomes_true,
        nomes_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report_dict
    ).T.to_csv(
        out_dir / "avaliacao_kfold_classification_report.csv",
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
            cm.sum(axis=1, keepdims=True),
            1,
        )
    )

    pd.DataFrame(
        cm,
        index=classes,
        columns=classes,
    ).to_csv(
        out_dir / "matriz_confusao_oof.csv",
        encoding="utf-8",
    )

    pd.DataFrame(
        cm_norm,
        index=classes,
        columns=classes,
    ).round(6).to_csv(
        out_dir / "matriz_confusao_oof_normalizada.csv",
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "indice_parquet": indices_originais,
            "fold": folds,
            "y_true_id": y_true,
            "y_pred_id": y_pred,
            "y_true": nomes_true,
            "y_pred": nomes_pred,
        }
    ).to_csv(
        out_dir / "predicoes_oof.csv",
        index=False,
        encoding="utf-8",
    )

    return {
        "n_documentos": int(len(y_true)),
        "n_classes": int(len(classes)),
        "accuracy": round(float(acc), 6),
        "f1_macro": round(float(f1_macro), 6),
        "f1_weighted": round(float(f1_weighted), 6),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    global BATCH_SIZE

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--amostra",
        type=int,
        default=None,
        help=(
            "Subamostra aleatória para teste. "
            "Os folds continuam sendo derivados do arquivo canônico."
        ),
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Diretório de saída (padrão: {DEFAULT_OUT}).",
    )

    parser.add_argument(
        "--folds-file",
        type=Path,
        default=DEFAULT_FOLDS_FILE,
        help="Arquivo canônico de folds compartilhado.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Documentos por lote (padrão: {BATCH_SIZE}).",
    )

    parser.add_argument(
        "--C",
        type=float,
        default=SVC_C,
        help=f"Parâmetro C do LinearSVC (padrão: {SVC_C}).",
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=SVC_TOL,
        help=f"Tolerância do LinearSVC (padrão: {SVC_TOL}).",
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=SVC_MAX_ITER,
        help=f"Máximo de iterações (padrão: {SVC_MAX_ITER}).",
    )

    parser.add_argument(
        "--sem-pesos",
        action="store_true",
        help="Não usa pesos balanceados por classe.",
    )

    args = parser.parse_args()

    BATCH_SIZE = max(
        100,
        int(args.batch_size),
    )

    inicio_total = time.time()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    log("=" * 78)
    log(
        "TEXTO2AREA -> LINEAR SVC | "
        "50 ÁREAS CAPES | 5-FOLD CROSS-VALIDATION"
    )
    log("=" * 78)

    verificar_arquivos()

    (
        y_full,
        indices_full,
        n_total_original,
        classes_50,
    ) = carregar_rotulos()

    folds_full = carregar_folds_canonicos(
        y=y_full,
        indices_originais=indices_full,
        classes=classes_50,
        folds_file=args.folds_file,
    )

    (
        y,
        indices_originais,
        folds,
    ) = aplicar_amostra(
        y=y_full,
        indices_originais=indices_full,
        folds=folds_full,
        amostra=args.amostra,
    )

    if args.amostra is not None:
        log(f"Amostra utilizada: {len(y):,} documentos")

    folds_execucao = pd.DataFrame(
        {
            "indice_parquet": indices_originais,
            "fold": folds,
            "id_classe": y,
            "classe": [
                classes_50[int(v)]
                for v in y
            ],
        }
    )

    folds_execucao.to_csv(
        args.out / "folds_usados_nesta_execucao.csv",
        index=False,
        encoding="utf-8",
    )

    (
        vec_1,
        clf_1,
        classes_1,
    ) = carregar_primeiro_estagio()

    feature_dim = (
        HASH_FEATURES
        + len(classes_1)
        + len(classes_1)
        + 6
    )

    # Deve ser 280 com 9 grandes áreas.
    log(f"Dimensão final das features: {feature_dim}")

    feature_path = gerar_features_em_memmap(
        y=y,
        indices_originais=indices_originais,
        n_total_original=n_total_original,
        vec_1=vec_1,
        clf_1=clf_1,
        classes_1=classes_1,
        out_dir=args.out,
        feature_dim=feature_dim,
    )

    X = np.memmap(
        feature_path,
        dtype=np.float32,
        mode="r",
        shape=(len(y), feature_dim),
    )

    np.save(args.out / "y.npy", y)
    np.save(
        args.out / "indices_parquet.npy",
        indices_originais,
    )
    np.save(args.out / "folds.npy", folds)

    use_balanced = not args.sem_pesos

    (
        metricas_folds,
        predicoes_oof,
    ) = executar_kfold(
        X=X,
        y=y,
        folds_por_documento=folds,
        indices_originais=indices_originais,
        classes=classes_50,
        out_dir=args.out,
        batch_size=BATCH_SIZE,
        c=float(args.C),
        tol=float(args.tol),
        max_iter=int(args.max_iter),
        use_balanced=use_balanced,
    )

    df_metricas = pd.DataFrame(
        metricas_folds
    )

    df_metricas.to_csv(
        args.out / "metricas_por_fold.csv",
        index=False,
        encoding="utf-8",
    )

    metricas_numericas = [
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "segundos_treinamento",
    ]

    resumo_metricas = {}

    for coluna in metricas_numericas:
        valores = df_metricas[coluna].astype(float)

        resumo_metricas[coluna] = {
            "media": float(valores.mean()),
            "desvio_padrao": float(
                valores.std(ddof=1)
            ),
            "minimo": float(valores.min()),
            "maximo": float(valores.max()),
        }

    avaliacao_oof = avaliar_oof(
        y_true=y,
        y_pred=predicoes_oof,
        classes=classes_50,
        indices_originais=indices_originais,
        folds=folds,
        out_dir=args.out,
    )

    contagens = (
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    distribuicao = [
        {
            "id": int(i),
            "classe": classes_50[i],
            "n_docs": int(contagens.get(i, 0)),
        }
        for i in range(len(classes_50))
    ]

    pd.DataFrame(
        distribuicao
    ).to_csv(
        args.out / "distribuicao_classes.csv",
        index=False,
        encoding="utf-8",
    )

    metadata = {
        "seed": SEED,
        "k_folds": N_FOLDS,
        "estrategia_validacao": "folds_capes_5.csv",
        "corpus": str(CORPUS.relative_to(REPO)),
        "n_docs_corpus_original": int(n_total_original),
        "n_docs_validos_corpus": int(len(y_full)),
        "n_docs_usados": int(len(y)),
        "n_classes": int(len(classes_50)),
        "classes_50": classes_50,
        "amostra": (
            None
            if args.amostra is None
            else int(args.amostra)
        ),
        "arquivo_folds_canonico": str(args.folds_file),
        "primeiro_estagio": {
            "modelo": str(
                MODELO_PRIMEIRO.relative_to(REPO)
            ),
            "vetorizador": str(
                VETORIZADOR_PRIMEIRO.relative_to(REPO)
            ),
            "classes": classes_1,
            "usa_decision_function": True,
            "usa_grande_area_prevista": True,
        },
        "feature_dim": int(feature_dim),
        "hash_features": int(HASH_FEATURES),
        "segundo_estagio": {
            "modelo": "LinearSVC",
            "C": float(args.C),
            "tol": float(args.tol),
            "max_iter": int(args.max_iter),
            "class_weight": "balanced" if use_balanced else None,
            "dual": "auto",
            "batch_size_predicao": int(BATCH_SIZE),
        },
        "balanceamento": (
            "balanced"
            if use_balanced
            else "none"
        ),
        "metricas_por_fold": metricas_folds,
        "resumo_metricas_folds": resumo_metricas,
        "avaliacao_oof": avaliacao_oof,
        "timestamp_segundos_total": round(
            time.time() - inicio_total,
            3,
        ),
    }

    (
        args.out / "treino_kfold.json"
    ).write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Copia folds para a pasta do experimento.
    try:
        if Path(args.folds_file).resolve() != (
            args.out / "folds_capes_5.csv"
        ).resolve():
            shutil.copy2(
                args.folds_file,
                args.out / "folds_capes_5.csv",
            )

        if DEFAULT_FOLDS_META.exists():
            shutil.copy2(
                DEFAULT_FOLDS_META,
                args.out / "folds_capes_5_meta.json",
            )
    except Exception as exc:
        log(
            "AVISO: não foi possível copiar os folds: "
            f"{exc}"
        )

    log("[7/7] Finalizando...")
    log("=" * 78)
    log("TREINO K-FOLD FINALIZADO")
    log("=" * 78)
    log(
        f"OOF Accuracy:    {avaliacao_oof['accuracy']:.4f}"
    )
    log(
        f"OOF F1-macro:    {avaliacao_oof['f1_macro']:.4f}"
    )
    log(
        f"OOF F1-weighted: {avaliacao_oof['f1_weighted']:.4f}"
    )
    log(
        "Média F1-macro entre folds: "
        f"{resumo_metricas['f1_macro']['media']:.4f}"
        " ± "
        f"{resumo_metricas['f1_macro']['desvio_padrao']:.4f}"
    )
    log(f"Saída: {args.out}")


if __name__ == "__main__":
    main()