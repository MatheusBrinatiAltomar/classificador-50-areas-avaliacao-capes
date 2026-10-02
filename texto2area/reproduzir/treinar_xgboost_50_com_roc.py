"""
Treina um segundo estágio XGBoost para classificar teses/dissertações nas
50 áreas de avaliação CAPES usando validação cruzada estratificada com K=5.

Arquitetura:
    lemmas_ext
            |
            +--> HashingVectorizer -> representação compacta do texto
            |
            +--> classificador original texto2area
                    -> 9 margens (decision_function)
                    -> grande_area prevista (one-hot)
            |
            +--> XGBoost multiclasses (50 classes, GPU CUDA)

Validação:
    StratifiedKFold(k=5, shuffle=True, random_state=42)

IMPORTANTE:
- O corpus já contém `lemmas_ext`; não há novo pré-processamento linguístico.
- O primeiro estágio é o artefato publicado em texto2area/data/.
- O arquivo de-para deve estar preenchido para os 60 rótulos históricos.
- O modelo usa a entrada textual + saídas numéricas do primeiro estágio.
- Os "termos decisivos" do primeiro estágio são explicativos/categóricos e
  não entram diretamente nas features.
- Os folds são definidos uma única vez sobre o corpus completo e salvos em:
      folds_capes_5.csv
  permitindo reutilização em outros modelos.

Uso rápido:
    python reproduzir/treinar_xgboost_50_kfold.py --amostra 100000

Treino completo:
    python reproduzir/treinar_xgboost_50_kfold.py

Especificar arquivo de folds:
    python reproduzir/treinar_xgboost_50_kfold.py \
        --folds-file reproduzir/folds_capes_5.csv

Dependências:
    pip install -U xgboost tqdm
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from tqdm import tqdm


# =============================================================================
# CAMINHOS
# =============================================================================

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

CORPUS = REPO / "dados" / "corpus_td_lemas.parquet"
MAPA_ROTULOS = REPO / "dados" / "de_para_area_avaliacao.csv"
AREAS_CAPES = REPO / "dados" / "areas_avaliacao_capes.csv"

# Artefatos do primeiro estágio.
MODELO_PRIMEIRO = REPO / "texto2area" / "data" / "modelo.joblib"
VETORIZADOR_PRIMEIRO = REPO / "texto2area" / "data" / "vetorizador.joblib"

DEFAULT_OUT = HERE / "modelo_treinado" / "xgboost_50_kfold"

# Arquivo canônico de folds.
#
# IMPORTANTE:
# Este arquivo deve ser reutilizado pelos demais modelos para garantir
# exatamente a mesma divisão de treino/validação.
DEFAULT_FOLDS_FILE = HERE / "folds_capes_5.csv"
DEFAULT_FOLDS_META = HERE / "folds_capes_5_meta.json"


# =============================================================================
# CONFIGURAÇÃO EXPERIMENTO
# =============================================================================

SEED = 42

N_FOLDS = 5

COL_TEXTO = "lemmas_ext"
COL_ROTULO = "area_avaliacao"

# Representação compacta para o segundo estágio.
HASH_FEATURES = 256

# Batch usado para geração das features e predição.
BATCH_SIZE = 10_000

# Hiperparâmetros iniciais para RTX 4050 6 GB.
N_ESTIMATORS = 1200
MAX_DEPTH = 8
LEARNING_RATE = 0.08
SUBSAMPLE = 0.85
COLSAMPLE_BYTREE = 0.85
MIN_CHILD_WEIGHT = 3.0
REG_ALPHA = 0.0
REG_LAMBDA = 1.0
MAX_BIN = 64
EARLY_STOPPING = 80

# Coluna que identifica o fold no arquivo compartilhado.
COL_FOLD = "fold"


# =============================================================================
# LOG
# =============================================================================

def log(msg: str) -> None:
    print(msg, flush=True)


# =============================================================================
# VERIFICAÇÃO DOS ARQUIVOS
# =============================================================================

def verificar_arquivos() -> None:
    arquivos = {
        "Corpus": CORPUS,
        "Mapa 60 -> 50": MAPA_ROTULOS,
        "Lista oficial 50": AREAS_CAPES,
        "Modelo texto2area": MODELO_PRIMEIRO,
        "Vetorizador texto2area": VETORIZADOR_PRIMEIRO,
    }

    faltantes = [
        f"{nome}: {path}"
        for nome, path in arquivos.items()
        if not path.exists()
    ]

    if faltantes:
        msg = (
            "Arquivos necessários não encontrados:\n  "
            + "\n  ".join(faltantes)
        )
        raise FileNotFoundError(msg)


# =============================================================================
# MAPEAMENTO 60 -> 50
# =============================================================================

def carregar_mapeamento() -> tuple[
    dict[str, str],
    list[str],
    pd.DataFrame,
]:
    """
    Valida o mapeamento dos 60 rótulos históricos para as 50 áreas oficiais.
    """

    mapa = pd.read_csv(
        MAPA_ROTULOS,
        dtype=str,
    ).fillna("")

    areas = pd.read_csv(
        AREAS_CAPES,
        dtype=str,
    ).fillna("")

    required_mapa = {"de", "para"}
    required_areas = {
        "area_avaliacao",
        "area_avaliacao_maiusculas",
    }

    if not required_mapa.issubset(mapa.columns):
        raise ValueError(
            f"{MAPA_ROTULOS} precisa conter colunas "
            f"{sorted(required_mapa)}"
        )

    if not required_areas.issubset(areas.columns):
        raise ValueError(
            f"{AREAS_CAPES} precisa conter colunas "
            f"{sorted(required_areas)}"
        )

    oficiais = set(
        areas["area_avaliacao_maiusculas"]
        .str.strip()
    )

    oficiais.discard("")

    if len(oficiais) != 50:
        raise ValueError(
            "A lista oficial deveria conter 50 áreas; "
            f"foram encontradas {len(oficiais)}."
        )

    if mapa["de"].duplicated().any():
        dup = mapa.loc[
            mapa["de"].duplicated(),
            "de"
        ].tolist()

        raise ValueError(
            "De-para contém rótulos históricos duplicados: "
            f"{dup}"
        )

    de_para = dict(
        zip(
            mapa["de"].str.strip(),
            mapa["para"].str.strip(),
        )
    )

    vazios = [
        de
        for de, para in de_para.items()
        if not para
    ]

    invalidos = sorted(
        {
            para
            for para in de_para.values()
            if para and para not in oficiais
        }
    )

    if len(de_para) != 60:
        raise ValueError(
            "O de-para deve possuir exatamente 60 "
            f"rótulos históricos; encontrados {len(de_para)}."
        )

    if vazios:
        raise ValueError(
            "O de-para ainda possui rótulos sem destino. "
            "Preencha a coluna `para` antes do treino.\n"
            f"Quantidade: {len(vazios)}\n"
            f"Rótulos: {vazios}"
        )

    if invalidos:
        raise ValueError(
            "O de-para contém destinos que não pertencem "
            "às 50 áreas oficiais:\n"
            + "\n".join(
                f"  - {x}"
                for x in invalidos
            )
        )

    destinos = set(de_para.values())

    faltam_oficiais = sorted(
        oficiais - destinos
    )

    if faltam_oficiais:
        log(
            "AVISO: algumas das 50 áreas oficiais não "
            "aparecem como destino no de-para.\n"
            + "\n".join(
                f"  - {x}"
                for x in faltam_oficiais
            )
        )

    return (
        de_para,
        sorted(oficiais),
        areas,
    )


# =============================================================================
# CARREGAMENTO DOS RÓTULOS
# =============================================================================

def carregar_rotulos() -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    list[str],
]:
    """
    Lê os rótulos do corpus completo.

    Retorna:
        y
        indices_originais
        n_total_original
        classes_50

    Não realiza amostragem aqui.

    A razão é importante:
    os folds canônicos devem ser criados uma única vez sobre o corpus completo.
    """

    log("[1/7] Lendo rótulos do corpus completo...")

    tabela = pq.read_table(
        CORPUS,
        columns=[COL_ROTULO],
    )

    serie = (
        tabela
        .to_pandas()[COL_ROTULO]
        .astype("string")
    )

    mascara = (
        serie.notna()
        & serie.str.strip().ne("")
    )

    serie = (
        serie[mascara]
        .reset_index(drop=True)
    )

    indices_originais = np.flatnonzero(
        mascara.to_numpy()
    )

    de_para, oficiais, _ = carregar_mapeamento()

    y_str = serie.map(
        lambda x: de_para.get(
            str(x).strip(),
            "",
        )
    )

    valid = y_str.ne("").to_numpy()

    y_str = (
        y_str[valid]
        .reset_index(drop=True)
    )

    indices_originais = indices_originais[valid]

    if len(y_str) == 0:
        raise ValueError(
            "Nenhum documento válido restou após "
            "aplicar o de-para 60 -> 50."
        )

    class_to_id = {
        classe: i
        for i, classe in enumerate(oficiais)
    }

    y = np.asarray(
        [
            class_to_id[str(v)]
            for v in y_str
        ],
        dtype=np.int32,
    )

    n_classes_presentes = len(
        np.unique(y)
    )

    if n_classes_presentes < 50:
        log(
            f"AVISO: o corpus contém "
            f"{n_classes_presentes} das 50 classes."
        )

    contagens = pd.Series(y).value_counts()

    min_docs = int(
        contagens.min()
    )

    if min_docs < N_FOLDS:
        raise ValueError(
            f"A menor classe possui apenas {min_docs} "
            f"documentos, mas K={N_FOLDS}. "
            "Cada classe precisa ter pelo menos K documentos "
            "para o StratifiedKFold."
        )

    return (
        y,
        indices_originais.astype(np.int64),
        int(len(tabela)),
        oficiais,
    )


# =============================================================================
# AMOSTRAGEM
# =============================================================================

def aplicar_amostra(
    y: np.ndarray,
    indices_originais: np.ndarray,
    amostra: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Faz uma amostragem após a definição dos folds canônicos.

    Isso garante que --amostra não crie uma nova divisão.
    """

    if amostra is None:
        return (
            y,
            indices_originais,
            None,
        )

    rng = np.random.default_rng(
        SEED
    )

    n = min(
        int(amostra),
        len(y),
    )

    escolha = np.sort(
        rng.choice(
            len(y),
            size=n,
            replace=False,
        )
    )

    return (
        y[escolha],
        indices_originais[escolha],
        escolha,
    )


# =============================================================================
# PRIMEIRO ESTÁGIO
# =============================================================================

def carregar_primeiro_estagio():
    """
    Carrega exatamente os artefatos publicados do primeiro estágio.
    """

    log("[2/7] Carregando artefatos do texto2area...")

    vec = joblib.load(
        VETORIZADOR_PRIMEIRO
    )

    clf = joblib.load(
        MODELO_PRIMEIRO
    )

    if not hasattr(
        clf,
        "decision_function",
    ):
        raise TypeError(
            "O modelo do primeiro estágio não possui "
            "decision_function()."
        )

    classes = [
        str(x)
        for x in clf.classes_
    ]

    if len(classes) != 9:
        raise ValueError(
            "O modelo do primeiro estágio deveria possuir "
            f"9 classes; encontrou {len(classes)}."
        )

    log(
        f"  modelo: {type(clf).__name__}"
    )

    log(
        f"  classes: {len(classes)}"
    )

    log(
        "  features do modelo original: "
        f"{len(vec.get_feature_names_out()):,}"
    )

    return (
        vec,
        clf,
        classes,
    )


# =============================================================================
# TEXTO
# =============================================================================

def normalizar_texto_batch(
    values,
) -> list[str]:
    """
    O parquet já armazena o texto no formato esperado pelo modelo publicado.
    Apenas tratamos nulos.
    """

    return [
        ""
        if v is None
        else str(v)
        for v in values
    ]


# =============================================================================
# HASHING VECTORizer
# =============================================================================

def criar_hash_vectorizer() -> HashingVectorizer:
    """
    Representação textual compacta e fixa.

    Não existe aprendizado de vocabulário nesta etapa.
    """

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


# =============================================================================
# ESTATÍSTICAS TEXTUAIS
# =============================================================================

def estatisticas_texto(
    textos: list[str],
) -> np.ndarray:
    """
    Características simples, baratas e determinísticas.
    """

    out = np.zeros(
        (
            len(textos),
            6,
        ),
        dtype=np.float32,
    )

    for i, texto in enumerate(textos):

        tokens = texto.split()

        n_tokens = len(tokens)

        if n_tokens:

            lengths = np.fromiter(
                (
                    len(t)
                    for t in tokens
                ),
                dtype=np.float32,
                count=n_tokens,
            )

            n_ngrams = sum(
                "_" in t
                for t in tokens
            )

            n_unigramas = (
                n_tokens
                - n_ngrams
            )

            n_unique = len(
                set(tokens)
            )

            out[i, 0] = n_tokens
            out[i, 1] = n_unique
            out[i, 2] = (
                n_unique
                / n_tokens
            )
            out[i, 3] = n_ngrams
            out[i, 4] = n_unigramas
            out[i, 5] = float(
                lengths.mean()
            )

    return out


# =============================================================================
# GRANDE ÁREA
# =============================================================================

def codificar_grande_area(
    preditas: np.ndarray,
    classes_1: list[str],
) -> np.ndarray:
    """
    One-hot da grande área prevista pelo primeiro estágio.
    """

    mapa = {
        classe: i
        for i, classe in enumerate(classes_1)
    }

    ids = np.asarray(
        [
            mapa[str(v)]
            for v in preditas
        ],
        dtype=np.int32,
    )

    one_hot = np.zeros(
        (
            len(ids),
            len(classes_1),
        ),
        dtype=np.float32,
    )

    one_hot[
        np.arange(len(ids)),
        ids,
    ] = 1.0

    return one_hot


# =============================================================================
# GERAÇÃO DOS FOLDS CANÔNICOS
# =============================================================================

def criar_folds_canonicos(
    y: np.ndarray,
    indices_originais: np.ndarray,
    classes: list[str],
    folds_file: Path,
    folds_meta_file: Path,
) -> np.ndarray:
    """
    Cria o arquivo compartilhado de folds.

    O arquivo é baseado no índice ORIGINAL do parquet, e não no índice
    filtrado do numpy. Assim, diferentes modelos podem reutilizar a mesma
    divisão.

    Retorna:
        fold_por_documento

    onde fold_por_documento[i] corresponde a y[i].
    """

    folds_file = Path(
        folds_file
    )

    folds_meta_file = Path(
        folds_meta_file
    )

    if folds_file.exists():

        log(
            f"[3/7] Reutilizando folds existentes: "
            f"{folds_file}"
        )

        folds_df = pd.read_csv(
            folds_file,
        )

        required = {
            "indice_parquet",
            "fold",
            "id_classe",
            "classe",
        }

        if not required.issubset(
            folds_df.columns
        ):
            faltam = sorted(
                required
                - set(folds_df.columns)
            )

            raise ValueError(
                "O arquivo de folds existente não contém "
                f"as colunas necessárias: {faltam}"
            )

        folds_df[
            "indice_parquet"
        ] = folds_df[
            "indice_parquet"
        ].astype(np.int64)

        folds_df["fold"] = folds_df[
            "fold"
        ].astype(np.int16)

        folds_df[
            "id_classe"
        ] = folds_df[
            "id_classe"
        ].astype(np.int32)

        if folds_df["indice_parquet"].duplicated().any():
            raise ValueError(
                "O arquivo de folds contém "
                "índices parquet duplicados."
            )

        indices_arquivo = set(
            folds_df[
                "indice_parquet"
            ].tolist()
        )

        indices_atual = set(
            indices_originais.tolist()
        )

        if indices_arquivo != indices_atual:

            faltam = indices_atual - indices_arquivo
            extras = indices_arquivo - indices_atual

            raise ValueError(
                "O arquivo de folds existente não corresponde "
                "exatamente ao corpus atual.\n"
                f"Documentos atuais ausentes no arquivo: "
                f"{len(faltam):,}\n"
                f"Documentos extras no arquivo: "
                f"{len(extras):,}\n\n"
                "Para manter comparabilidade, não será criada "
                "uma nova divisão automaticamente. "
                "Verifique se o corpus foi alterado ou remova "
                "o arquivo de folds para recriá-lo."
            )

        mapa_fold = dict(
            zip(
                folds_df[
                    "indice_parquet"
                ],
                folds_df["fold"],
            )
        )

        mapa_id = dict(
            zip(
                folds_df[
                    "indice_parquet"
                ],
                folds_df["id_classe"],
            )
        )

        mapa_classe = dict(
            zip(
                folds_df[
                    "indice_parquet"
                ],
                folds_df["classe"].astype(str),
            )
        )

        fold_por_documento = np.asarray(
            [
                mapa_fold[int(idx)]
                for idx in indices_originais
            ],
            dtype=np.int16,
        )

        ids_esperados = np.asarray(
            [
                mapa_id[int(idx)]
                for idx in indices_originais
            ],
            dtype=np.int32,
        )

        classes_esperadas = np.asarray(
            [
                mapa_classe[int(idx)]
                for idx in indices_originais
            ],
            dtype=object,
        )

        ids_atuais = y.astype(
            np.int32
        )

        classes_atuais = np.asarray(
            [
                classes[int(v)]
                for v in y
            ],
            dtype=object,
        )

        if not np.array_equal(
            ids_esperados,
            ids_atuais,
        ):
            raise ValueError(
                "Os rótulos armazenados no arquivo de folds "
                "não correspondem aos rótulos atuais."
            )

        if not np.array_equal(
            classes_esperadas,
            classes_atuais,
        ):
            raise ValueError(
                "Os nomes das classes armazenados no arquivo "
                "de folds não correspondem aos rótulos atuais."
            )

        folds_validos = sorted(
            folds_df["fold"].unique().tolist()
        )

        esperados = list(
            range(1, N_FOLDS + 1)
        )

        if folds_validos != esperados:
            raise ValueError(
                "Os folds existentes devem ser exatamente "
                f"{esperados}, encontrados {folds_validos}."
            )

        return fold_por_documento

    # -------------------------------------------------------------------------
    # Criação
    # -------------------------------------------------------------------------

    log(
        f"[3/7] Criando divisão estratificada K={N_FOLDS}..."
    )

    skf = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    fold_por_documento = np.full(
        len(y),
        -1,
        dtype=np.int16,
    )

    for fold_zero_based, (
        idx_train,
        idx_valid,
    ) in enumerate(
        skf.split(
            np.zeros(len(y)),
            y,
        )
    ):

        fold_id = fold_zero_based + 1

        fold_por_documento[
            idx_valid
        ] = fold_id

    if np.any(
        fold_por_documento < 1
    ):
        raise RuntimeError(
            "Existem documentos sem fold definido."
        )

    # -------------------------------------------------------------------------
    # DataFrame compartilhado
    # -------------------------------------------------------------------------

    folds_df = pd.DataFrame(
        {
            "indice_parquet": indices_originais,
            "fold": fold_por_documento,
            "id_classe": y,
            "classe": [
                classes[int(v)]
                for v in y
            ],
        }
    )

    folds_df = folds_df.sort_values(
        "indice_parquet"
    ).reset_index(drop=True)

    folds_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_df.to_csv(
        folds_file,
        index=False,
        encoding="utf-8",
    )

    meta = {
        "seed": SEED,
        "k": N_FOLDS,
        "n_documentos": int(len(y)),
        "n_classes": int(len(classes)),
        "classes": classes,
        "arquivo_corpus": str(
            CORPUS.relative_to(REPO)
        ),
        "coluna_indice": "indice_parquet",
        "coluna_fold": "fold",
        "estrategia": "StratifiedKFold",
        "shuffle": True,
        "observacao": (
            "Os folds são definidos sobre o corpus completo "
            "de documentos válidos após o mapeamento 60 -> 50."
        ),
    }

    folds_meta_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_meta_file.write_text(
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    log(
        f"  arquivo: {folds_file}"
    )

    log(
        f"  documentos: {len(folds_df):,}"
    )

    log(
        f"  folds: {sorted(folds_df['fold'].unique().tolist())}"
    )

    distribuicao = (
        folds_df
        .groupby(["fold", "classe"])
        .size()
        .reset_index(name="n")
    )

    # Mostra somente o total de cada fold.
    totais_fold = (
        folds_df
        .groupby("fold")
        .size()
        .sort_index()
    )

    for fold, total in totais_fold.items():
        log(
            f"  fold {fold}: {int(total):,} documentos"
        )

    return fold_por_documento


# =============================================================================
# PREPARAÇÃO DE FEATURES
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
    """
    Gera as features do segundo estágio em memmap.

    Cada linha permanece na posição correspondente ao vetor `y`.
    """

    feature_path = (
        out_dir
        / "features_xgboost.float32.dat"
    )

    meta_path = (
        out_dir
        / "features_meta.json"
    )

    log(
        "[4/7] Gerando features do segundo estágio em disco..."
    )

    log(
        f"  dimensão: {feature_dim:,}"
    )

    log(
        f"  documentos: {len(y):,}"
    )

    log(
        f"  batch: {BATCH_SIZE:,}"
    )

    mmap = np.memmap(
        feature_path,
        dtype=np.float32,
        mode="w+",
        shape=(
            len(y),
            feature_dim,
        ),
    )

    # -------------------------------------------------------------------------
    # Linha original parquet -> posição dentro de y/X
    # -------------------------------------------------------------------------

    posicao = np.full(
        n_total_original,
        -1,
        dtype=np.int64,
    )

    posicao[
        indices_originais
    ] = np.arange(
        len(indices_originais),
        dtype=np.int64,
    )

    hv = criar_hash_vectorizer()

    parquet = pq.ParquetFile(
        CORPUS
    )

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

        destinos = posicao[
            indices_batch
        ]

        cursor_original += n

        mask = destinos >= 0

        if not np.any(mask):
            continue

        posicoes_selecionadas = np.flatnonzero(
            mask
        )

        textos_sel = [
            valores[i]
            for i in posicoes_selecionadas
        ]

        destinos_sel = destinos[
            mask
        ]

        # ---------------------------------------------------------------------
        # Primeiro estágio
        # ---------------------------------------------------------------------

        X1 = vec_1.transform(
            textos_sel
        )

        margem = np.asarray(
            clf_1.decision_function(
                X1
            ),
            dtype=np.float32,
        )

        if margem.ndim == 1:
            margem = np.column_stack(
                [-margem, margem]
            ).astype(
                np.float32,
                copy=False,
            )

        preditas = np.asarray(
            clf_1.predict(
                X1
            ),
            dtype=object,
        )

        area_one_hot = codificar_grande_area(
            preditas,
            classes_1,
        )

        # ---------------------------------------------------------------------
        # Texto por hashing
        # ---------------------------------------------------------------------

        Xhash = hv.transform(
            textos_sel
        ).astype(
            np.float32
        )

        Xhash_dense = Xhash.toarray()

        # ---------------------------------------------------------------------
        # Estatísticas
        # ---------------------------------------------------------------------

        stats = estatisticas_texto(
            textos_sel
        )

        # ---------------------------------------------------------------------
        # Organização das features
        #
        # [hash_texto]
        # [margens_9]
        # [grande_area_9]
        # [estatisticas_6]
        # ---------------------------------------------------------------------

        bloco = np.empty(
            (
                len(textos_sel),
                feature_dim,
            ),
            dtype=np.float32,
        )

        a = 0

        b = a + HASH_FEATURES

        bloco[
            :,
            a:b
        ] = Xhash_dense

        a = b

        b = (
            a
            + len(classes_1)
        )

        bloco[
            :,
            a:b
        ] = margem

        a = b

        b = (
            a
            + len(classes_1)
        )

        bloco[
            :,
            a:b
        ] = area_one_hot

        a = b

        bloco[
            :,
            a:
        ] = stats

        mmap[
            destinos_sel
        ] = bloco

        gravados += len(
            destinos_sel
        )

        pbar.update(
            len(destinos_sel)
        )

    pbar.close()

    mmap.flush()

    del mmap

    gc.collect()

    if gravados != len(y):

        raise RuntimeError(
            "Foram geradas features para "
            f"{gravados:,}/{len(y):,} documentos."
        )

    meta = {
        "n_docs": int(len(y)),
        "n_features": int(feature_dim),
        "hash_features": HASH_FEATURES,
        "primeiro_estagio_classes": classes_1,
        "ordem_features": (
            [
                f"hash_{i:04d}"
                for i in range(HASH_FEATURES)
            ]
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
        json.dumps(
            meta,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return feature_path


# =============================================================================
# CALLBACK XGBOOST + TQDM
# =============================================================================

class TqdmCallback(
    xgb.callback.TrainingCallback
):
    """
    Barra de progresso por árvore.
    """

    def __init__(
        self,
        total: int,
        descricao: str,
    ):
        self.total = total
        self.descricao = descricao
        self.pbar: tqdm | None = None

    def before_training(
        self,
        model,
    ):
        self.pbar = tqdm(
            total=self.total,
            desc=self.descricao,
            unit="tree",
            dynamic_ncols=True,
        )

        return model

    def after_iteration(
        self,
        model,
        epoch: int,
        evals_log,
    ):

        if self.pbar is not None:

            self.pbar.update(1)

            if evals_log:

                try:

                    nome_dataset = list(
                        evals_log.keys()
                    )[-1]

                    nome_metrica = list(
                        evals_log[
                            nome_dataset
                        ].keys()
                    )[-1]

                    valor = evals_log[
                        nome_dataset
                    ][
                        nome_metrica
                    ][-1]

                    self.pbar.set_postfix(
                        **{
                            nome_metrica:
                            f"{valor:.5f}"
                        }
                    )

                except Exception:
                    pass

        return False

    def after_training(
        self,
        model,
    ):

        if self.pbar is not None:
            self.pbar.close()

        return model


# =============================================================================
# TREINAMENTO DE UM FOLD
# =============================================================================

def treinar_um_fold(
    X: np.memmap,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_valid: np.ndarray,
    classes: list[str],
    fold_id: int,
    fold_dir: Path,
    num_boost_round: int,
    early_stopping_rounds: int,
    use_balanced: bool = True,
) -> tuple[xgb.Booster, dict]:
    """
    Treina um único modelo do K-fold.
    """

    log("=" * 78)

    log(
        f"FOLD {fold_id}/{N_FOLDS}"
    )

    log("=" * 78)

    log(
        f"Treinamento: {len(idx_train):,}"
    )

    log(
        f"Validação:   {len(idx_valid):,}"
    )

    X_train = X[
        idx_train
    ]

    X_valid = X[
        idx_valid
    ]

    y_train = y[
        idx_train
    ]

    y_valid = y[
        idx_valid
    ]

    # -------------------------------------------------------------------------
    # Pesos balanceados calculados APENAS no treino do fold
    # -------------------------------------------------------------------------

    if use_balanced:

        sample_weight = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        sample_weight = np.asarray(
            sample_weight,
            dtype=np.float32,
        )

    else:

        sample_weight = np.ones(
            len(y_train),
            dtype=np.float32,
        )

    # -------------------------------------------------------------------------
    # Quantile DMatrix
    # -------------------------------------------------------------------------

    dtrain = xgb.QuantileDMatrix(
        X_train,
        label=y_train,
        weight=sample_weight,
        max_bin=MAX_BIN,
    )

    dvalid = xgb.QuantileDMatrix(
        X_valid,
        label=y_valid,
        max_bin=MAX_BIN,
        ref=dtrain,
    )

    # -------------------------------------------------------------------------
    # Parâmetros
    # -------------------------------------------------------------------------

    params = {
        "objective": "multi:softprob",
        "num_class": len(classes),
        "eval_metric": [
            "mlogloss",
            "merror",
        ],
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "reg_alpha": REG_ALPHA,
        "reg_lambda": REG_LAMBDA,
        "max_bin": MAX_BIN,
        "seed": SEED,
        "verbosity": 1,
    }

    log(
        f"Features:    {X.shape[1]:,}"
    )

    log(
        f"Classes:     {len(classes)}"
    )

    log(
        "device=cuda | tree_method=hist"
    )

    callbacks = [
        TqdmCallback(
            num_boost_round,
            f"XGBoost fold {fold_id}",
        ),
        xgb.callback.EarlyStopping(
            rounds=early_stopping_rounds,
            metric_name="mlogloss",
            data_name="valid",
            maximize=False,
            save_best=True,
        ),
    ]

    evals_result = {}

    t0 = time.time()

    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[
            (
                dtrain,
                "train",
            ),
            (
                dvalid,
                "valid",
            ),
        ],
        evals_result=evals_result,
        callbacks=callbacks,
        verbose_eval=False,
    )

    segundos = (
        time.time()
        - t0
    )

    best_iteration = getattr(
        booster,
        "best_iteration",
        None,
    )

    best_score = getattr(
        booster,
        "best_score",
        None,
    )

    log(
        f"Fold {fold_id} concluído em "
        f"{segundos:,.1f}s"
    )

    log(
        f"best_iteration: {best_iteration}"
    )

    log(
        f"best_score: {best_score}"
    )

    # -------------------------------------------------------------------------
    # Salva artefatos
    # -------------------------------------------------------------------------

    fold_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        fold_dir
        / f"modelo_xgboost_50_fold_{fold_id}.ubj"
    )

    booster.save_model(
        model_path
    )

    (
        fold_dir
        / "parametros_xgboost.json"
    ).write_text(
        json.dumps(
            params,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        fold_dir
        / "evals_result.json"
    ).write_text(
        json.dumps(
            evals_result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    info = {
        "fold": int(fold_id),
        "n_train": int(len(y_train)),
        "n_valid": int(len(y_valid)),
        "segundos_treinamento": round(
            segundos,
            3,
        ),
        "best_iteration": (
            None
            if best_iteration is None
            else int(best_iteration)
        ),
        "best_score": (
            None
            if best_score is None
            else float(best_score)
        ),
    }

    # -------------------------------------------------------------------------
    # Libera objetos grandes
    # -------------------------------------------------------------------------

    del dtrain
    del dvalid
    del X_train
    del X_valid

    gc.collect()

    return (
        booster,
        info,
    )


# =============================================================================
# PREDIÇÃO DE UM FOLD
# =============================================================================

def prever_fold(
    booster: xgb.Booster,
    X: np.memmap,
    idx_valid: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    """
    Prediz probabilidades em batches.
    """

    pred_ids = np.empty(
        len(idx_valid),
        dtype=np.int32,
    )

    for inicio in range(
        0,
        len(idx_valid),
        batch_size,
    ):

        fim = min(
            inicio + batch_size,
            len(idx_valid),
        )

        Xb = X[
            idx_valid[inicio:fim]
        ]

        db = xgb.QuantileDMatrix(
            Xb
        )

        prob = booster.predict(
            db
        )

        pred_ids[
            inicio:fim
        ] = np.argmax(
            prob,
            axis=1,
        ).astype(
            np.int32
        )

        del db
        del Xb
        del prob

    gc.collect()

    return pred_ids


# =============================================================================
# AVALIAÇÃO DE UM FOLD
# =============================================================================

def avaliar_fold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    fold_id: int,
    fold_dir: Path,
) -> dict:
    """
    Calcula as métricas de um fold.
    """

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

    resumo = {
        "fold": int(fold_id),
        "n_valid": int(len(y_true)),
        "accuracy": round(
            float(acc),
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

    log(
        f"Fold {fold_id}: "
        f"accuracy={acc:.4f} | "
        f"F1-macro={f1_macro:.4f} | "
        f"F1-weighted={f1_weighted:.4f}"
    )

    return resumo


# =============================================================================
# EXECUÇÃO K-FOLD
# =============================================================================

def executar_kfold(
    X: np.memmap,
    y: np.ndarray,
    folds_por_documento: np.ndarray,
    classes: list[str],
    out_dir: Path,
    num_boost_round: int,
    early_stopping_rounds: int,
    use_balanced: bool,
) -> tuple[
    list[dict],
    np.ndarray,
]:
    """
    Executa os 5 folds.

    Retorna:
        metricas_por_fold
        predicoes_oof
    """

    log(
        "[6/7] Treinando modelos nos 5 folds..."
    )

    predicoes_oof = np.full(
        len(y),
        -1,
        dtype=np.int32,
    )

    # Float32: cerca de 200 MB para 1 milhão de documentos e 50 classes.
    # A linha i corresponde exatamente a y[i], folds[i] e indices_parquet[i].
    scores_oof = np.lib.format.open_memmap(
        out_dir / "scores_oof.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(y), len(classes)),
    )
    scores_oof[:] = np.nan

    metricas = []

    for fold_id in range(
        1,
        N_FOLDS + 1,
    ):

        idx_valid = np.flatnonzero(
            folds_por_documento
            == fold_id
        ).astype(
            np.int64
        )

        idx_train = np.flatnonzero(
            folds_por_documento
            != fold_id
        ).astype(
            np.int64
        )

        if len(idx_valid) == 0:
            raise RuntimeError(
                f"Fold {fold_id} ficou sem documentos."
            )

        # ---------------------------------------------------------------------
        # Salva os índices locais
        #
        # Esses índices correspondem à posição dentro de y/X desta execução.
        # ---------------------------------------------------------------------

        np.save(
            out_dir
            / f"idx_train_fold_{fold_id}.npy",
            idx_train,
        )

        np.save(
            out_dir
            / f"idx_valid_fold_{fold_id}.npy",
            idx_valid,
        )

        # ---------------------------------------------------------------------
        # Salva também os índices ORIGINAIS do parquet.
        #
        # Isso é especialmente útil para outros códigos.
        # ---------------------------------------------------------------------

        np.save(
            out_dir
            / f"idx_train_fold_{fold_id}_parquet.npy",
            idx_train,
        )

        np.save(
            out_dir
            / f"idx_valid_fold_{fold_id}_parquet.npy",
            idx_valid,
        )

        fold_dir = (
            out_dir
            / f"fold_{fold_id}"
        )

        booster, treino_info = treinar_um_fold(
            X=X,
            y=y,
            idx_train=idx_train,
            idx_valid=idx_valid,
            classes=classes,
            fold_id=fold_id,
            fold_dir=fold_dir,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            use_balanced=use_balanced,
        )

        # ---------------------------------------------------------------------
        # Predição
        # ---------------------------------------------------------------------

        pbar = tqdm(
            total=len(idx_valid),
            desc=f"Predição fold {fold_id}",
            unit="doc",
            dynamic_ncols=True,
        )

        pred_fold = np.empty(
            len(idx_valid),
            dtype=np.int32,
        )

        for inicio in range(
            0,
            len(idx_valid),
            BATCH_SIZE,
        ):

            fim = min(
                inicio + BATCH_SIZE,
                len(idx_valid),
            )

            Xb = X[
                idx_valid[inicio:fim]
            ]

            db = xgb.QuantileDMatrix(
                Xb
            )

            prob = booster.predict(
                db
            )

            if prob.shape != (fim - inicio, len(classes)):
                raise ValueError(f"Formato inesperado de scores no fold {fold_id}: {prob.shape}")

            scores_oof[idx_valid[inicio:fim], :] = prob

            pred_fold[
                inicio:fim
            ] = np.argmax(
                prob,
                axis=1,
            ).astype(
                np.int32
            )

            pbar.update(
                fim - inicio
            )

            del db
            del Xb
            del prob

        pbar.close()

        predicoes_oof[
            idx_valid
        ] = pred_fold

        # ---------------------------------------------------------------------
        # Avalia fold
        # ---------------------------------------------------------------------

        metricas_fold = avaliar_fold(
            y_true=y[idx_valid],
            y_pred=pred_fold,
            classes=classes,
            fold_id=fold_id,
            fold_dir=fold_dir,
        )

        metricas_fold.update(
            {
                "n_train": int(
                    len(idx_train)
                ),
                "best_iteration": treino_info[
                    "best_iteration"
                ],
                "best_score": treino_info[
                    "best_score"
                ],
                "segundos_treinamento": treino_info[
                    "segundos_treinamento"
                ],
            }
        )

        metricas.append(
            metricas_fold
        )

        # ---------------------------------------------------------------------
        # Libera modelo do fold anterior
        # ---------------------------------------------------------------------

        del booster
        del pred_fold
        gc.collect()

    if np.any(
        predicoes_oof < 0
    ):
        raise RuntimeError(
            "Existem documentos sem predição OOF. "
            "Cada documento deveria pertencer a exatamente "
            "um fold de validação."
        )

    scores_oof.flush()
    if np.isnan(scores_oof).any():
        raise RuntimeError("Existem documentos sem scores OOF.")

    del scores_oof

    return (
        metricas,
        predicoes_oof,
    )


def gerar_curvas_roc_oof(y: np.ndarray, classes: list[str], out_dir: Path) -> dict:
    """ROC um contra os demais a partir dos scores dos respectivos folds."""
    scores = np.load(out_dir / "scores_oof.npy", mmap_mode="r")
    if scores.shape != (len(y), len(classes)):
        raise ValueError("scores_oof.npy não corresponde aos rótulos e classes desta execução.")

    fpr_grid = np.linspace(0.0, 1.0, 1001)
    tprs = []
    linhas = []
    pontos = []
    fig, ax = plt.subplots(figsize=(8.5, 7))

    for classe_id, nome in enumerate(classes):
        verdade = (y == classe_id)
        if not verdade.any() or verdade.all():
            linhas.append({"classe_id": classe_id, "classe": nome,
                           "auc_ovr": np.nan, "n_positivos": int(verdade.sum())})
            continue
        fpr, tpr, thresholds = roc_curve(verdade, scores[:, classe_id])
        valor = auc(fpr, tpr)
        linhas.append({"classe_id": classe_id, "classe": nome,
                       "auc_ovr": valor, "n_positivos": int(verdade.sum())})
        pontos.extend((classe_id, nome, float(x), float(z), float(th))
                      for x, z, th in zip(fpr, tpr, thresholds))
        tprs.append(np.interp(fpr_grid, fpr, tpr))
        ax.plot(fpr, tpr, color="0.75", alpha=0.27, linewidth=0.8)

    pd.DataFrame(linhas).to_csv(out_dir / "auc_por_classe.csv", index=False)
    pd.DataFrame(pontos, columns=["classe_id", "classe", "fpr", "tpr", "limiar"]).to_csv(
        out_dir / "pontos_roc_por_classe.csv", index=False)

    # Micro: agrega todos os pares (documento, classe) sem criar uma matriz
    # binária gigante na memória; o vetor booliano ocupa cerca de 50 MB/1M.
    binario = np.zeros(scores.shape, dtype=np.bool_)
    binario[np.arange(len(y)), y] = True
    fpr_micro, tpr_micro, thresholds_micro = roc_curve(binario.ravel(), scores.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    pd.DataFrame({"fpr": fpr_micro, "tpr": tpr_micro,
                  "limiar": thresholds_micro}).to_csv(out_dir / "pontos_roc_micro.csv", index=False)
    del binario

    auc_macro = float(np.mean([linha["auc_ovr"] for linha in linhas
                               if np.isfinite(linha["auc_ovr"])]))
    # Macro ROC = média de sensibilidades interpoladas numa grade comum.
    tpr_macro = np.mean(tprs, axis=0)
    pd.DataFrame({"fpr": fpr_grid, "tpr": tpr_macro}).to_csv(
        out_dir / "pontos_roc_macro.csv", index=False)
    ax.plot(fpr_micro, tpr_micro, color="#2166ac", linewidth=2.5,
            label=f"Micro (AUC = {auc_micro:.3f})")
    ax.plot(fpr_grid, tpr_macro, color="#b2182b", linewidth=2.5,
            label=f"Macro (AUC média = {auc_macro:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="black", alpha=0.5, label="Aleatório")
    ax.set(xlabel="Taxa de falsos positivos", ylabel="Taxa de verdadeiros positivos",
           title="Curvas ROC OOF — XGBoost, 50 áreas CAPES", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "curva_roc_oof.png", dpi=220)
    fig.savefig(out_dir / "curva_roc_oof.pdf")
    plt.close(fig)
    resumo = {"auc_micro_ovr_oof": float(auc_micro), "auc_macro_ovr_oof": auc_macro}
    (out_dir / "auc_oof.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
    return resumo


# =============================================================================
# AVALIAÇÃO OOF
# =============================================================================

def avaliar_oof(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    out_dir: Path,
) -> dict:
    """
    Avaliação agregada das predições Out-of-Fold.
    """

    log(
        "[7/7] Gerando avaliação agregada OOF..."
    )

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

    log(
        f"OOF accuracy    = {acc:.4f}"
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
        out_dir
        / "avaliacao_kfold_classification_report.csv",
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Matriz de confusão
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

    pd.DataFrame(
        {
            "y_true_id": y_true,
            "y_pred_id": y_pred,
            "y_true": nomes_true,
            "y_pred": nomes_pred,
        }
    ).to_csv(
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
            float(acc),
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
            "Usa uma subamostra aleatória para testes rápidos. "
            "A divisão dos folds continua sendo derivada do "
            "arquivo canônico."
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
        "--folds-file",
        type=Path,
        default=DEFAULT_FOLDS_FILE,
        help=(
            "Arquivo canônico de folds compartilhado entre "
            "os modelos."
        ),
    )

    parser.add_argument(
        "--folds-meta",
        type=Path,
        default=DEFAULT_FOLDS_META,
        help=(
            "Arquivo JSON de metadados dos folds."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=(
            f"Linhas processadas por lote "
            f"(padrão: {BATCH_SIZE})."
        ),
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=N_ESTIMATORS,
        help=(
            f"Número máximo de árvores por fold "
            f"(padrão: {N_ESTIMATORS})."
        ),
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=EARLY_STOPPING,
        help=(
            f"Rounds de early stopping "
            f"(padrão: {EARLY_STOPPING})."
        ),
    )

    parser.add_argument(
        "--sem-pesos",
        action="store_true",
        help=(
            "Não usa pesos balanceados por classe."
        ),
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
        "TEXTO2AREA -> XGBOOST | "
        "CLASSIFICAÇÃO NAS 50 ÁREAS CAPES | "
        "5-FOLD CROSS-VALIDATION"
    )
    log("=" * 78)

    log(
        f"Repositório: {REPO}"
    )

    log(
        f"Corpus:      {CORPUS}"
    )

    log(
        f"Saída:       {args.out}"
    )

    log(
        f"Folds:       {args.folds_file}"
    )

    log(
        f"K:           {N_FOLDS}"
    )

    # -------------------------------------------------------------------------
    # Verificação
    # -------------------------------------------------------------------------

    verificar_arquivos()

    (
        de_para,
        classes_50,
        areas_df,
    ) = carregar_mapeamento()

    # -------------------------------------------------------------------------
    # Carrega o corpus COMPLETO
    # -------------------------------------------------------------------------

    (
        y_full,
        indices_full,
        n_total_original,
        classes_50,
    ) = carregar_rotulos()

    # -------------------------------------------------------------------------
    # Cria ou carrega folds canônicos
    # -------------------------------------------------------------------------

    folds_full = criar_folds_canonicos(
        y=y_full,
        indices_originais=indices_full,
        classes=classes_50,
        folds_file=args.folds_file,
        folds_meta_file=args.folds_meta,
    )

    # -------------------------------------------------------------------------
    # Faz amostragem somente DEPOIS da definição dos folds
    # -------------------------------------------------------------------------

    (
        y,
        indices_originais,
        indices_escolhidos,
    ) = aplicar_amostra(
        y=y_full,
        indices_originais=indices_full,
        amostra=args.amostra,
    )

    if indices_escolhidos is None:

        folds = folds_full.copy()

    else:

        folds = folds_full[
            indices_escolhidos
        ].copy()

        log(
            f"Amostra utilizada: {len(y):,} documentos"
        )

    # -------------------------------------------------------------------------
    # Copia dos folds utilizados nesta execução
    # -------------------------------------------------------------------------

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

    folds_execucao = folds_execucao.sort_values(
        "indice_parquet"
    ).reset_index(drop=True)

    folds_execucao.to_csv(
        args.out
        / "folds_usados_nesta_execucao.csv",
        index=False,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Verificação dos folds da execução
    # -------------------------------------------------------------------------

    log(
        "Distribuição dos documentos por fold:"
    )

    totais = (
        pd.Series(folds)
        .value_counts()
        .sort_index()
    )

    for fold_id in range(
        1,
        N_FOLDS + 1,
    ):

        log(
            f"  fold {fold_id}: "
            f"{int(totais.get(fold_id, 0)):,}"
        )

    # -------------------------------------------------------------------------
    # Primeiro estágio
    # -------------------------------------------------------------------------

    (
        vec_1,
        clf_1,
        classes_1,
    ) = carregar_primeiro_estagio()

    # -------------------------------------------------------------------------
    # Dimensão total
    # -------------------------------------------------------------------------

    feature_dim = (
        HASH_FEATURES
        + len(classes_1)
        + len(classes_1)
        + 6
    )

    # -------------------------------------------------------------------------
    # Features
    # -------------------------------------------------------------------------

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
        shape=(
            len(y),
            feature_dim,
        ),
    )

    # -------------------------------------------------------------------------
    # Salva y e índices
    # -------------------------------------------------------------------------

    np.save(
        args.out / "y.npy",
        y,
    )

    np.save(
        args.out / "indices_parquet.npy",
        indices_originais,
    )

    np.save(
        args.out / "folds.npy",
        folds,
    )

    # -------------------------------------------------------------------------
    # Mapa rápido:
    #
    # posição em X/y -> fold
    #
    # Isso é o vetor usado pelo treinamento.
    # -------------------------------------------------------------------------

    use_balanced = not args.sem_pesos

    # -------------------------------------------------------------------------
    # K-FOLD
    # -------------------------------------------------------------------------

    (
        metricas_folds,
        predicoes_oof,
    ) = executar_kfold(
        X=X,
        y=y,
        folds_por_documento=folds,
        classes=classes_50,
        out_dir=args.out,
        num_boost_round=max(
            1,
            int(args.n_estimators),
        ),
        early_stopping_rounds=max(
            1,
            int(args.early_stopping),
        ),
        use_balanced=use_balanced,
    )

    # -------------------------------------------------------------------------
    # Métricas por fold
    # -------------------------------------------------------------------------

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
    # Estatísticas agregadas dos folds
    # -------------------------------------------------------------------------

    metricas_numericas = [
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "segundos_treinamento",
    ]

    resumo_metricas = {}

    for coluna in metricas_numericas:

        if coluna not in df_metricas.columns:
            continue

        valores = (
            df_metricas[coluna]
            .astype(float)
        )

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
    # Avaliação OOF
    # -------------------------------------------------------------------------

    avaliacao_oof = avaliar_oof(
        y_true=y,
        y_pred=predicoes_oof,
        classes=classes_50,
        out_dir=args.out,
    )

    auc_oof = gerar_curvas_roc_oof(y, classes_50, args.out)
    log(f"AUC micro OOF: {auc_oof['auc_micro_ovr_oof']:.4f}")
    log(f"AUC macro OOF: {auc_oof['auc_macro_ovr_oof']:.4f}")

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
            "classe": classes_50[i],
            "n_docs": int(
                contagens.get(
                    i,
                    0,
                )
            ),
        }
        for i in range(
            len(classes_50)
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
    # Metadados completos
    # -------------------------------------------------------------------------

    metadata = {
        "seed": SEED,
        "k_folds": N_FOLDS,
        "estrategia_validacao": (
            "StratifiedKFold"
        ),
        "shuffle": True,
        "corpus": str(
            CORPUS.relative_to(REPO)
        ),
        "n_docs_corpus_original": int(
            n_total_original
        ),
        "n_docs_validos_corpus": int(
            len(y_full)
        ),
        "n_docs_usados": int(
            len(y)
        ),
        "n_classes": int(
            len(classes_50)
        ),
        "classes_50": classes_50,
        "amostra": (
            None
            if args.amostra is None
            else int(args.amostra)
        ),
        "arquivo_folds_canonico": str(
            args.folds_file
        ),
        "arquivo_folds_meta": str(
            args.folds_meta
        ),
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
        "de_para": str(
            MAPA_ROTULOS.relative_to(REPO)
        ),
        "lista_oficial": str(
            AREAS_CAPES.relative_to(REPO)
        ),
        "feature_dim": int(
            feature_dim
        ),
        "hash_features": int(
            HASH_FEATURES
        ),
        "balanceamento": (
            "balanced"
            if use_balanced
            else "none"
        ),
        "xgboost_gpu": {
            "device": "cuda",
            "tree_method": "hist",
            "max_bin": MAX_BIN,
        },
        "hiperparametros": {
            "n_estimators_max": int(
                args.n_estimators
            ),
            "max_depth": MAX_DEPTH,
            "learning_rate": LEARNING_RATE,
            "subsample": SUBSAMPLE,
            "colsample_bytree": COLSAMPLE_BYTREE,
            "min_child_weight": MIN_CHILD_WEIGHT,
            "reg_alpha": REG_ALPHA,
            "reg_lambda": REG_LAMBDA,
            "early_stopping": int(
                args.early_stopping
            ),
        },
        "metricas_por_fold": metricas_folds,
        "resumo_metricas_folds": resumo_metricas,
        "avaliacao_oof": avaliacao_oof,
        "auc_oof": auc_oof,
        "de_para_n_linhas": int(
            len(de_para)
        ),
        "timestamp_segundos_total": round(
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
    # Copia informações dos folds para o diretório do experimento
    # -------------------------------------------------------------------------

    try:

        if Path(
            args.folds_file
        ).resolve() != (
            args.out
            / "folds_capes_5.csv"
        ).resolve():

            shutil.copy2(
                args.folds_file,
                args.out
                / "folds_capes_5.csv",
            )

        if Path(
            args.folds_meta
        ).exists():

            if Path(
                args.folds_meta
            ).resolve() != (
                args.out
                / "folds_capes_5_meta.json"
            ).resolve():

                shutil.copy2(
                    args.folds_meta,
                    args.out
                    / "folds_capes_5_meta.json",
                )

    except Exception as exc:

        log(
            "AVISO: não foi possível copiar os "
            f"arquivos de folds para a saída: {exc}"
        )

    # -------------------------------------------------------------------------
    # Final
    # -------------------------------------------------------------------------

    log("=" * 78)
    log(
        "TREINO K-FOLD FINALIZADO"
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
        "Média F1-macro entre folds: "
        f"{resumo_metricas['f1_macro']['media']:.4f}"
        " ± "
        f"{resumo_metricas['f1_macro']['desvio_padrao']:.4f}"
    )

    log(
        f"Folds reutilizáveis: {args.folds_file}"
    )

    log(
        f"Saída dos modelos: {args.out}"
    )


if __name__ == "__main__":
    main()
