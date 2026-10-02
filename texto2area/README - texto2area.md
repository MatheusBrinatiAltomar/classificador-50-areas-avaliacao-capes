# texto2area

[![PyPI](https://img.shields.io/pypi/v/texto2area.svg)](https://pypi.org/project/texto2area/)
[![Python](https://img.shields.io/pypi/pyversions/texto2area.svg)](https://pypi.org/project/texto2area/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21082954.svg)](https://doi.org/10.5281/zenodo.21082954)

Biblioteca Python que classifica o **título e/ou resumo** de uma tese/dissertação
brasileira em uma das **9 grandes áreas do conhecimento CAPES**, a partir do texto —
**ponta a ponta**: recebe texto cru em português e faz normalização, lematização,
n-gramas e classificação.

```python
from texto2area import classificar

area, margens, termos = classificar(
    "A atuação do enfermeiro no cuidado ao paciente idoso em saúde coletiva."
)
# area    -> 'CIÊNCIAS DA SAÚDE'
# margens -> [('CIÊNCIAS DA SAÚDE', 1.50), ('CIÊNCIAS EXATAS E DA TERRA', -1.10), ...]  (9 classes, decrescente)
# termos  -> ['saúde_coletivo', 'enfermeiro', 'paciente', 'cuidado', 'paciente_idoso']  (5 mais decisivos)
```

## Instalação
```bash
pip install texto2area
python -m spacy download pt_core_news_lg     # modelo de lematização (~568 MB), obrigatório
```
Versão de desenvolvimento (GitHub): `pip install git+https://github.com/reneveloso/texto2area`
Ou, a partir de um clone: `pip install .`

## O que você precisa, dependendo do que vai fazer

| Objetivo | Precisa de | Não precisa de |
|---|---|---|
| **Usar** o classificador | `pip install texto2area` + modelo spaCy | nada mais |
| **Treinar ou avaliar** um modelo (ex.: as 50 áreas de avaliação) | este repositório, `pip install -e ".[treino]"`, o modelo spaCy e **um arquivo**: `dados/corpus_td_lemas.parquet` (806 MB, ver [`dados/README.md`](dados/README.md)) | banco de dados, GPU, pasta `corpus/` |
| **Regenerar o corpus** do zero (ex.: incluir 2025) | acesso ao banco Sucupira + pasta [`corpus/`](corpus/README.md) + GPU | nada além disso |

> **Para quem vai continuar a biblioteca:** você precisa apenas do parquet. Ele já está
> pronto. A pasta `corpus/` existe para que o parquet possa ser refeito no futuro; não é
> pré-requisito e você não precisa rodá-la.
>
> **Termos que aparecem nos comentários e documentos:** *Tarrafa* e *Camada 1a* são o
> projeto interno da UNIMONTES (Universidade Estadual de Montes Claros) que gerou o corpus
> original; *DEC-nnn* são os registros de decisão desse projeto; *Bloco 0* é a etapa dele que
> criou os n-gramas. Nada disso é necessário para usar ou estender esta biblioteca: as
> referências existem só para rastrear a origem de cada escolha.

## Uso pretendido e domínio de validade
- **Idioma:** português. Para textos em outro idioma, **traduza antes** (a tradução
  não é embutida — exigiria um modelo pesado).
- **Domínio:** teses e dissertações (título/resumo), 2013–2024 (Catálogo Sucupira).
  Fora disso (outros gêneros, outras taxonomias) o desempenho não é garantido.
- **Saída:** grande área + margens (`decision_function`) por classe + termos do texto
  que mais pesaram na decisão (interpretabilidade do modelo linear).

## Como funciona (fiel ao pipeline de treino)
1. Normalização (NFC, limpeza, colapso de espaços).
2. Lematização com spaCy `pt_core_news_lg`, mantendo POS de conteúdo
   (`NOUN, PROPN, ADJ, VERB`), lema minúsculo, `len>=2`.
3. Injeção de n-gramas: bi/trigramas adjacentes unidos por `_`, mantidos os que
   existem no vocabulário do modelo (85% das 359.402 features são n-gramas).
4. TF-IDF (`sublinear_tf`, `min_df=50`) + `LinearSVC` (one-vs-rest, `class_weight='balanced'`).

## Desempenho (avaliação em conjuntos retidos)
| Protocolo | Acurácia | F1-macro | Baseline (maj.) |
|---|---|---|---|
| In-distribution (split 80/20) | 0,792 | **0,791** | 0,165 |
| Out-of-time (treino ≤2023, teste 2024) | 0,729 | **0,732** | 0,170 |

F1 por área varia de Linguística/Letras/Artes 0,879 e Saúde 0,860 a
**Multidisciplinar 0,549** (classe difusa, sem vocabulário próprio). O artefato é
treinado em 100% dos dados (1.017.727 documentos: os 1.017.734 do corpus menos 7 com texto
vazio após o pré-processamento); as métricas vêm dos protocolos de avaliação.

## Limitações
- Rótulo administrativo como verdade-base (parte dos "erros" é interdisciplinaridade real).
- Multidisciplinar pouco separável (F1 0,549).
- Não distingue as 50 áreas de avaliação (apenas as 9 grandes áreas); ver `docs/GUIA_AREAS_DE_AVALIACAO.md`.
- Português apenas (sem tradução embutida).

## Segurança
O modelo é carregado via `joblib`, que **executa código ao desserializar**. Use apenas
os artefatos versionados neste repositório ou de fonte confiável.

## Dados de treino e reprodução
O corpus de treino (1.017.734 teses e dissertações, texto lematizado, rótulos de
grande área e de área de avaliação) está documentado em [`dados/README.md`](dados/README.md)
e é distribuído como um único parquet de 806 MB, anexado ao release
[`dados-v1.0.0`](https://github.com/reneveloso/texto2area/releases/tag/dados-v1.0.0)
(depósito com DOI no Zenodo: `[Zenodo — DOI a incluir]`).
Com ele e este repositório, o treino é reproduzível ponta a ponta:
[`reproduzir/REPRODUCAO.md`](reproduzir/REPRODUCAO.md) (determinístico, `SEED=42`).
O próprio parquet pode ser reconstruído a partir do banco Sucupira com
[`corpus/`](corpus/README.md) (extração, idioma, lematização, n-gramas), sem
depender de outro repositório.
Para estender o modelo às áreas de avaliação, ver
[`docs/GUIA_AREAS_DE_AVALIACAO.md`](docs/GUIA_AREAS_DE_AVALIACAO.md).

## Como citar
Ver [`CITATION.cff`](CITATION.cff). DOI (Zenodo, concept): [10.5281/zenodo.21082954](https://doi.org/10.5281/zenodo.21082954).

## Licença
[MIT](LICENSE) © 2026 Renê Rodrigues Veloso.
