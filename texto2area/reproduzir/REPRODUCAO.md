# Reprodução e retreino

Tudo o que é preciso para treinar, avaliar e empacotar o classificador está
neste repositório mais **um arquivo de dados**: `dados/corpus_td_lemas.parquet`
(806 MB, ver [`dados/README.md`](../dados/README.md) para o conteúdo e onde obter).
O repositório da Camada 1a (Tarrafa), que gerou esse arquivo originalmente,
**não** é necessário: a pasta [`corpus/`](../corpus/README.md) reconstrói o parquet
diretamente do banco Sucupira.

## Ambiente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[treino]"       # texto2area + scikit-learn, spacy, e o extra de treino (pandas, pyarrow, pytest)
python -m spacy download pt_core_news_lg   # só para classificar texto cru (testes/uso)
sha256sum -c dados/corpus_td_lemas.sha256  # confere o download
```

Versões fixadas no `pyproject.toml`; o `scikit-learn==1.8.0` importa, porque os
`.joblib` são específicos da versão.

## Arquivos

| Arquivo | Papel |
|---|---|
| `stopwords/*.txt` | As três listas de stopwords do treino (clássicas, acadêmicas, n-gramas). Copiadas da Camada 1a (DEC-022). |
| `stopwords_lib.py` | Une as três listas. |
| `modelo_lib.py` | `tokenizar`: analyzer do TF-IDF no treino = `split` + remoção de stopwords. |
| `avaliar.py` | Protocolos de avaliação (A aleatório 80/20; B out-of-time treino ≤2023 / teste 2024). |
| `treinar_e_salvar.py` | Treina no corpus completo e salva `modelo_treinado/<rotulo>/`. |
| `_construir_artefatos.py` | Converte o modelo treinado nos artefatos do pacote (`texto2area/data/`). |

## Fluxo

```bash
# 1. avaliar (opcional; reproduz os números do README)
python reproduzir/avaliar.py                       # ~9 grandes áreas, corpus completo, demora
python reproduzir/avaliar.py --amostra 200000      # versão rápida para iterar

# 2. treinar no corpus completo
python reproduzir/treinar_e_salvar.py              # -> reproduzir/modelo_treinado/grande_area/

# 3. empacotar
python reproduzir/_construir_artefatos.py          # -> texto2area/data/{vetorizador,modelo}.joblib, classes.json
pytest                                             # smoke test do pacote
```

Determinístico (`SEED=42`, `min_df=50`, `LinearSVC(C=1, class_weight='balanced')`).

## Treinar com outro rótulo (ex.: área de avaliação)

```bash
# dados/de_para_area_avaliacao.csv: colunas `de,para,...`, UMA LINHA PARA CADA UM DOS 60 nomes
# (o esqueleto já vem com as 60); `para` vazio descarta os documentos daquele rótulo
python reproduzir/avaliar.py          --rotulo area_avaliacao --mapa-rotulos dados/de_para_area_avaliacao.csv --amostra 200000
python reproduzir/treinar_e_salvar.py --rotulo area_avaliacao --mapa-rotulos dados/de_para_area_avaliacao.csv
python reproduzir/_construir_artefatos.py --rotulo area_avaliacao --dest texto2area/data/area_avaliacao
```

Ver [`docs/GUIA_AREAS_DE_AVALIACAO.md`](../docs/GUIA_AREAS_DE_AVALIACAO.md) para o
roteiro completo dessa extensão.

## Notas

- `avaliar.py` exclui da avaliação classes com menos de 2 documentos (impossível
  estratificar) e avisa quais foram. No protocolo B, uma classe que só exista em 2024
  fica sem documento de treino: o script avisa e ela entra com F1 = 0.
- Saídas ficam em `reproduzir/avaliacao/<rotulo>/`: `avaliacao_resumo.csv`,
  `avaliacao_report_{A_aleatorio,B_out_of_time}.csv` (por classe),
  `avaliacao_confusao_*.csv` (matriz normalizada) e `avaliacao_top_features_*.csv`.
- `treinar_e_salvar.py --vetorizador X.joblib` reaproveita um TF-IDF já ajustado (ex.: o do
  pacote) em vez de refazer o fit; `_construir_artefatos.py --origem PASTA` empacota um
  modelo treinado com `--out`.
- `_construir_artefatos.py` grava os coeficientes em **float32** (metade do tamanho) e verifica
  numa amostra de 20 mil docs que as previsões não mudam; `--float64` desliga.
- A troca de analyzer em `_construir_artefatos.py` (`tokenizar` → `split`) é
  verificada: como as stopwords não entram no vocabulário, o `transform` é idêntico.
- Custos de referência (corpus completo, 9 classes): vetorizar ≈ minutos; treinar o
  LinearSVC ≈ dezenas de minutos; 50 classes ≈ 5× o tempo de treino (um SVM por classe).
- O texto cru **não** está no dataset. Para retreinar com outro pré-processamento é
  preciso voltar ao Sucupira via `id_producao`.
