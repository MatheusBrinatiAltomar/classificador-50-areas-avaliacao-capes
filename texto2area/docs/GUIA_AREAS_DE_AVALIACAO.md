# Guia: estender a texto2area para as áreas de avaliação CAPES

Roteiro para quem vai continuar a biblioteca, saindo das 9 grandes áreas para as
áreas de avaliação da CAPES (50 na lista oficial de 2026; os dados trazem 60 nomes
históricos). Pressupõe o repositório clonado e o dataset baixado (`dados/README.md`).

## Antes de tudo: o que você precisa ter

Só duas coisas: este repositório clonado e o arquivo `dados/corpus_td_lemas.parquet`
(806 MB; onde obter está em `dados/README.md`). **Não** precisa de acesso ao banco,
de GPU nem de rodar a pasta `corpus/`. Ela existe para regenerar o parquet no futuro
e não faz parte do seu trabalho.

Ambiente (uma vez):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[treino]"                   # pacote + pandas, pyarrow, pytest
python -m spacy download pt_core_news_lg     # ~570 MB; só para classificar texto cru (exemplo, testes)
gh release download dados-v1.0.0 -p corpus_td_lemas.parquet -D dados/   # ou o curl de dados/README.md
sha256sum -c dados/corpus_td_lemas.sha256
```

Máquina: 16 GB de RAM bastam para amostras; o corpus completo pede ~32 GB.

## O que já existe

- Um classificador linear (TF-IDF + LinearSVC) que lê título+resumo lematizados e
  prevê a grande área com F1-macro 0,79 (0,73 fora do tempo). Está empacotado em
  `texto2area/` e é reproduzível por `reproduzir/`.
- O dataset de treino com **1.017.734** teses e dissertações já tem a coluna
  `area_avaliacao`. Não é preciso coletar nada.
- Os scripts de treino e avaliação aceitam `--rotulo area_avaliacao`.

## Passo 1. Entender o material (1 dia)

1. Instale o pacote e rode `examples/exemplo.py` e `pytest`.
2. Leia `dados/README.md` inteiro. Abra o parquet e olhe alguns `lemmas_ext`.
3. Rode a avaliação rápida das 9 áreas para ver o formato das saídas:
   `python reproduzir/avaliar.py --amostra 200000` (uns 10 minutos). As saídas ficam em
   `reproduzir/avaliacao/grande_area/`.

## Passo 2. O de-para 60 → 50 (a entrega mais importante)

O arquivo traz **60 nomes** de área porque guarda o nome vigente na data da
defesa; a lista oficial atual tem **50** (`dados/areas_avaliacao_capes.csv`, com colégio,
grande área e link para a página da CAPES de cada área, consultada em 2026-09-07).

O esqueleto `dados/de_para_area_avaliacao.csv` já existe, com **uma linha para cada um
dos 60 nomes** e as colunas `de`, `para`, `n_docs`, `obs`. Ele vem com `para` = `de` e a
coluna `obs` dizendo, para cada nome, se ele já é oficial (48 nomes: manter) ou não
(11 nomes: decidir) ou se é a grafia defeituosa `LINGUíSTICA E LITERATURA` (1). Sua
tarefa é preencher o `para` dos 12 e justificar em `obs`. Regras:

- **Todas as 60 linhas precisam existir**: o treino aborta se faltar algum nome.
- Nome antigo que virou um nome novo: `para` = nome oficial, em maiúsculas exatamente como
  em `areas_avaliacao_capes.csv` (ex.: `CIÊNCIA DA COMPUTAÇÃO` → `COMPUTAÇÃO`; a CAPES voltou ao nome curto).
- Nome que não existe mais e não tem sucessor claro (ex.: `CIÊNCIAS SOCIAIS APLICADAS I`):
  decidir com o orientador. Deixar `para` vazio descarta os documentos.
- A grafia correta de `LINGUÍSTICA E LITERATURA` **não ocorre nos dados**; escreva-a à mão no `para`.
- Áreas que só existem em anos recentes (`CIÊNCIAS E HUMANIDADES PARA A EDUCAÇÃO BÁSICA`, 2.504
  docs) ficam sem documento de treino no protocolo out-of-time em amostras pequenas; o script avisa.

Esse arquivo é versionado no git e citado no artigo. Sem ele nenhum número é comparável.

## Passo 3. Avaliar antes de treinar

```bash
python reproduzir/avaliar.py --rotulo area_avaliacao --mapa-rotulos dados/de_para_area_avaliacao.csv --amostra 200000
python reproduzir/avaliar.py --rotulo area_avaliacao --mapa-rotulos dados/de_para_area_avaliacao.csv   # completo
```

As saídas ficam em `reproduzir/avaliacao/area_avaliacao/`: `avaliacao_resumo.csv`
(métricas globais dos dois protocolos), `avaliacao_report_A_aleatorio.csv` e
`..._B_out_of_time.csv` (precision/recall/F1/support **por classe**),
`avaliacao_confusao_*.csv` (matriz de confusão normalizada por linha) e
`avaliacao_top_features_*.csv` (termos mais discriminativos por classe). A amostra de
200 mil leva uns 15 minutos; o corpus completo, mais de uma hora.

Leia o report **por classe**. Com ~50 classes desbalanceadas o F1-macro será bem menor
que 0,79 (referência: numa amostra de 200 mil sem de-para, 0,50 e 0,48). Perguntas a responder:

- Quais áreas têm F1 baixo? São as pequenas, as guarda-chuva (Interdisciplinar,
  Ensino, Ciências Ambientais) ou pares irmãos (Medicina I/II/III, Engenharias I–IV)?
- A matriz de confusão mostra os erros dentro da mesma grande área? Se sim, um
  modelo hierárquico (grande área → área) pode ajudar; se não, não vale a complexidade.
- O out-of-time cai muito mais que o aleatório? Áreas renomeadas recentemente sofrem aqui.

## Passo 4. Treinar e empacotar

```bash
python reproduzir/treinar_e_salvar.py --rotulo area_avaliacao --mapa-rotulos dados/de_para_area_avaliacao.csv
python reproduzir/_construir_artefatos.py --rotulo area_avaliacao --dest texto2area/data/area_avaliacao
```

Coloque o modelo novo em **`texto2area/data/area_avaliacao/`**: o `pyproject.toml` e o
`MANIFEST.in` já empacotam subpastas de `data/` no wheel (confira com `python -m build`
e `unzip -l dist/*.whl`). Se treinou com `--out`, passe a pasta em `--origem`.

O pacote precisa de uma mudança para carregar dois modelos: hoje `_classify.py` carrega
um único par vetorizador/modelo de `texto2area/data/`. A forma mais simples é uma função
`classificar(texto, nivel="grande_area")` que escolhe a pasta.

Sobre o vetorizador: o TF-IDF não depende do rótulo, mas depende do **conjunto de
documentos**. Se o de-para não descartar nenhum documento, o vetorizador novo é igual ao
publicado e pode ser compartilhado; para garantir isso, treine com
`--vetorizador texto2area/data/vetorizador.joblib` (reaproveita o TF-IDF do pacote sem
refazer o fit). Nesse caso `_construir_artefatos.py` detecta (pelo `treino.json`) e **não** copia o vetorizador: só `modelo.joblib` e `classes.json` novos entram no wheel, e `_classify.py` deve carregar o vetorizador de `texto2area/data/`.

## Passo 5. Documentar e publicar

- Atualizar `README.md` com a tabela de desempenho por protocolo e a lista de áreas.
- Subir a versão (`pyproject.toml`), gerar release, atualizar `CITATION.cff` e o depósito no Zenodo.
- Tamanho do pacote no PyPI (limite padrão 100 MB por arquivo): um modelo de 50 classes tem `coef_` de 50 × 359 mil floats, 144 MB em float64. Por isso `_construir_artefatos.py` converte para **float32** (72 MB) e verifica, em 20 mil documentos do corpus, que as previsões continuam idênticas. O wheel final fica com o vetorizador (12 MB) + modelo de 9 classes (26 MB) + modelo de 50 classes (72 MB) ≈ 110 MB: ainda acima do limite. Opções: converter também o modelo de 9 classes (13 MB, total ≈ 97 MB, apertado), pedir aumento de limite ao PyPI (formulário simples, comum para modelos), ou distribuir o modelo de 50 classes como asset de release baixado no primeiro uso.

## O que vale saber sobre os dados

- O rótulo é o do **programa**, não do conteúdo. Erros do modelo podem ser interdisciplinaridade legítima.
- `class_weight='balanced'` já compensa parte do desbalanceamento; não invente re-amostragem sem medir.
- Não use `min_df` menor que 50 no corpus completo sem motivo: o vocabulário explode e o resultado quase não muda.
- Rótulos novos (ex.: ODS) devem ficar numa tabela à parte chaveada por `id_producao`, com coluna de proveniência. O parquet do corpus não muda.
