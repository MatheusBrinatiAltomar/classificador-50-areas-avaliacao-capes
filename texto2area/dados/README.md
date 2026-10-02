# Dataset de treino: `corpus_td_lemas.parquet`

> **Este é o único arquivo de dados de que você precisa para treinar.** Ele já está
> pronto e é o mesmo que treinou o modelo publicado. Não é necessário acessar banco,
> rodar a pasta `corpus/` nem ter GPU. Quem só vai usar o classificador não precisa
> nem dele.
>
> Nesta pasta, só o que está listado abaixo importa. A subpasta `interim/`, se existir,
> é resíduo de uma reconstrução (`corpus/`) e pode ser apagada.

Corpus de **teses e dissertações brasileiras (2013–2024)** com o texto já
pré-processado no formato que o classificador consome. É o conjunto exato sobre
o qual o modelo publicado na `texto2area` foi treinado e avaliado.

| | |
|---|---|
| Documentos | **1.017.734** (um por linha; sem duplicatas de `id_producao`) |
| Colunas | 11 (descritas abaixo) |
| Tamanho | ~806 MB (parquet, compressão zstd) |
| Período | defesas de 2013 a 2024, três quadriênios CAPES |
| Idioma do texto | português (lemas) |
| Fonte primária | Catálogo de Teses e Dissertações da CAPES (Plataforma Sucupira), dados abertos |
| Gerado por | `corpus/construir.py` deste repositório (réplica autocontida do pipeline Tarrafa / Camada 1a, UNIMONTES). A versão atual foi exportada do pipeline original por `gerar_parquet.py` nesta pasta |
| Integridade | `corpus_td_lemas.sha256` |

**Onde obter:** o arquivo não está no git (tamanho). Está anexado ao release
[`dados-v1.0.0`](https://github.com/reneveloso/texto2area/releases/tag/dados-v1.0.0) do repositório:

```bash
curl -L -o dados/corpus_td_lemas.parquet https://github.com/reneveloso/texto2area/releases/download/dados-v1.0.0/corpus_td_lemas.parquet
sha256sum -c dados/corpus_td_lemas.sha256      # confere o download
# ou, com o GitHub CLI:  gh release download dados-v1.0.0 -p corpus_td_lemas.parquet -D dados/
```

Depósito com DOI no Zenodo: `[Zenodo — DOI a incluir]` (mesmo arquivo, mesmo sha256).

## Arquivos desta pasta

| Arquivo | Papel | Necessário para treinar? |
|---|---|---|
| `corpus_td_lemas.parquet` | **O corpus de treino.** 806 MB, fora do git (baixar do release). | **Sim, é o essencial** |
| `corpus_td_lemas.sha256` | Conferência de integridade do download. | Sim |
| `rotulos_area_avaliacao.csv` | Os 60 nomes de área de avaliação nos dados, com grande área e contagem. | Sim |
| `areas_avaliacao_capes.csv` | A lista **oficial** das 50 áreas atuais (colégio, grande área, nome, link), consultada em 2026-09-07. Gabarito do de-para. | Sim |
| `de_para_area_avaliacao.csv` | Esqueleto do de-para 60 → 50 (uma linha por nome dos dados; `para` a preencher). | Sim |
| `README.md` | Este documento. | Sim |
| `gerar_parquet.py` | Proveniência histórica: como a versão atual foi exportada do pipeline de origem. | Não |
| `publicar_zenodo.py` | Depósito do dataset no Zenodo (uso do mantenedor). | Não |
| `interim/`, `modelos/` | Intermediários de uma reconstrução via `corpus/`. Regeneráveis, fora do git. | Não; pode apagar |

## Leitura rápida

```python
import pandas as pd
df = pd.read_parquet("dados/corpus_td_lemas.parquet")            # tudo (~4 GB em RAM)
df = pd.read_parquet("dados/corpus_td_lemas.parquet",
                     columns=["area_avaliacao", "ano_publicacao", "lemmas_ext"])  # só o necessário
```

O arquivo tem grupos de 100 mil linhas; `pyarrow.parquet.ParquetFile(...).iter_batches()`
permite ler em pedaços se a memória for curta.

## Colunas

| Coluna | Tipo | Descrição |
|---|---|---|
| `id_producao` | string | Identificador da tese/dissertação no Sucupira. **Chave primária** (única). Use-a para juntar rótulos novos (ex.: ODS) em tabela separada, ou para recuperar título e resumo originais nos dados abertos da CAPES. |
| `id_programa` | string | Identificador do programa de pós-graduação (4.880 programas). O rótulo de área vem do programa, não do texto. |
| `chave_canonica` | string | `hash:` + SHA-1 truncado de `título normalizado \| ano`. Serve para detectar registros com **o mesmo título no mesmo ano** (918.528 valores distintos; 99.206 linhas repetem a chave de outra). Casos típicos: cotutela ou trabalho registrado em mais de um programa. O corpus **não** foi deduplicado por essa chave; o modelo foi treinado com todas as linhas. |
| `grande_area` | string | Uma das **9 grandes áreas** do conhecimento (CAPES). Rótulo do modelo publicado. Registros cuja grande área é "Interdisciplinar" foram excluídos na extração (não confundir com a *área de avaliação* Interdisciplinar, que pertence à grande área Multidisciplinar e está presente). |
| `area_avaliacao` | string | **Área de avaliação** CAPES do programa, com o nome vigente na época da defesa. Há **60 nomes distintos**, porque a CAPES renomeou e fundiu áreas ao longo do período (a lista oficial atual tem 50: `areas_avaliacao_capes.csv`). Ver `rotulos_area_avaliacao.csv` e a seção *Cuidados* abaixo. |
| `ano_publicacao` | int16 | Ano da defesa (2013–2024). Usado no protocolo out-of-time (treino ≤ 2023, teste = 2024). |
| `quadrienio` | string | Quadriênio de avaliação CAPES: `2013-2016`, `2017-2020`, `2021-2024`. Derivado do ano. |
| `idioma_original` | string | Idioma detectado (langid) de título+resumo antes do pré-processamento: `pt` 1.014.243, `en` 2.703, `es` 788. Os poucos em en/es foram traduzidos para pt (OPUS-MT) antes da lematização. |
| `n_palavras` | int32 | Número de palavras do texto original (título + resumo). Mediana 333. |
| `lemmas_ext` | string | **O texto que o modelo lê.** Sequência, separada por espaço, de (1) lemas de conteúdo do título + resumo e (2) n-gramas retidos, anexados ao fim como um token com `_`. Detalhes abaixo. |
| `n_lemas_ext` | int32 | Quantidade de tokens em `lemmas_ext` (mediana 269; 7 documentos têm 0 e são descartados no treino). |

### Como `lemmas_ext` foi produzido

O processo completo, passo a passo e executável, está em [`corpus/`](../corpus/README.md). Em resumo:

1. **Texto de entrada:** `título + ". " + resumo`, em minúsculas, Unicode NFC, sem quebras de linha. Só entraram registros com resumo de pelo menos 100 caracteres.
2. **Lematização:** spaCy `pt_core_news_lg`. Mantidos apenas tokens com POS em {NOUN, PROPN, ADJ, VERB}; lema em minúsculas; descartados lemas com menos de 2 caracteres ou sem letra. Isso remove artigos, preposições, pronomes e pontuação, e conflaciona flexões (`solos`, `solo` → `solo`; `irrigados` → `irrigar`).
3. **N-gramas:** bigramas e trigramas adjacentes da sequência de lemas foram contados no corpus inteiro; os **retidos** (frequência ≥ 50, LLR acima da mediana para bigramas, sem stopwords, sem lixo) são acrescentados **ao final** da sequência, unidos por `_` (ex.: `saúde_coletivo`, `ensino_médio`, `aprendizado_máquina`). Os unigramas que os compõem **também ficam**. Cerca de 85% das 359 mil features do modelo são n-gramas.
4. **Stopwords não foram removidas** desta coluna. A remoção acontece no treino, pelo *analyzer* do vetorizador (ver `reproduzir/stopwords/`). Quem treinar um modelo novo deve reutilizar esse analyzer, ou obterá um vocabulário diferente.

Exemplo (início de um documento de Ciências Agrárias):

```
desempenho agronômico feijão-caupi consórcio irrigar efeito manejo solo planta daninho savana roraima ... feijão-caupi_consórcio manejo_solo planta_daninho ...
```

A ordem dos lemas é preservada, então a coluna serve também para métodos que usem contexto local. O texto original **não** está no arquivo; ele pode ser recuperado pelo `id_producao` nos dados abertos da CAPES.

## Distribuições

**Grande área** (9 classes):

| Grande área | Docs |
|---|---:|
| Ciências Humanas | 168.181 |
| Ciências da Saúde | 158.193 |
| Ciências Sociais Aplicadas | 147.687 |
| Multidisciplinar | 143.247 |
| Engenharias | 99.005 |
| Ciências Agrárias | 94.136 |
| Ciências Exatas e da Terra | 83.309 |
| Ciências Biológicas | 62.510 |
| Linguística, Letras e Artes | 61.466 |

**Ano:** 63.543 (2013) a 129.220 (2022); 2021 e 2022 concentram mais defesas.
**Quadriênio:** 274.720 / 320.625 / 422.389.

**Área de avaliação:** 60 nomes, de 67.718 (Interdisciplinar) a 119 (Arquitetura e Urbanismo, nome antigo). Lista completa com contagens em `rotulos_area_avaliacao.csv` (61 linhas: `BIODIVERSIDADE` aparece em duas grandes áreas, 24.423 docs em Ciências Biológicas e 1.220 em Ciências Exatas e da Terra).

## Cuidados ao usar `area_avaliacao`

A CAPES tem hoje **50 áreas de avaliação** (lista oficial em `areas_avaliacao_capes.csv`, consultada no site da CAPES em 2026-09-07), mas o arquivo traz **60 nomes** porque guarda o rótulo **da época da defesa**. Antes de treinar um classificador de áreas de avaliação é preciso construir um **de-para** (60 → 50): decidir, para cada um dos 60 nomes, qual nome oficial ele vira, ou se os documentos são descartados.

Para isso existe o esqueleto `de_para_area_avaliacao.csv` (60 linhas, colunas `de`, `para`, `n_docs`, `obs`). A coluna `obs` já diz, para cada nome, se ele **é** um nome oficial atual (48 casos: basta manter) ou **não é** (11 casos: é preciso escolher o `para`; mais 1 caso de grafia defeituosa). Exemplos do segundo grupo:

| Nome nos dados (antigo ou variante) | Nome oficial atual mais provável | Docs |
|---|---|---|
| CIÊNCIA DA COMPUTAÇÃO | COMPUTAÇÃO (a CAPES voltou ao nome curto) | 17.283 (vs. 3.394 já com o nome novo) |
| COMUNICAÇÃO E INFORMAÇÃO | COMUNICAÇÃO, INFORMAÇÃO E MUSEOLOGIA | 16.846 |
| EDUCAÇÃO FÍSICA | EDUCAÇÃO FÍSICA, FISIOTERAPIA, FONOAUDIOLOGIA E TERAPIA OCUPACIONAL | 13.741 (vs. 3.449) |
| LETRAS / LINGUÍSTICA | LINGUÍSTICA E LITERATURA | 13.556 |
| ARTES / MÚSICA | ARTES | 3.329 |
| ARQUITETURA E URBANISMO | ARQUITETURA, URBANISMO E DESIGN | 119 |
| ADMINISTRAÇÃO, CIÊNCIAS CONTÁBEIS E TURISMO | ADMINISTRAÇÃO PÚBLICA E DE EMPRESAS, CIÊNCIAS CONTÁBEIS E TURISMO | 567 |
| TEOLOGIA; FILOSOFIA/TEOLOGIA:subcomissão TEOLOGIA | CIÊNCIAS DA RELIGIÃO E TEOLOGIA | 1.231; 315 |
| FILOSOFIA/TEOLOGIA:subcomissão FILOSOFIA | FILOSOFIA | 147 |
| CIÊNCIAS SOCIAIS APLICADAS I | (rótulo residual: verificar ou descartar) | 205 |

Atenção ao rótulo `LINGUíSTICA E LITERATURA` (35.088 docs), grafado com um `í` minúsculo no meio por defeito de codificação na fonte. **A grafia correta não ocorre nos dados**: o `para` dele deve ser escrito à mão como `LINGUÍSTICA E LITERATURA`, o nome oficial. A tabela de de-para é uma decisão humana e deve ser versionada junto com o modelo que a usa (`treinar_e_salvar.py --mapa-rotulos`).

Outros pontos:

- **Desbalanceamento forte.** Várias áreas têm menos de 1.000 documentos. O `class_weight='balanced'` ajuda, mas o F1 dessas classes será instável. Avalie sempre por classe.
- **Rótulo administrativo.** A área é a do programa, não do conteúdo. Uma dissertação sobre ensino de física num programa de Educação tem rótulo Educação. Parte dos "erros" do classificador é interdisciplinaridade real.
- **Áreas guarda-chuva.** Interdisciplinar, Ensino e Ciências Ambientais não têm vocabulário próprio e absorvem muita confusão, como Multidisciplinar nas 9 grandes áreas (F1 0,55).
- **Sem texto original.** Para rotulagem por LLM ou por consulta a frases inteiras, recupere título e resumo pelo `id_producao`.

## Glossário

- **Lema / lematização:** forma de dicionário de uma palavra (`irrigados` → `irrigar`), obtida aqui com o spaCy.
- **POS:** classe gramatical (*part of speech*): NOUN substantivo, PROPN nome próprio, ADJ adjetivo, VERB verbo.
- **NFC:** forma de normalização Unicode que unifica acentos compostos e pré-compostos.
- **N-grama:** sequência de 2 (bigrama) ou 3 (trigrama) lemas adjacentes; aqui vira um token só, unido por `_`.
- **PMI / LLR:** medidas de associação entre as palavras de um n-grama (informação mútua pontual; razão de verossimilhança de Dunning). Usadas só para decidir quais n-gramas entram.
- **langid, fastText:** detectores de idioma. **OPUS-MT, NLLB:** modelos de tradução automática usados nos 0,34% de textos em inglês/espanhol.
- **Sucupira:** plataforma da CAPES de onde vêm os metadados das teses. **Tarrafa / Camada 1a:** projeto da UNIMONTES que construiu o corpus original; **DEC-nnn** são seus registros de decisão. Nenhum deles é necessário aqui.

## Licença e citação

Dados derivados do Catálogo de Teses e Dissertações da CAPES (dados abertos). O
texto está lematizado; títulos e resumos originais não são redistribuídos.
Ao usar, cite a `texto2area` (ver `CITATION.cff`) e a CAPES como fonte primária.
