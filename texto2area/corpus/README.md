# Construção do corpus (do banco ao parquet)

> **Você provavelmente não precisa desta pasta.** Ela refaz `dados/corpus_td_lemas.parquet`
> a partir do banco Sucupira e exige acesso ao banco, 32 GB de RAM, GPU e ~2 h.
> Para **treinar** modelos, baixe o parquet pronto (ver `dados/README.md`) e ignore isto.
> Rode esta pasta apenas para regenerar o corpus (ex.: ao entrar um ano novo).
> Tudo que ela grava em `dados/interim/` é intermediário e pode ser apagado depois.

Esta pasta reconstrói `dados/corpus_td_lemas.parquet` a partir do banco Sucupira,
sem depender de nenhum outro repositório. É a réplica, enxuta e autocontida, do
pipeline que gerou o corpus original (Tarrafa / Camada 1a, junho de 2026).
Quem só vai **treinar** modelos não precisa disto: basta baixar o parquet
(ver `dados/README.md`).

## Passos

| # | Script | Entrada → saída (em `dados/interim/`) | O que faz | Tempo |
|---|---|---|---|---|
| 1 | `extrair.py` | banco → `01_bruto.parquet` | SQL sobre `teses_completas`: título + ". " + resumo, rótulos, ano. Filtros: grande área informada e ≠ Interdisciplinar, resumo ≥ 100 caracteres, 2013–2024. Chave canônica (hash). | ~15 min |
| 2 | `idioma.py` | → `02_normalizado.parquet` | Idioma por langid restrito a pt/en/es; texto NFC; contagem de palavras. | ~10 min |
| 3 | `lematizar.py` | → `03_lemas_pt.parquet` | spaCy `pt_core_news_lg` nos docs em pt; lemas de conteúdo. Em blocos, retomável. | ~70 min |
| 3b | `traduzir_cauda.py` | → `03_lemas_cauda.parquet` | Docs en/es (~3,5 mil): fastText re-detecta, Opus-MT (en) ou NLLB (outros) traduzem, spaCy lematiza. | minutos em GPU; horas em CPU |
| 4 | `ngramas.py` | → `04_bigramas.csv`, `04_trigramas.csv` | Contagem de bi/trigramas (freq ≥ 50) com PMI, LLR e filtro de lixo. | ~15 min |
| 5 | `injetar.py` | → `dados/corpus_td_lemas.parquet` | Retém n-gramas (LLR ≥ mediana, sem stopwords) e os anexa ao texto como `a_b`. Monta o esquema final. | ~5 min |

`construir.py` encadeia tudo, com `--from`/`--ate` para retomar ou parar num passo e
`--sem-cauda` para pular a tradução. `comparar.py` confronta um corpus reconstruído
com o de referência, documento a documento.

## Requisitos

```bash
pip install -e . -r requirements-corpus.txt
python -m spacy download pt_core_news_lg
pip install -r requirements-corpus-traducao.txt   # só para o passo 3b (torch, transformers)
```

- **Banco:** Postgres `tarrafadb_sucupira`, tabela `teses_completas` (`id`, `ano`, `dados` JSONB).
  Normalmente acessado por túnel SSH em `127.0.0.1:5432`. Credenciais por variáveis
  de ambiente da libpq: `PGPASSWORD` (obrigatória), `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`.
- **Máquina:** 32 GB de RAM (a contagem de n-gramas e a lematização foram dimensionadas
  para isso), ~10 GB de disco em `dados/interim/`, GPU com 8 GB para a cauda.
- **Downloads na primeira execução:** modelo spaCy (~570 MB), fastText `lid.176.bin`
  (~131 MB, automático), Opus-MT (~300 MB) e NLLB-600M (~2,4 GB) via Hugging Face.

## Execução

```bash
export PGPASSWORD=...          # túnel SSH já aberto
python corpus/construir.py     # ~2 h; escreve dados/corpus_td_lemas.parquet
python corpus/comparar.py dados/corpus_td_lemas.parquet REFERENCIA.parquet   # opcional
```

Se um passo cair, corrija e retome com `--from <passo>`. A lematização retoma
sozinha das partes já gravadas em `dados/interim/_lemas_parts/`.

## Decisões replicadas da origem (não mudar sem retreinar)

- Texto de caracterização = título + resumo; documentos sem resumo (ou com menos de 100 caracteres) ficam fora.
- Grande área "Interdisciplinar" (rótulo de grande área, raro) é excluída; a *área de avaliação* Interdisciplinar fica.
- Idioma por langid restrito a três línguas; a cauda en/es é traduzida, não descartada.
- Lemas: POS ∈ {NOUN, PROPN, ADJ, VERB}, minúsculos, ≥ 2 caracteres com letra.
- N-gramas: piso de frequência 50; bigramas cortados na mediana do LLR; lixo (markup, códigos) descartado; lista de n-gramas-stopword em `reproduzir/stopwords/`.
- As stopwords de unigramas **não** são aplicadas aqui; entram só no treino (analyzer).

## Resultado da reconstrução de referência (7 set 2026)

Execução completa em 135 min (extração 15, idioma 5, lematização 91, cauda 22 em GPU,
n-gramas 12, injeção 5), comparada com `dados/corpus_td_lemas.parquet` por `comparar.py`:

| Item | Resultado |
|---|---|
| Documentos | 1.017.734 nos dois lados, sem faltas nem sobras |
| Rótulos, ano, quadriênio, idioma, n_palavras, id_programa | 100% iguais |
| Chave canônica | 100% iguais (após calcular o `lower()` no banco; ver abaixo) |
| Lemas (unigramas) dos 1.014.243 docs em português | 100% iguais |
| Lemas dos 3.491 docs traduzidos | 2.918 iguais; 573 com diferenças (tradução em GPU) |
| N-gramas retidos | 352.351 contra 352.363 na referência; 98 diferem na fronteira do corte |
| `lemmas_ext` exato | 99,3% dos docs; os 0,7% restantes diferem em 1 ou 2 tokens de n-grama |

Leitura: a parte determinística do pipeline (extração, idioma, lematização, chave)
reproduz o corpus original exatamente. A tradução da cauda não é bit a bit reprodutível
e, como os n-gramas são contados sobre o corpus inteiro, ela desloca uma centena de
n-gramas na fronteira de retenção (frequência 50 ou mediana do LLR), o que toca 0,6%
dos documentos em um token. Para o treino do classificador, o efeito é desprezível.

## Limites da reprodução

- A ordem das linhas pode diferir da referência (o banco não garante ordem física). Compare por `id_producao`.
- O `lower()` do Postgres e o do Python divergem em casos raros (sigma final grego, 18 resumos). A chave canônica usa o do banco, como a original.
- A lematização é determinística para a mesma versão do spaCy. A tradução da cauda (en/es, 0,34% do corpus) **não** é bit a bit reprodutível em GPU (fp16, composição dos lotes): num teste de 3 mil docs, 2 dos 8 traduzidos tiveram lemas ligeiramente diferentes da referência. Os 99,66% em português reproduzem exatamente.
- O banco muda quando é recoletado do Sucupira. O número de documentos elegíveis em set/2026 era 1.017.734.
