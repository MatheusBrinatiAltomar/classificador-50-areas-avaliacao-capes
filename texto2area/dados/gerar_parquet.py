"""
Gera o dataset de treino da texto2area a partir do corpus da Camada 1a (Tarrafa).

Este script documenta a PROVENIÊNCIA do arquivo `corpus_td_lemas.parquet`.
Ele só roda na máquina que tem o repositório da Camada 1a; a aluna/usuário
final NÃO precisa rodá-lo — basta baixar o parquet (ver README.md desta pasta).

Origem : <camada1a>/data/td/interim/corpus_consolidado_normalizado_pt_ext.parquet
Saída  : <texto2area>/dados/corpus_td_lemas.parquet

Transformações (e nada mais):
  - remove as colunas `idioma` (cópia de `idioma_original`), `lemmas` e `n_lemas`
    (unigramas sem n-gramas; não usados pelo modelo);
  - converte `ano_publicacao` e `n_palavras` de texto para inteiro (sem perda);
  - remove os espaços de preenchimento à direita de `area_avaliacao` (campo CHAR
    de largura fixa na origem; 60 rótulos distintos antes e depois);
  - mantém a ordem original das linhas.

Uso:  python dados/gerar_parquet.py [--origem CAMINHO]
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ORIGEM_PADRAO = Path("/home/rene/projetos/tarrafa/classprod/camada1a/data/td/interim/"
                     "corpus_consolidado_normalizado_pt_ext.parquet")
SAIDA = HERE / "corpus_td_lemas.parquet"

COLUNAS = [
    "id_producao", "id_programa", "chave_canonica",
    "grande_area", "area_avaliacao",
    "ano_publicacao", "quadrienio",
    "idioma_original", "n_palavras",
    "lemmas_ext", "n_lemas_ext",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--origem", type=Path, default=ORIGEM_PADRAO)
    args = ap.parse_args()

    print(f"Lendo {args.origem} ...", flush=True)
    t = pq.read_table(args.origem, columns=COLUNAS)
    n = t.num_rows

    t = t.set_column(t.schema.get_field_index("ano_publicacao"), "ano_publicacao",
                     pc.cast(t["ano_publicacao"], pa.int16()))
    t = t.set_column(t.schema.get_field_index("n_palavras"), "n_palavras",
                     pc.cast(t["n_palavras"], pa.int32()))
    t = t.set_column(t.schema.get_field_index("area_avaliacao"), "area_avaliacao",
                     pc.utf8_trim_whitespace(t["area_avaliacao"]))
    t = t.set_column(t.schema.get_field_index("n_lemas_ext"), "n_lemas_ext",
                     pc.cast(t["n_lemas_ext"], pa.int32()))
    # strings largas -> strings comuns nas colunas curtas (lemmas_ext fica large_string:
    # a coluna passa de 2 GB, limite do tipo string por bloco)
    for c in ("id_producao", "id_programa", "chave_canonica", "grande_area",
              "area_avaliacao", "quadrienio", "idioma_original"):
        t = t.set_column(t.schema.get_field_index(c), c, pc.cast(t[c], pa.string()))

    # metadados embutidos no arquivo (aparecem em pq.read_metadata)
    meta = {
        b"dataset": b"texto2area/corpus_td_lemas",
        b"origem": b"Catalogo de Teses e Dissertacoes CAPES (Sucupira), 2013-2024; pipeline Tarrafa/Camada 1a",
        b"texto": b"lemmas_ext = lemas (NOUN/PROPN/ADJ/VERB, spaCy pt_core_news_lg) de titulo+resumo + n-gramas retidos unidos por '_'",
        b"documentacao": b"README.md na pasta dados/ do repositorio texto2area",
    }
    t = t.replace_schema_metadata(meta)  # descarta o metadado "pandas" herdado (declarava tipos antigos)

    print(f"Gravando {SAIDA} ({n:,} linhas, {t.num_columns} colunas) ...", flush=True)
    pq.write_table(t, SAIDA, compression="zstd", compression_level=9, row_group_size=100_000)

    h = hashlib.sha256()
    with SAIDA.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    (HERE / "corpus_td_lemas.sha256").write_text(f"{h.hexdigest()}  corpus_td_lemas.parquet\n")
    print(f"OK  {SAIDA.stat().st_size/1e6:,.0f} MB  sha256={h.hexdigest()[:16]}...", flush=True)


if __name__ == "__main__":
    main()
