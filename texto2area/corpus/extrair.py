"""Passo 1 — extração das teses e dissertações do banco Sucupira (Postgres).

Consulta idêntica à da origem (sql/extrai_td.sql da Camada 1a): texto de
caracterização = título + ". " + resumo; mantém registros com grande área
informada e diferente de INTERDISCIPLINAR, resumo com >= 100 caracteres.
Acrescenta a chave canônica (hash de título+resumo normalizado | ano).

Conexão via variáveis de ambiente da libpq: PGHOST (padrão 127.0.0.1), PGPORT
(5432), PGDATABASE (tarrafadb_sucupira), PGUSER (postgres), PGPASSWORD (obrigatória).
Normalmente o banco é alcançado por um túnel SSH aberto antes.

Saída: dados/interim/01_bruto.parquet  (uma linha por tese; ORDER BY id -> determinístico)
"""
from __future__ import annotations

import hashlib
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from _comum import BRUTO, ANO_INI, ANO_FIM, log

SQL = f"""
SELECT
  id::text                                                  AS id_producao,
  (dados->>'idPrograma')                                    AS id_programa,
  CASE upper(btrim(dados->>'nomeGrandeAreaConhecimento'))
    WHEN 'LINGÜÍSTICA, LETRAS E ARTES' THEN 'LINGUÍSTICA, LETRAS E ARTES'
    ELSE upper(btrim(dados->>'nomeGrandeAreaConhecimento'))
  END                                                       AS grande_area,
  btrim(dados->>'nomeAreaAvaliacao')                        AS area_avaliacao,
  ano                                                       AS ano_publicacao,
  CASE WHEN ano BETWEEN 2013 AND 2016 THEN '2013-2016'
       WHEN ano BETWEEN 2017 AND 2020 THEN '2017-2020'
       WHEN ano BETWEEN 2021 AND 2024 THEN '2021-2024' END  AS quadrienio,
  regexp_replace(btrim(coalesce(dados->>'nomeProducao','')||'. '||coalesce(dados->>'resumo','')),
                 '[\\n\\r\\t]+', ' ', 'g')                   AS titulo_original,
  dados->>'idioma'                                          AS idioma_sucupira,
  -- minúsculas calculadas NO BANCO (base do hash): o lower() do Postgres difere do
  -- Python em casos raros (ex.: sigma final grego), e a chave original usou o do banco
  lower(regexp_replace(btrim(coalesce(dados->>'nomeProducao','')||'. '||coalesce(dados->>'resumo','')),
                       '[\\n\\r\\t]+', ' ', 'g'))            AS titulo_lower
FROM teses_completas
WHERE dados->>'nomeGrandeAreaConhecimento' IS NOT NULL
  AND upper(btrim(dados->>'nomeGrandeAreaConhecimento')) <> 'INTERDISCIPLINAR'
  AND length(coalesce(dados->>'resumo','')) >= 100
  AND ano BETWEEN {ANO_INI} AND {ANO_FIM}
ORDER BY id
"""

SCHEMA = pa.schema([
    ("id_producao", pa.string()), ("id_programa", pa.string()), ("chave_canonica", pa.string()),
    ("grande_area", pa.string()), ("area_avaliacao", pa.string()),
    ("ano_publicacao", pa.int16()), ("quadrienio", pa.string()),
    ("idioma_sucupira", pa.string()), ("titulo_original", pa.large_string()),
])


def chave_canonica(titulo_lower: str, ano: int) -> str:
    """'hash:' + sha1(titulo_lower | ano | issn='')[:16]  (DEC-004 da origem, sem DOI).

    `titulo_lower` deve vir do lower() do Postgres (ver SQL), não do Python."""
    raw = f"{titulo_lower or ''}|{ano}|"
    return "hash:" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limite", type=int, default=None, help="só as N primeiras linhas (testes)")
    args = ap.parse_args()
    try:
        import psycopg
    except ImportError:
        sys.exit("psycopg ausente: pip install 'psycopg[binary]'")
    if not os.environ.get("PGPASSWORD"):
        sys.exit("Defina PGPASSWORD (e, se preciso, PGHOST/PGPORT/PGDATABASE/PGUSER).")
    os.environ.setdefault("PGHOST", "127.0.0.1")
    os.environ.setdefault("PGPORT", "5432")
    os.environ.setdefault("PGDATABASE", "tarrafadb_sucupira")
    os.environ.setdefault("PGUSER", "postgres")

    BRUTO.parent.mkdir(parents=True, exist_ok=True)
    log(f"conectando a {os.environ['PGUSER']}@{os.environ['PGHOST']}:{os.environ['PGPORT']}/{os.environ['PGDATABASE']}")
    n = 0
    with psycopg.connect("") as conn, conn.cursor(name="extrai_td") as cur, \
            pq.ParquetWriter(BRUTO, SCHEMA, compression="zstd") as w:
        cur.itersize = 20_000
        cur.execute(SQL + (f" LIMIT {int(args.limite)}" if args.limite else ""))
        lote = {k: [] for k in SCHEMA.names}
        for r in cur:
            id_prod, id_prog, ga, aa, ano, quad, txt, idi, tl = r
            lote["id_producao"].append(id_prod); lote["id_programa"].append(id_prog)
            lote["chave_canonica"].append(chave_canonica(tl, ano))
            lote["grande_area"].append(ga); lote["area_avaliacao"].append(aa)
            lote["ano_publicacao"].append(ano); lote["quadrienio"].append(quad)
            lote["idioma_sucupira"].append(idi); lote["titulo_original"].append(txt)
            n += 1
            if n % 50_000 == 0:
                w.write_table(pa.table(lote, schema=SCHEMA)); lote = {k: [] for k in SCHEMA.names}
                log(f"  {n:,} linhas")
        if lote["id_producao"]:
            w.write_table(pa.table(lote, schema=SCHEMA))
    log(f"OK {n:,} linhas -> {BRUTO} ({BRUTO.stat().st_size/1e6:,.0f} MB)")


if __name__ == "__main__":
    main()
