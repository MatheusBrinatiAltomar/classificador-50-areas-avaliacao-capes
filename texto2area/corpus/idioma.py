"""Passo 2 — detecção de idioma (langid restrito a pt/en/es) e normalização NFC.

Replica a etapa 2 da origem para o corpus de teses:
  - idioma: langid (norm_probs) sobre o texto em minúsculas com espaços colapsados,
    restrito a {pt, en, es} (títulos curtos geram falsos positivos sem restrição);
  - texto: NFC + remoção de caracteres de controle + colapso de espaços;
  - n_palavras: contagem por split;
  - sem limite de comprimento (título+resumo é longo por desenho); docs vazios são descartados.

O fastText (lid.176) da origem só alimentava uma flag de revisão que não chega ao
corpus final; aqui ele é usado apenas no passo da cauda (roteamento da tradução).

Entrada: dados/interim/01_bruto.parquet
Saída:   dados/interim/02_normalizado.parquet  (+ idioma, confianca, texto, n_palavras)
"""
from __future__ import annotations

from multiprocessing import Pool

import pyarrow as pa
import pyarrow.parquet as pq

from _comum import (BRUTO, NORMALIZADO, IDIOMAS, N_PROCESS, CHUNK,
                    normalizar_nfc, texto_para_idioma, iter_parquet, log, exigir)

_ID = None


def _init():
    global _ID
    from langid.langid import LanguageIdentifier, model
    _ID = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    _ID.set_languages(IDIOMAS)


def _detectar(textos: list[str]) -> list[tuple[str, float]]:
    out = []
    for t in textos:
        s = texto_para_idioma(t)
        if not s:
            out.append(("", 0.0)); continue
        lang, prob = _ID.classify(s)
        out.append((lang, float(prob)))
    return out


def main() -> None:
    exigir(BRUTO, "rode corpus/extrair.py")
    schema = pa.schema([
        ("id_producao", pa.string()), ("id_programa", pa.string()), ("chave_canonica", pa.string()),
        ("grande_area", pa.string()), ("area_avaliacao", pa.string()),
        ("ano_publicacao", pa.int16()), ("quadrienio", pa.string()), ("idioma_sucupira", pa.string()),
        ("idioma", pa.string()), ("confianca", pa.float32()),
        ("n_palavras", pa.int32()), ("texto", pa.large_string()),
    ])
    n = vazios = 0
    from collections import Counter
    cont = Counter()
    with Pool(N_PROCESS, initializer=_init) as pool, \
            pq.ParquetWriter(NORMALIZADO, schema, compression="zstd") as w:
        for df in iter_parquet(BRUTO, batch_size=CHUNK):
            textos = df["titulo_original"].fillna("").tolist()
            # detecção em paralelo (fatias por processo)
            k = max(1, len(textos) // (N_PROCESS * 4))
            fatias = [textos[i:i + k] for i in range(0, len(textos), k)]
            det = [x for fat in pool.map(_detectar, fatias) for x in fat]
            df["idioma"] = [d[0] for d in det]
            df["confianca"] = [d[1] for d in det]
            df["texto"] = [normalizar_nfc(t) for t in textos]
            df["n_palavras"] = [len(t.split()) for t in df["texto"]]
            m = df["texto"].str.len() > 0
            vazios += int((~m).sum())
            df = df[m]
            cont.update(df["idioma"])
            w.write_table(pa.Table.from_pandas(df.drop(columns=["titulo_original"]), schema=schema,
                                               preserve_index=False))
            n += len(df)
            log(f"  {n:,} docs | idiomas {dict(cont)}")
    log(f"OK {n:,} docs ({vazios} vazios descartados) -> {NORMALIZADO}")


if __name__ == "__main__":
    main()
