"""Passo 3b — cauda não-pt: traduz para português e lematiza (~0,34% do corpus).

Replica scripts/incorporar_cauda_td.py da origem (decisão de 2026-06-23):
  - docs com idioma (langid) != 'pt';
  - idioma re-detectado com fastText lid.176 sobre o texto em minúsculas, para rotear:
      en -> Opus-MT (Helsinki-NLP/opus-mt-tc-big-en-pt); outros -> NLLB-200 distilled 600M;
      re-detectado como pt -> sem tradução;
  - max_length=512 tokens, decodificação gulosa, fp16 em GPU;
  - lematização spaCy da tradução (mesma regra do passo 3);
  - idioma_original = idioma do langid; idioma = 'pt'.

Requer torch + transformers (+ sentencepiece, sacremoses). GPU recomendada (minutos);
em CPU leva horas. Sem esta etapa o corpus fica só com os docs em pt (--sem-cauda no construir).

Entrada: dados/interim/02_normalizado.parquet
Saída:   dados/interim/03_lemas_cauda.parquet
"""
from __future__ import annotations

import sys
import urllib.request
from collections import defaultdict

import pandas as pd

from _comum import (NORMALIZADO, LEMAS_CAUDA, MODELOS, META_COLS, texto_para_idioma,
                    filtrar_lemas, carregar_spacy, iter_parquet, log, exigir)

OPUS = "Helsinki-NLP/opus-mt-tc-big-en-pt"
NLLB = "facebook/nllb-200-distilled-600M"
NLLB_CODE = {"es": "spa_Latn", "fr": "fra_Latn", "gl": "glg_Latn", "it": "ita_Latn",
             "de": "deu_Latn", "ca": "cat_Latn", "en": "eng_Latn"}
MAXLEN = 512
LID_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
LID = MODELOS / "lid.176.bin"
OUT_COLS = META_COLS + ["idioma", "n_palavras", "n_lemas", "lemmas", "idioma_original"]


def _fasttext():
    import fasttext
    if not LID.exists():
        MODELOS.mkdir(parents=True, exist_ok=True)
        log(f"baixando {LID_URL} (~131 MB)")
        urllib.request.urlretrieve(LID_URL, LID.with_suffix(".part"))
        LID.with_suffix(".part").replace(LID)
    return fasttext.load_model(str(LID))


def main() -> None:
    exigir(NORMALIZADO, "rode corpus/idioma.py")
    try:
        import torch
    except ImportError:
        sys.exit("torch/transformers ausentes: pip install -r requirements-corpus-traducao.txt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    parts = [df[df["idioma"] != "pt"] for df in iter_parquet(NORMALIZADO)]
    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    df["low"] = df["texto"].map(texto_para_idioma)
    log(f"device={dev} | cauda: {len(df):,} docs | langid {df['idioma'].value_counts().to_dict()}")

    fm = _fasttext()
    labs, _ = fm.predict(df["low"].tolist(), k=1)
    df["idi"] = [l[0].replace("__label__", "") for l in labs]
    log(f"  fastText: {df['idi'].value_counts().head(6).to_dict()}")

    def opus(txts):
        from transformers import MarianMTModel, MarianTokenizer
        tok = MarianTokenizer.from_pretrained(OPUS)
        mt = MarianMTModel.from_pretrained(OPUS).to(dev).eval()
        if dev == "cuda":
            mt = mt.half()
        out = []; B = 16
        for i in range(0, len(txts), B):
            b = tok(txts[i:i + B], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
            with torch.no_grad():
                o = mt.generate(**b, num_beams=1, max_new_tokens=MAXLEN)
            out += tok.batch_decode(o, skip_special_tokens=True)
            if (i // B) % 20 == 0:
                log(f"    opus {min(i + B, len(txts)):,}/{len(txts):,}")
        del mt
        if dev == "cuda":
            torch.cuda.empty_cache()
        return out

    def nllb(txts, srcs):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(NLLB)
        mt = AutoModelForSeq2SeqLM.from_pretrained(NLLB).to(dev).eval()
        if dev == "cuda":
            mt = mt.half()
        por = tok.convert_tokens_to_ids("por_Latn")
        res = [None] * len(txts); g = defaultdict(list)
        for i, s in enumerate(srcs):
            g[s].append(i)
        B = 16
        for s, idxs in g.items():
            tok.src_lang = s
            for j in range(0, len(idxs), B):
                sel = idxs[j:j + B]
                b = tok([txts[k] for k in sel], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
                with torch.no_grad():
                    o = mt.generate(**b, forced_bos_token_id=por, num_beams=1, max_new_tokens=MAXLEN)
                for k, d in zip(sel, tok.batch_decode(o, skip_special_tokens=True)):
                    res[k] = d
            log(f"    nllb {s}: {len(idxs):,} docs")
        del mt
        if dev == "cuda":
            torch.cuda.empty_cache()
        return res

    df["texto_pt"] = df["low"]
    en = df["idi"] == "en"
    if en.any():
        log(f"  traduzindo en->pt (Opus-MT, {int(en.sum()):,})")
        df.loc[en, "texto_pt"] = opus(df.loc[en, "low"].tolist())
    ot = (~en) & (df["idi"] != "pt")
    if ot.any():
        log(f"  traduzindo outros->pt (NLLB, {int(ot.sum()):,})")
        src = [NLLB_CODE.get(l, "spa_Latn") for l in df.loc[ot, "idi"]]
        df.loc[ot, "texto_pt"] = nllb(df.loc[ot, "low"].tolist(), src)

    log("  lematizando (spaCy pt)")
    nlp = carregar_spacy()
    lem = [filtrar_lemas(d) for d in nlp.pipe(df["texto_pt"].fillna("").tolist(), batch_size=32)]
    df["lemmas"] = [" ".join(l) for l in lem]
    df["n_lemas"] = [len(l) for l in lem]
    df["idioma_original"] = df["idioma"]
    df["idioma"] = "pt"
    df[OUT_COLS].to_parquet(LEMAS_CAUDA, index=False, compression="zstd")
    log(f"OK {len(df):,} docs -> {LEMAS_CAUDA}")


if __name__ == "__main__":
    main()
