"""Constrói o corpus de treino da texto2area, do banco Sucupira ao parquet final.

Passos (cada um é um script desta pasta, executado em processo próprio):
  extrair    -> 01_bruto.parquet        (Postgres via túnel; precisa de PGPASSWORD)
  idioma     -> 02_normalizado.parquet  (langid + NFC)
  lematizar  -> 03_lemas_pt.parquet     (spaCy; ~1h; retomável)
  cauda      -> 03_lemas_cauda.parquet  (tradução en/es -> pt + spaCy; GPU; opcional)
  ngramas    -> 04_bigramas.csv, 04_trigramas.csv
  injetar    -> corpus_td_lemas.parquet (final)

Uso:
  python corpus/construir.py                       # tudo
  python corpus/construir.py --from lematizar      # retoma a partir de um passo
  python corpus/construir.py --ate idioma          # para depois de um passo
  python corpus/construir.py --sem-cauda           # pula a tradução (corpus só pt)
  python corpus/construir.py --out dados/interim/novo.parquet --sobrescrever
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PASSOS = ["extrair", "idioma", "lematizar", "cauda", "ngramas", "injetar"]
SCRIPT = {"cauda": "traduzir_cauda.py"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="de", choices=PASSOS, default=PASSOS[0])
    ap.add_argument("--ate", choices=PASSOS, default=PASSOS[-1])
    ap.add_argument("--sem-cauda", action="store_true", help="não traduzir a cauda en/es (corpus só pt)")
    ap.add_argument("--out", type=Path, default=None, help="parquet final (passado ao injetar)")
    ap.add_argument("--sobrescrever", action="store_true")
    args = ap.parse_args()

    i0, i1 = PASSOS.index(args.de), PASSOS.index(args.ate)
    if i0 > i1:
        sys.exit("--from vem depois de --ate")
    t_total = time.time()
    for passo in PASSOS[i0:i1 + 1]:
        if passo == "cauda" and args.sem_cauda:
            print(f"\n##### {passo}: pulado (--sem-cauda)"); continue
        cmd = [sys.executable, str(HERE / SCRIPT.get(passo, f"{passo}.py"))]
        if passo == "injetar":
            if args.out:
                cmd += ["--out", str(args.out)]
            if args.sobrescrever:
                cmd += ["--sobrescrever"]
        print(f"\n##### {time.strftime('%H:%M:%S')} {passo}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"passo '{passo}' falhou (código {r.returncode}); corrija e retome com --from {passo}")
        print(f"##### {passo} concluído em {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"\n##### CONCLUÍDO em {(time.time()-t_total)/60:.1f} min")


if __name__ == "__main__":
    main()
