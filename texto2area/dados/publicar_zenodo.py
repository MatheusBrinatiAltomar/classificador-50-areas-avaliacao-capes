"""Deposita o corpus de treino no Zenodo como um DATASET (registro próprio, com DOI).

Cria um rascunho (draft) com metadados e arquivos e NÃO publica: revise no site e
clique em "Publish", ou rode de novo com --publicar. Idempotente por --deposition-id
(reaproveita um rascunho existente em vez de criar outro).

Arquivos enviados (desta pasta): corpus_td_lemas.parquet, corpus_td_lemas.sha256,
rotulos_area_avaliacao.csv, README.md.

Uso:
  export ZENODO_TOKEN=...            # token pessoal com escopos deposit:write e deposit:actions
  python dados/publicar_zenodo.py                 # cria o rascunho e envia os arquivos
  python dados/publicar_zenodo.py --sandbox       # idem em sandbox.zenodo.org (para testar)
  python dados/publicar_zenodo.py --deposition-id 123 --publicar   # publica um rascunho revisado
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ARQUIVOS = ["corpus_td_lemas.parquet", "corpus_td_lemas.sha256", "rotulos_area_avaliacao.csv", "README.md"]
SOFTWARE_DOI = "10.5281/zenodo.21082954"

METADATA = {
    "upload_type": "dataset",
    "title": "Corpus de teses e dissertações brasileiras (2013–2024) lematizado, com grande área e área de avaliação CAPES — dados de treino da texto2area",
    "creators": [{"name": "Veloso, Renê Rodrigues", "affiliation": "Universidade Estadual de Montes Claros (UNIMONTES)"}],
    "description": (
        "<p>1.017.734 teses e dissertações do Catálogo de Teses e Dissertações da CAPES (Plataforma Sucupira), "
        "2013–2024, com o texto de título + resumo já pré-processado (lemas de conteúdo via spaCy pt_core_news_lg "
        "e n-gramas retidos unidos por '_'), rótulos de grande área (9) e de área de avaliação (60 nomes históricos), "
        "ano, quadriênio e idioma original. É o conjunto exato que treinou e avaliou o classificador publicado na "
        "biblioteca <a href='https://doi.org/" + SOFTWARE_DOI + "'>texto2area</a>.</p>"
        "<p>Um único arquivo parquet (806 MB, zstd) mais um CSV com os rótulos de área de avaliação e a documentação "
        "completa das colunas e do pré-processamento (README.md). Títulos e resumos originais não são redistribuídos; "
        "podem ser recuperados pelo id_producao nos dados abertos da CAPES.</p>"
        "<p>Uso pretendido: treinar/avaliar classificadores de área a partir do texto (ex.: estender a texto2area "
        "às 50 áreas de avaliação). Licença dos dados derivados: CC BY 4.0; fonte primária: CAPES (dados abertos).</p>"
    ),
    "access_right": "open",
    "license": "cc-by-4.0",
    "language": "por",
    "keywords": ["teses e dissertações", "CAPES", "Sucupira", "grandes áreas", "áreas de avaliação",
                 "classificação de texto", "corpus", "português", "lematização", "texto2area"],
    "related_identifiers": [
        {"identifier": SOFTWARE_DOI, "relation": "isSupplementTo", "resource_type": "software"},
        {"identifier": "https://github.com/reneveloso/texto2area", "relation": "isSupplementTo", "resource_type": "software"},
    ],
    "version": "1.0.0",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", action="store_true")
    ap.add_argument("--deposition-id", type=int, default=None, help="reaproveita um rascunho existente")
    ap.add_argument("--publicar", action="store_true", help="publica (irreversível: gera o DOI)")
    ap.add_argument("--sem-arquivos", action="store_true", help="só metadados (não reenviar arquivos)")
    args = ap.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit("Defina ZENODO_TOKEN (token pessoal do Zenodo, escopos deposit:write + deposit:actions).")
    base = "https://sandbox.zenodo.org/api" if args.sandbox else "https://zenodo.org/api"
    s = requests.Session(); s.params = {"access_token": token}

    for a in ARQUIVOS:
        if not (HERE / a).exists():
            sys.exit(f"arquivo ausente: {HERE / a}")

    if args.deposition_id:
        r = s.get(f"{base}/deposit/depositions/{args.deposition_id}"); r.raise_for_status(); dep = r.json()
        print(f"rascunho existente: {dep['links']['html']}")
    else:
        r = s.post(f"{base}/deposit/depositions", json={}); r.raise_for_status(); dep = r.json()
        print(f"rascunho criado: id={dep['id']} {dep['links']['html']}")

    r = s.put(f"{base}/deposit/depositions/{dep['id']}", json={"metadata": METADATA})
    if r.status_code >= 400:
        sys.exit(f"metadados rejeitados: {r.status_code} {r.text[:800]}")
    print("metadados gravados")

    if not args.sem_arquivos:
        bucket = r.json()["links"]["bucket"]
        existentes = {f["filename"] for f in s.get(f"{base}/deposit/depositions/{dep['id']}/files").json()}
        for a in ARQUIVOS:
            p = HERE / a
            if a in existentes:
                print(f"  já enviado: {a}"); continue
            print(f"  enviando {a} ({p.stat().st_size/1e6:,.0f} MB) ...", flush=True)
            with p.open("rb") as fh:
                rr = s.put(f"{bucket}/{a}", data=fh)
            if rr.status_code >= 400:
                sys.exit(f"falha no envio de {a}: {rr.status_code} {rr.text[:300]}")
            print(f"    ok checksum={rr.json().get('checksum')}")

    if args.publicar:
        r = s.post(f"{base}/deposit/depositions/{dep['id']}/actions/publish")
        if r.status_code >= 400:
            sys.exit(f"publicação falhou: {r.status_code} {r.text[:800]}")
        d = r.json()
        print(f"PUBLICADO: DOI {d['doi']}  concept DOI {d.get('conceptdoi')}  {d['links']['html']}")
        print("Agora substitua '[Zenodo — DOI a incluir]' em README.md e dados/README.md pelo DOI acima.")
    else:
        print(f"\nRascunho pronto para revisão: {dep['links']['html']}\n"
              f"Para publicar: python dados/publicar_zenodo.py --deposition-id {dep['id']} --sem-arquivos --publicar")


if __name__ == "__main__":
    main()
