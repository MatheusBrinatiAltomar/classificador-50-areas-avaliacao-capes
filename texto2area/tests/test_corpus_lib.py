"""Testes das funções puras da construção do corpus (não precisam de banco nem de modelos)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "corpus"))

from ngramas_lib import ngrama_e_lixo, juntar_token, gerar_ngramas, calcular_llr_bigrama  # noqa: E402
from _comum import normalizar_nfc, texto_para_idioma  # noqa: E402
from extrair import chave_canonica  # noqa: E402


def test_lixo():
    assert not ngrama_e_lixo("saúde coletivo")
    assert not ngrama_e_lixo("covid-19 pandemia")      # alfanumérico com radical válido
    assert ngrama_e_lixo("math mrow")                   # marcador de markup
    assert ngrama_e_lixo("a b")                         # < 2 letras
    assert ngrama_e_lixo("c57bl/6 camundongo")          # símbolo fora de [alfa, dígito, '-']


def test_ngramas():
    assert list(gerar_ngramas(["a", "b", "c"], 2)) == [("a", "b"), ("b", "c")]
    assert list(gerar_ngramas(["a"], 2)) == []
    assert juntar_token(("saúde", "coletivo")) == "saúde_coletivo"
    assert calcular_llr_bigrama(100, 200, 300, 10_000) > calcular_llr_bigrama(10, 200, 300, 10_000)


def test_normalizacao():
    assert normalizar_nfc("  a\x00b   c\n") == "a b c"
    assert texto_para_idioma(" Título  X ") == "título x"


def test_chave_canonica():
    k = chave_canonica("título. resumo", 2020)            # entrada já em minúsculas (lower() do banco)
    assert k.startswith("hash:") and len(k) == 21
    assert k == chave_canonica("título. resumo", 2020)
    assert k != chave_canonica("título. resumo", 2021)
    assert k != chave_canonica("Título. Resumo", 2020)     # a função NÃO baixa caixa: isso é do SQL
