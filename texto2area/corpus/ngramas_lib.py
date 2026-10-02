"""Funções puras de n-grama (cópia fiel de src/etapa3_tokenizacao/ngramas_lib.py da origem).

Tokens de n-grama são unidos por '_' para sobreviver ao split() do analyzer.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterator

_OCR_MARKERS = {"math", "mrow", "mi", "mo", "msub", "display", "inline",
                "amp", "lt", "gt", "n.o", "n°", "ue"}
_RE_TEM_DIGITO = re.compile(r"\d")


def calcular_llr_bigrama(c_xy: int, c_x: int, c_y: int, N: int) -> float:
    """Log-likelihood ratio (Dunning 1993) para o bigrama (x, y)."""
    def L(k: int, n: int, p: float) -> float:
        if p <= 0 or p >= 1 or k < 0 or k > n:
            return 0.0
        return k * math.log(p) + (n - k) * math.log(1 - p)

    if c_xy <= 0 or c_x <= 0 or c_y <= 0 or N <= 0:
        return 0.0
    p = c_y / N
    p1 = c_xy / c_x if c_x else 0
    p2 = (c_y - c_xy) / (N - c_x) if (N - c_x) > 0 else 0
    return -2 * (L(c_xy, c_x, p) + L(c_y - c_xy, N - c_x, p)
                 - L(c_xy, c_x, p1) - L(c_y - c_xy, N - c_x, p2))


def gerar_ngramas(tokens: list[str], n: int) -> Iterator[tuple[str, ...]]:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i + n])


def juntar_token(ngrama) -> str:
    if isinstance(ngrama, str):
        ngrama = ngrama.split()
    return "_".join(ngrama)


def ngrama_e_lixo(ngrama) -> bool:
    """True se o n-grama deve ser descartado (OCR/markup/código/curto). Ver DEC-022 da origem."""
    toks = ngrama.split() if isinstance(ngrama, str) else list(ngrama)
    if not toks:
        return True
    for t in toks:
        if t in _OCR_MARKERS:
            return True
        alpha = [c for c in t if c.isalpha()]
        if len(alpha) < 2:
            return True
        if _RE_TEM_DIGITO.search(t):
            if len(alpha) < 2 or not all(c.isalpha() or c.isdigit() or c == "-" for c in t):
                return True
            continue
        if not all(c.isalpha() or c == "-" for c in t):
            return True
    return False
