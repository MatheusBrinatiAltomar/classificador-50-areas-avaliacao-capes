"""Exemplo de uso da texto2area: texto cru (PT) -> grande area CAPES."""
from texto2area import classificar

TEXTOS = [
    "Biblioteca Python que classifica o titulo/resumo de uma tese ou dissertacao brasileira em uma das 9 grandes areas de avaliacao CAPES, a partir do texto (em portugues), de forma ponta a ponta.",
]

for txt in TEXTOS:
    area, margens, termos = classificar(txt)
    print(f"{area} | margem {margens} | termos: {termos}")
