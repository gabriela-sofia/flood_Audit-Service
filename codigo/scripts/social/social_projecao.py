"""Reprojeção de geometrias shapely sem geopandas, para o caminho servido das medidas sociais.

`geopandas.GeoSeries.to_crs` faz, por dentro, `pyproj.Transformer.from_crs(origem, destino, always_xy=True)` aplicado
às coordenadas de cada geometria. Aqui o mesmo transformador é aplicado com `shapely.transform` (vetorizado, mantém
tipo, anéis e ordem dos vértices). Não há simplificação, bbox nem densificação: os polígonos são os mesmos vértices,
em outro CRS. A equivalência com o geopandas é verificada fora deste processo (ambiente que tem geopandas) por
`scripts/planejamento/pt_social_equivalencia.py`.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import shapely
from pyproj import CRS, Transformer


@lru_cache(maxsize=8)
def _transformador(origem: int, destino: int) -> Transformer:
    if not CRS.from_epsg(destino).is_projected:
        raise ValueError("destino precisa ser CRS projetado para medir área")
    return Transformer.from_crs(origem, destino, always_xy=True)


def reprojetar(geometrias, origem: int = 4326, destino: int = 6933) -> list:
    """Lista de geometrias shapely em `origem` → mesma lista em `destino` (sem mudar vértices nem tipos)."""
    t = _transformador(origem, destino)

    def _xy(c: np.ndarray) -> np.ndarray:
        x, y = t.transform(c[:, 0], c[:, 1])
        return np.column_stack([x, y])
    arr = np.asarray(list(geometrias), dtype=object)
    if arr.size == 0:
        return []
    out = shapely.transform(arr, _xy)
    if not np.all(np.isfinite(shapely.get_coordinates(out))):
        raise ValueError("coordenada nao finita apos reprojecao")
    return list(out)
