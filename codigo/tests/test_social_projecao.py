"""Reprojeção do caminho social servido sem geopandas: mesma operação do GeoSeries.to_crs (pyproj always_xy)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import shapely
from pyproj import Transformer
from shapely.geometry import Polygon, box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "social"))

from social_projecao import reprojetar  # noqa: E402


def test_mesmos_vertices_que_o_transformador_always_xy_com_furo():
    pol = Polygon([(-49.3, -25.5), (-49.2, -25.5), (-49.2, -25.4), (-49.3, -25.4)],
                  [[(-49.27, -25.47), (-49.23, -25.47), (-49.23, -25.43), (-49.27, -25.43)]])
    out = reprojetar([pol])[0]
    t = Transformer.from_crs(4326, 6933, always_xy=True)
    c = shapely.get_coordinates(pol)
    x, y = t.transform(c[:, 0], c[:, 1])
    assert out.geom_type == "Polygon" and len(out.interiors) == 1
    assert np.array_equal(shapely.get_coordinates(out), np.column_stack([x, y]))


def test_lista_vazia_e_destino_geografico_recusado():
    assert reprojetar([]) == []
    with pytest.raises(ValueError, match="projetado"):
        reprojetar([box(0, 0, 1, 1)], 4326, 4326)


def test_coordenada_nao_finita_e_recusada():
    with pytest.raises(ValueError, match="nao finita"):
        reprojetar([box(0, 89.9, 1, 90.5)])            # latitude fora do domínio vira inf no EASE


def test_consulta_curitiba_sem_geopandas_usa_poligonos():
    s7 = pytest.importorskip("social07_servico_medidas")
    if not (s7.PACOTE / "manifesto.json").exists():
        pytest.skip("pacote social ausente")
    r = s7.consultar("curitiba", {"lat_min": -25.609, "lon_min": -49.359, "lat_max": -25.601, "lon_max": -49.351})
    ctx = r["exposicao"]["contexto_aoi"]
    assert r["setores_intersectados"] > 0 and ctx["unidades_intersectadas"]
    assert all(0 < u["fracao_area"] <= 1 for u in ctx["unidades_intersectadas"])
    assert r["escore_fisico_alterado"] is False


def test_equivalencia_registrada_com_o_geopandas():
    p = ROOT / "local_runs" / "planejamento-tese" / "v2" / "social" / "equivalencia.json"
    if not p.exists():
        pytest.skip("rode: python -X utf8 scripts/planejamento/pt.py social_equivalencia")
    e = json.loads(p.read_text(encoding="utf-8"))
    assert e["veredito"] == "equivalente" and e["divergencias_total"] == 0 and not e["binarios_misturados"]
    assert all(g["dif_max_coordenada_m"] == 0.0 for g in e["geometria_por_regiao"].values())
