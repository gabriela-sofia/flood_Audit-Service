"""Gates obrigatórios antes de qualquer inferência -- implementa a auditoria
gate-a-gate já decidida em `revp_fase2_decisoes_design_contrato.md`.

Atualização (SUSC-20F): o gate de features físicas agora tenta, nesta ordem,
(1) ponto(s) já conhecido(s) do v12 dentro da geometria (caminho rápido, com
rótulo histórico de contexto) e, se nenhum existir, (2) cálculo real sob
demanda (terreno amostrado de raster + chuva ao vivo) se o centróide da
geometria cair dentro da cobertura real do DTM merged -- ver
`susc_20f_pipeline_geoprocessamento_sob_demanda_recife/`. Fora dessa
cobertura: `insufficient_data` honesto, sem inventar.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from shapely.geometry import shape
from shapely.validation import explain_validity

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "susc_20f_pipeline_geoprocessamento_sob_demanda_recife" / "scripts"))
from motor_variaveis_sob_demanda import compute_features_on_demand, is_within_terrain_coverage  # noqa: E402

from registro_regioes import REGIONS, find_region_for_bbox

MIN_AREA_DEG2 = 1e-8  # area minima trivial (evita poligono degenerado/ponto)


def gate_geometry_valid(geojson: dict) -> tuple[bool, str]:
    try:
        geom = shape(geojson)
    except Exception as exc:
        return False, f"geometria_invalida: {exc}"
    if not geom.is_valid:
        return False, f"geometria_topologicamente_invalida: {explain_validity(geom)}"
    if geom.area < MIN_AREA_DEG2:
        return False, "geometria_abaixo_da_area_minima"
    return True, ""


def gate_crs_known(crs: str) -> tuple[bool, str]:
    if crs.strip().upper() != "EPSG:4326":
        return False, f"crs_nao_suportado: {crs} (só EPSG:4326 aceito nesta versão)"
    return True, ""


def gate_region_supported(geom) -> tuple[bool, str, str | None]:
    lon_min, lat_min, lon_max, lat_max = geom.bounds
    regiao = find_region_for_bbox(lon_min, lat_min, lon_max, lat_max)
    if regiao is None:
        return False, "regiao_fora_da_cobertura_conhecida (recife/curitiba/petropolis)", None
    return True, "", regiao


def gate_model_valid_for_region(regiao: str) -> tuple[bool, str]:
    info = REGIONS[regiao]
    if info.versao_modelo is None:
        return False, f"sem_modelo_estatistico_valido_para_regiao: {info.status_note}"
    return True, ""


def gate_physical_features_available(
    regiao: str, geom, known_points: list[dict], end_date: date,
) -> tuple[bool, str, list[dict], dict | None]:
    """DEM/HAND/TWI/chuva. Caminho 1 (rápido): ponto(s) conhecido(s) do v12
    dentro da geometria. Caminho 2 (SUSC-20F): se nenhum ponto conhecido
    casar, tenta cálculo real sob demanda no centróide da geometria --
    terreno amostrado de raster + chuva ao vivo (Open-Meteo). Retorna
    (ok, motivo, pontos_conhecidos_casados, features_computadas_sob_demanda)."""
    matched = [p for p in known_points if geom.contains(shape({
        "type": "Point", "coordinates": [float(p["lon"]), float(p["lat"])]}))]
    if matched:
        return True, "", matched, None

    centroid = geom.centroid
    lon, lat = centroid.x, centroid.y
    if not is_within_terrain_coverage(lat, lon):
        return False, "fora_da_cobertura_real_do_dtm_e_sem_ponto_conhecido_na_geometria", [], None

    computed = compute_features_on_demand(lat, lon, end_date)
    if computed is None:
        return False, "calculo_sob_demanda_falhou (terreno_ou_chuva_indisponivel_para_esta_coordenada_periodo)", [], None

    return True, "", [], computed


def evaluate_gates(geojson: dict, crs: str, known_points: list[dict], end_date: date | None = None) -> dict:
    """Avalia todos os portoes em ordem. Retorna dict com status final,
    region_maturity, regiao (ou None), pontos casados, features computadas sob
    demanda (ou None), e lista de motivos de bloqueio (vazia se tudo passou).
    `end_date` (default: hoje) delimita a janela de chuva pro cálculo sob
    demanda -- ver `susc_20f_pipeline_geoprocessamento_sob_demanda_recife/`."""
    if end_date is None:
        end_date = date.today()
    blockers: list[str] = []

    ok, reason = gate_geometry_valid(geojson)
    if not ok:
        return {"status": "insufficient_data", "region_maturity": "insufficient",
                "regiao": None, "matched_points": [], "computed_features": None, "blockers": [reason]}

    ok, reason = gate_crs_known(crs)
    if not ok:
        return {"status": "insufficient_data", "region_maturity": "insufficient",
                "regiao": None, "matched_points": [], "computed_features": None, "blockers": [reason]}

    geom = shape(geojson)
    ok, reason, matched_region = gate_region_supported(geom)
    if not ok or matched_region is None:
        return {"status": "region_not_supported", "region_maturity": "insufficient",
                "regiao": None, "matched_points": [], "computed_features": None, "blockers": [reason]}
    regiao: str = matched_region

    region_info = REGIONS[regiao]

    ok, reason = gate_model_valid_for_region(regiao)
    if not ok:
        blockers.append(reason)
        return {"status": "insufficient_data", "region_maturity": region_info.region_maturity,
                "regiao": regiao, "matched_points": [], "computed_features": None, "blockers": blockers}

    ok, reason, matched, computed = gate_physical_features_available(regiao, geom, known_points, end_date)
    if not ok:
        blockers.append(reason)
        return {"status": "insufficient_data", "region_maturity": region_info.region_maturity,
                "regiao": regiao, "matched_points": [], "computed_features": None, "blockers": blockers}

    return {"status": "ok", "region_maturity": region_info.region_maturity,
            "regiao": regiao, "matched_points": matched, "computed_features": computed, "blockers": []}
