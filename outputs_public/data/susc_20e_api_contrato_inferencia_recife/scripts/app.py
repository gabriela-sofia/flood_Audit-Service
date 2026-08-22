"""SUSC-20E -- API local do contrato de inferência (Fase 5 do
PLANO_ACAO_produto_v1.md), só depois de 1-3 terem números reais (já têm).

Implementa o rascunho v0 do contrato (`txtpragab.docx` / `contract_schema.py`)
com as duas decisões já tomadas na Fase 2
(`revp_fase2_decisoes_design_contrato.md`): CI = bootstrap preditivo (motor
SUSC-20D, já validado); DINO nunca soma ao score, só aparece em
`evidence.dino_*` quando o índice de patches estiver configurado.

Roda só localmente (uvicorn). Não é deploy, não é infraestrutura
compartilhada -- é o MVP local descrito na Fase 3, exposto por HTTP em vez de
chamado direto em Python, para provar o contrato de verdade antes de
qualquer decisão de hospedagem real.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from shapely.geometry import shape

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "susc_20d_motor_inferencia_local_mvp_recife" / "scripts"))

from contract_schema import Evidence, FeatureUsed, ScoreBlock, ScoreRequest, ScoreResponse  # noqa: E402
from engine_bridge import RecifeEngine  # noqa: E402
from gates import evaluate_gates  # noqa: E402
from region_registry import REGIONS  # noqa: E402
from susc_20d_score_engine import find_dino_evidence, load_dino_evidence_index  # noqa: E402

DATA_VERSION = "SUSC-20D/v12 (n=269) + Curitiba Leads A/B/C (parcial) -- 2026-07-24"

app = FastAPI(title="REV-P Contrato de Inferência (SUSC-20E, MVP local)")

_engine: RecifeEngine | None = None
_dino_bboxes: dict = {}
_dino_emb: dict = {}


@app.on_event("startup")
def _startup() -> None:
    global _engine, _dino_bboxes, _dino_emb
    _engine = RecifeEngine()
    _dino_bboxes, _dino_emb = load_dino_evidence_index()


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _weight_tier_to_contract(contrib_dict: dict) -> str:
    return contrib_dict["weight_tier"]


def _stability_for_feature(feat: str) -> str:
    # Achado já documentado em SUSC-20C/SUSC-20D: hand_m_dinf não tem sinal
    # robusto no bootstrap (48.7% de flips de sinal). Todas as outras 5
    # features físicas são consideradas robustas (mesmo sinal em v9->v12).
    return "unstable" if feat == "hand_m_dinf" else "robust"


def _parse_period_end(req: ScoreRequest) -> date | None:
    if req.period is None or not req.period.end:
        return None
    try:
        return datetime.strptime(req.period.end[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest) -> ScoreResponse:
    assert _engine is not None, "engine não inicializado (startup não rodou)"

    end_date = _parse_period_end(req)
    gate_result = evaluate_gates(req.region.geometry, req.region.crs, _engine.known_points, end_date)
    status = gate_result["status"]
    region_maturity = gate_result["region_maturity"]
    region = gate_result["region"]
    matched_points = gate_result["matched_points"]
    computed = gate_result["computed_features"]
    blockers = gate_result["blockers"]

    base_limitations = list(_engine.limitations()) if region == "recife" else []
    if region is not None and REGIONS[region].status_note:
        base_limitations = [REGIONS[region].status_note] + base_limitations
    if blockers:
        base_limitations = [f"gate_bloqueado: {b}" for b in blockers] + base_limitations

    if status != "ok":
        return ScoreResponse(
            request_id=req.request_id, status=status, region_maturity=region_maturity,
            score=ScoreBlock(value=None, confidence_interval=None,
                              model_version=(REGIONS[region].model_version if region else None)),
            features_used=[], evidence=Evidence(observational_points_used=0, sources=[]),
            limitations=base_limitations, data_version=DATA_VERSION, generated_at=_now_iso(),
        )

    # region == "recife" garantido pelo gate_model_valid_for_region (só Recife tem modelo).
    on_demand = computed is not None and not matched_points
    if matched_points:
        # Multiplos pontos conhecidos na geometria: usa o de label mais
        # incerto (score mais proximo de 0.5 no ponto) seria arbitrario;
        # reporta o primeiro e deixa explicito quantos existem via evidence.
        target = matched_points[0]
        out = _engine.score_known_point(target)
        target_lat, target_lon = float(target["lat"]), float(target["lon"])
    else:
        # SUSC-20F: geometria sem ponto conhecido, mas dentro da cobertura
        # real do DTM -- score calculado sob demanda (terreno + chuva ao vivo).
        assert computed is not None
        out = _engine.score_known_point(computed["features"])
        centroid = shape(req.region.geometry).centroid
        target_lon, target_lat = centroid.x, centroid.y

    features_used = [
        FeatureUsed(name=c["feature"],
                    contribution=_weight_tier_to_contract(c),
                    stability=_stability_for_feature(c["feature"]))
        for c in out["features_used"]
    ]

    dino_available = False
    dino_patch_id = None
    if _dino_bboxes and _dino_emb:
        try:
            ev = find_dino_evidence(target_lat, target_lon, _dino_bboxes, _dino_emb)
        except Exception:
            ev = None
        if ev:
            dino_available = True
            dino_patch_id = ev["patch_id"]

    if on_demand:
        sources = ["DTM PE3D merged (D-infinity, SUSC-20B)", "Open-Meteo ERA5-Land archive (ao vivo)"]
        base_limitations = [
            f"score calculado sob demanda (SUSC-20F), nao a partir de um ponto historico conhecido: {computed['provenance']['elevation_slope_caveat']}",
        ] + base_limitations
    else:
        sources = ["SEDEC", "Diário Oficial Recife", "ANA", "Global Flood Database"]

    evidence = Evidence(
        observational_points_used=len(matched_points),
        sources=sources,
        dino_embedding_available=dino_available,
        dino_patch_id=dino_patch_id,
    )

    return ScoreResponse(
        request_id=req.request_id, status="ok", region_maturity=region_maturity,
        score=ScoreBlock(value=out["score"], confidence_interval=out["confidence_interval"],
                          model_version=REGIONS["recife"].model_version),
        features_used=features_used, evidence=evidence,
        limitations=base_limitations, data_version=DATA_VERSION, generated_at=_now_iso(),
    )


@app.get("/regions")
def regions() -> dict:
    return {name: {"region_maturity": info.region_maturity, "model_version": info.model_version,
                    "status_note": info.status_note} for name, info in REGIONS.items()}
