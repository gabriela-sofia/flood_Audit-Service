"""Schema de entrada/saída -- implementa literalmente o rascunho v0 do
contrato de inferência (`txtpragab.docx`, extraído em
`revp_fase2_decisoes_design_contrato.md`), com os dois pontos que a Fase 2
já decidiu (CI = bootstrap preditivo; DINO nunca soma ao score).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RegionGeometry(BaseModel):
    geometry: dict = Field(..., description="GeoJSON Polygon ou MultiPolygon")
    crs: str = Field(default="EPSG:4326")


class Period(BaseModel):
    start: str | None = None
    end: str | None = None


class ScoreRequest(BaseModel):
    request_id: str
    region: RegionGeometry
    period: Period | None = None
    requested_layers: list[str] = Field(default_factory=list)


class FeatureUsed(BaseModel):
    name: str
    contribution: Literal["high", "medium", "low"]
    stability: Literal["robust", "unstable"]


class ScoreBlock(BaseModel):
    value: float | None
    confidence_interval: list[float] | None
    model_version: str | None


class Evidence(BaseModel):
    observational_points_used: int
    sources: list[str]
    dino_embedding_available: bool = False
    dino_patch_id: str | None = None


class ScoreResponse(BaseModel):
    request_id: str
    status: Literal["ok", "insufficient_data", "region_not_supported"]
    region_maturity: Literal["available", "limited_evidence", "insufficient"]
    score: ScoreBlock
    features_used: list[FeatureUsed]
    evidence: Evidence
    limitations: list[str]
    data_version: str
    generated_at: str
