"""Ponte pro motor SUSC-20D -- reusa fit_firth/bootstrap/score_point já
validados (não reimplementa nada de estatística aqui). Precomputa o ponto
estimado + os 1000 draws de bootstrap UMA VEZ no startup da API (não a cada
request) -- é a otimização já anotada como necessária em
`revp_fase2_decisoes_design_contrato.md` seção 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SUSC20D_SCRIPTS = Path(__file__).resolve().parents[2] / "susc_20d_motor_inferencia_local_mvp_recife" / "scripts"
if str(_SUSC20D_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SUSC20D_SCRIPTS))

import numpy as np  # noqa: E402

from susc_20d_motor_escore import (  # noqa: E402
    FEATURE_COLS, LIMITATIONS, SEED, bootstrap_projection_draws, fit_firth,
    load_training_frame, score_point,
)


class RecifeEngine:
    """Estado precomputado (fit-once) do motor Recife. Instanciar uma vez no
    startup da API (`app.py`), não por request."""

    def __init__(self) -> None:
        self.kept, X, y, self.scaler = load_training_frame()
        Xs = self.scaler.transform(X)
        Xd = np.column_stack([np.ones(len(Xs)), Xs])
        res = fit_firth(Xd, y)
        self.beta_point = res["beta"]
        self.boot_draws = bootstrap_projection_draws(X, y, n_boot=1000, seed=SEED)
        self.known_points = [
            {"ponto_id": j["ponto_id"], "lat": j["lat"], "lon": j["lon"],
             "rotulo": j["rotulo"], **{f: j[f] for f in FEATURE_COLS}}
            for j in self.kept
        ]

    def score_known_point(self, point: dict) -> dict:
        return score_point(point, self.scaler, self.beta_point, self.boot_draws)

    def limitations(self) -> list[str]:
        return list(LIMITATIONS)
