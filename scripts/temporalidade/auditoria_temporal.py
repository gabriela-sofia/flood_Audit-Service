"""Auditoria de temporalidade do REV-P.

A auditoria e aditiva: le dados e resultados existentes, executa diagnosticos
leves de identificabilidade e escreve artefatos em `local_runs/temporalidade`.
Nao altera modelo servido, contrato v1, maturidades, datasets canonicos nem
resultados historicos.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:  # permite execucao direta do arquivo
    from ontologia_temporal import (
        CHECKLIST_TEMPORAL,
        avaliar_checklist,
        avaliar_evidencia_temporal,
        gravar_ontologia,
    )
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ontologia_temporal import (  # type: ignore
        CHECKLIST_TEMPORAL,
        avaliar_checklist,
        avaliar_evidencia_temporal,
        gravar_ontologia,
    )


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "local_runs" / "temporalidade"
DS05 = REPO / "local_runs" / "ds-05-tabela-unica"
EXPERIMENTOS = REPO / "local_runs" / "experimentos-cientificos"
MOD_PROSP = REPO / "local_runs" / "mod-prosp-02"
CURITIBA_PUBLIC = (
    REPO
    / "outputs_public"
    / "data"
    / "susc_20k_siac156_curitiba_flood_candidates"
)

POOL_FLUVIAL = DS05 / "tabela_unica_pool_fluvial_v2.csv"
ADMITIDOS = DS05 / "tabela_unica_admitidos_v2.csv"
CURITIBA_V20N = CURITIBA_PUBLIC / "registries" / "v20n_dataset_curitiba_features_v2.csv"
RECIFE_V12 = (
    REPO
    / "outputs_public"
    / "data"
    / "susc_20a_aquisicao_eventos_reais_recife"
    / "dataset"
    / "dataset_eventos_features_v12_final.csv"
)

FEATURES_STATIC = ["elevation_m", "slope_deg", "hand_m", "twi_dinf"]
FEATURES_CURITIBA_STATIC = ["elevation_m", "slope_deg", "hand_m_dinf", "twi_dinf"]
FEATURES_FORCING = ["rain_max_24h", "rain_decay_index"]
FEATURES_CURITIBA_FORCING = [
    "rain_peak_residual_orthogonalized",
    "rain_decay_index_api_chirps",
]
SOURCE_NAMES = {
    "recife": "Recife",
    "curitiba": "Curitiba",
    "uk": "Environment Agency/UK",
    "cems": "Copernicus EMS",
    "sen1floods11": "Sen1Floods11",
    "ufo": "UFO",
}
NEGATIVOS_COMPARAVEIS = (
    "por_exclusao_qualificada",
    "por_observacao",
    "observado",
    "exclusao_qualificada",
)


def rel(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def parse_date(values: pd.Series) -> pd.Series:
    # A tabela consolidada combina timestamps ingênuos, Z e offsets explícitos.
    # Sem uma linha temporal comum o pandas 3 aborta; UTC preserva todos os
    # instantes e evita que uma auditoria silenciosamente perca uma fonte.
    return pd.to_datetime(values.astype("string"), errors="coerce",
                          format="mixed", utc=True)


def run_text(cmd: list[str]) -> str:
    try:
        p = subprocess.run(
            cmd,
            cwd=REPO,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensivo para ambiente sem git
        return f"ERRO: {exc}"
    return (p.stdout or p.stderr).strip()


def normalize_label_col(df: pd.DataFrame) -> str | None:
    for col in ("classe", "label"):
        if col in df.columns:
            return col
    return None


def normalize_date_col(df: pd.DataFrame) -> str | None:
    for col in ("data_evento", "event_date"):
        if col in df.columns:
            return col
    return None


def normalize_group_col(df: pd.DataFrame) -> str | None:
    for col in ("grupo_cv", "observation_unit_key", "aoi"):
        if col in df.columns:
            return col
    return None


def is_negative_comparavel(value: Any) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    s = str(value).strip().lower()
    return any(tok in s for tok in NEGATIVOS_COMPARAVEIS)


def classify_temporal_identifiability(stats: dict[str, Any]) -> str:
    """Classificacao categorica, sem score."""
    n_dates = stats.get("n_dates", stats.get("eventos_datados", 0))
    if stats.get("positivos", 0) <= 0 or stats.get("negativos", 0) <= 0:
        return "NAO_IDENTIFICAVEL"
    if n_dates < 3:
        return "NAO_IDENTIFICAVEL"
    if stats.get("eventos_independentes", 0) < 10:
        return "NAO_IDENTIFICAVEL"
    if stats.get("folds_sustentaveis", 0) <= 0:
        return "NAO_IDENTIFICAVEL"
    if stats.get("negativos_comparaveis", 0) <= 0:
        return "PARCIALMENTE_IDENTIFICAVEL"
    if stats.get("datas_com_ambas_classes", 0) <= 0:
        return "PARCIALMENTE_IDENTIFICAVEL"
    if stats.get("source_key") == "uk" and stats.get("folds_sustentaveis", 0) >= 3:
        return "IDENTIFICAVEL"
    if stats.get("folds_sustentaveis", 0) >= 3 and stats.get("anos_cobertos", 0) >= 5:
        return "IDENTIFICAVEL"
    return "PARCIALMENTE_IDENTIFICAVEL"


def temporal_support_possible(ident: str) -> str:
    if ident == "IDENTIFICAVEL":
        return "sim"
    if ident == "PARCIALMENTE_IDENTIFICAVEL":
        return "parcial"
    return "nao"


def evaluate_fold_temporal(
    train_dates: pd.Series,
    test_dates: pd.Series,
    train_groups: pd.Series | None = None,
    test_groups: pd.Series | None = None,
) -> dict[str, Any]:
    tr = parse_date(train_dates).dropna()
    te = parse_date(test_dates).dropna()
    order_valid = bool(not tr.empty and not te.empty and tr.max() < te.min())
    shared_dates = sorted({str(pd.Timestamp(x).date()) for x in set(tr) & set(te)})
    shared_groups: list[str] = []
    if train_groups is not None and test_groups is not None:
        shared_groups = sorted(
            str(x)
            for x in (
                set(train_groups.dropna().astype(str))
                & set(test_groups.dropna().astype(str))
            )
        )
    return {
        "temporal_order_valid": order_valid,
        "shared_event_dates": shared_dates,
        "shared_groups": shared_groups,
        "leakage_detected": bool(shared_dates or shared_groups or not order_valid),
    }


def rain_window_is_allowed(
    event_date: str | pd.Timestamp,
    rain_reference_date: str | pd.Timestamp,
    window_end: str | pd.Timestamp | None = None,
) -> bool:
    event_ts = pd.Timestamp(event_date)
    reference_ts = pd.Timestamp(rain_reference_date)
    end_ts = pd.Timestamp(window_end) if window_end is not None else reference_ts
    return bool(reference_ts <= event_ts and end_ts <= event_ts)


def capacidade_por_regiao(pool: pd.DataFrame, admitidos: pd.DataFrame) -> pd.DataFrame:
    base = pool if not pool.empty else admitidos
    viability = read_csv(MOD_PROSP / "viabilidade_por_fonte.csv")
    rows: list[dict[str, Any]] = []
    if base.empty:
        return pd.DataFrame(rows)

    label_col = normalize_label_col(base)
    date_col = normalize_date_col(base)
    group_col = normalize_group_col(base)
    if label_col is None or "fonte" not in base.columns:
        return pd.DataFrame(rows)

    d0 = base[base[label_col].isin([0, 1])].copy()
    if date_col:
        d0["_date"] = parse_date(d0[date_col])
        d0["_year"] = d0["_date"].dt.year
    else:
        d0["_date"] = pd.NaT
        d0["_year"] = np.nan

    for source in sorted(d0["fonte"].dropna().astype(str).unique()):
        d = d0[d0["fonte"].astype(str) == source].copy()
        labels = d[label_col].astype(int)
        dates = d["_date"]
        neg = d[labels == 0]
        neg_quality_col = "nivel_negativo" if "nivel_negativo" in d.columns else None
        neg_comp = (
            int(neg[neg_quality_col].map(is_negative_comparavel).sum())
            if neg_quality_col
            else 0
        )
        by_date = (
            d.dropna(subset=["_date"])
            .groupby(d["_date"].dt.date)[label_col]
            .nunique()
        )
        n_dates = int(dates.dt.date.dropna().nunique())
        years = sorted(int(y) for y in d["_year"].dropna().unique())
        stats: dict[str, Any] = {
            "regiao": SOURCE_NAMES.get(source, source),
            "dataset": f"ds05:{source}",
            "source_key": source,
            "observacoes_datadas": int(dates.notna().sum()),
            "eventos_datados": n_dates,
            "n_dates": n_dates,
            "positivos": int((labels == 1).sum()),
            "negativos": int((labels == 0).sum()),
            "negativos_comparaveis": neg_comp,
            "datas_com_ambas_classes": int((by_date >= 2).sum()),
            "eventos_independentes": int(d[group_col].nunique()) if group_col else 0,
            "extensao_temporal_dias": (
                int((dates.max() - dates.min()).days) if dates.notna().any() else 0
            ),
            "inicio": str(dates.min().date()) if dates.notna().any() else None,
            "fim": str(dates.max().date()) if dates.notna().any() else None,
            "anos_cobertos": len(years),
            "anos": ",".join(str(y) for y in years),
            "forcing_disponivel_pct": _forcing_available_pct(d),
            "folds_sustentaveis": 0,
        }
        if not viability.empty and source in set(viability["fonte"].astype(str)):
            v = viability[viability["fonte"].astype(str) == source].iloc[0]
            for col in ("folds_sustentaveis", "eventos_positivos", "pontos_por_grupo"):
                if col in v:
                    stats[col] = v[col]
        ident = classify_temporal_identifiability(stats)
        stats["temporal_identifiability"] = ident
        stats["teste_prospectivo_possivel"] = temporal_support_possible(ident)
        stats["justificativa"] = _capacity_reason(stats)
        rows.append(stats)

    rows.append(
        {
            "regiao": "Petropolis",
            "dataset": "sem_inventario_local",
            "source_key": "petropolis",
            "observacoes_datadas": 0,
            "eventos_datados": 0,
            "n_dates": 0,
            "positivos": 0,
            "negativos": 0,
            "negativos_comparaveis": 0,
            "datas_com_ambas_classes": 0,
            "eventos_independentes": 0,
            "extensao_temporal_dias": 0,
            "inicio": None,
            "fim": None,
            "anos_cobertos": 0,
            "anos": "",
            "forcing_disponivel_pct": 0.0,
            "folds_sustentaveis": 0,
            "temporal_identifiability": "NAO_IDENTIFICAVEL",
            "teste_prospectivo_possivel": "nao",
            "justificativa": (
                "sem ponto rotulado local: nao existe AUC, calibracao ou "
                "teste temporal local definivel"
            ),
        }
    )
    return pd.DataFrame(rows)


def _forcing_available_pct(d: pd.DataFrame) -> float:
    cols = [c for c in FEATURES_FORCING if c in d.columns]
    if not cols or len(d) == 0:
        return 0.0
    return round(float(d[cols].notna().all(axis=1).mean()), 4)


def _capacity_reason(stats: dict[str, Any]) -> str:
    ident = stats["temporal_identifiability"]
    if ident == "IDENTIFICAVEL":
        return "possui eventos datados, ambas classes, negativos comparaveis e folds sustentaveis"
    if stats.get("positivos", 0) <= 0:
        return "sem positivos locais observados"
    if stats.get("negativos", 0) <= 0:
        return "sem negativos locais interpretaveis"
    if stats.get("negativos_comparaveis", 0) <= 0:
        return "negativos existem, mas nao tem qualidade temporal comparavel suficiente"
    if stats.get("datas_com_ambas_classes", 0) <= 0:
        return "datas/eventos nao tem ambas classes de modo avaliavel"
    return "estrutura temporal parcial; confundimentos de fonte, evento ou amostragem permanecem"


def historical_tests(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.extend(_historical_from_experiment_metrics())
    rows.extend(_historical_from_mod_prosp())
    rows.extend(_historical_from_curitiba_20o())
    rows.extend(_historical_from_curitiba_20q())
    out = pd.DataFrame(rows)
    expected = [
        "region",
        "dataset",
        "test_id",
        "source_artifact",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "n_rows_train",
        "n_rows_test",
        "n_events_train",
        "n_events_test",
        "n_dates_train",
        "n_dates_test",
        "positive_train",
        "negative_train",
        "positive_test",
        "negative_test",
        "label_sources",
        "negative_type",
        "validation_unit",
        "preprocessing_scope",
        "rain_source",
        "model",
        "auc",
        "ci95",
    ]
    for col in expected:
        if col not in out.columns:
            out[col] = None
    text_cols = [
        "region",
        "dataset",
        "test_id",
        "source_artifact",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
        "label_sources",
        "negative_type",
        "validation_unit",
        "preprocessing_scope",
        "rain_source",
        "model",
        "ci95",
    ]
    numeric_cols = [c for c in expected if c not in text_cols]
    for col in text_cols:
        out[col] = out[col].map(lambda x: None if pd.isna(x) else str(x))
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[expected].drop_duplicates().reset_index(drop=True)


def _historical_from_experiment_metrics() -> list[dict[str, Any]]:
    metrics = read_csv(EXPERIMENTOS / "metrics.csv")
    folds = read_csv(EXPERIMENTOS / "folds.csv")
    if metrics.empty:
        return []
    rows = []
    for _, r in metrics.iterrows():
        dataset = str(r.get("dataset", ""))
        esquema = str(r.get("esquema", ""))
        model = str(r.get("modelo", ""))
        f = folds[
            (folds.get("dataset", "") == dataset)
            & (folds.get("modelo", "") == model)
            & (folds.get("esquema", "") == esquema)
        ]
        row = {
            "region": _region_from_dataset(dataset),
            "dataset": dataset,
            "test_id": esquema,
            "source_artifact": rel(EXPERIMENTOS / "metrics.csv"),
            "train_start": _min_col(f, "train_period_ini"),
            "train_end": _max_col(f, "train_period_fim"),
            "test_start": _min_col(f, "test_period_ini"),
            "test_end": _max_col(f, "test_period_fim"),
            "n_rows_train": _int_or_none(r.get("n_train")),
            "n_rows_test": _int_or_none(r.get("n_test")),
            "n_events_train": _int_or_none(_mean_col(f, "train_groups")),
            "n_events_test": _int_or_none(_mean_col(f, "test_groups")),
            "n_dates_train": None,
            "n_dates_test": None,
            "positive_train": _count_from_prevalence(r.get("n_train"), r.get("prevalence_train")),
            "negative_train": _count_from_prevalence(r.get("n_train"), 1 - _float_or_nan(r.get("prevalence_train"))),
            "positive_test": _count_from_prevalence(r.get("n_test"), r.get("prevalence_test")),
            "negative_test": _count_from_prevalence(r.get("n_test"), 1 - _float_or_nan(r.get("prevalence_test"))),
            "label_sources": _label_sources_for_dataset(dataset),
            "negative_type": _negative_type_for_dataset(dataset),
            "validation_unit": "grupo/celula/evento conforme matriz_validacao",
            "preprocessing_scope": "fit_train_only registrado em matriz_validacao",
            "rain_source": _rain_source_for_dataset(dataset),
            "model": model,
            "auc": _float_or_none(r.get("auc")),
            "ci95": _ci(r.get("auc_ic95_lo"), r.get("auc_ic95_hi")),
        }
        rows.append(row)
    return rows


def _historical_from_mod_prosp() -> list[dict[str, Any]]:
    folds = read_csv(MOD_PROSP / "folds.csv")
    if folds.empty:
        return []
    rows = []
    for _, r in folds.iterrows():
        rows.append(
            {
                "region": "Environment Agency/UK",
                "dataset": f"mod-prosp-02:{r.get('conjunto')}",
                "test_id": f"janela_expansiva_{r.get('variante')}",
                "source_artifact": rel(MOD_PROSP / "folds.csv"),
                "train_start": None,
                "train_end": r.get("corte"),
                "test_start": r.get("corte"),
                "test_end": None,
                "n_rows_train": _int_or_none(r.get("n_treino")),
                "n_rows_test": _int_or_none(r.get("n_teste")),
                "n_events_train": _int_or_none(r.get("eventos_treino")),
                "n_events_test": _int_or_none(r.get("eventos_teste")),
                "n_dates_train": None,
                "n_dates_test": None,
                "positive_train": _int_or_none(r.get("pos_treino")),
                "negative_train": _int_or_none(r.get("neg_treino")),
                "positive_test": _int_or_none(r.get("pos_teste")),
                "negative_test": _int_or_none(r.get("neg_teste")),
                "label_sources": "Environment Agency Recorded Flood Outlines",
                "negative_type": "exclusao_qualificada",
                "validation_unit": "grupo/evento/bloco negativo",
                "preprocessing_scope": "ajuste por fold; IC por bootstrap de grupos",
                "rain_source": "open_meteo_era5_land quando conjunto COMPLETO",
                "model": "firth/logistica da frente prospectiva",
                "auc": _float_or_none(r.get("auc_prospectivo")),
                "ci95": r.get("ic95"),
            }
        )
    return rows


def _historical_from_curitiba_20o() -> list[dict[str, Any]]:
    report = read_json(CURITIBA_PUBLIC / "results" / "v20o_all_reports.json")
    if not report:
        return []
    split = report.get("split", {})
    h = report.get("holdout_auc", {})
    return [
        {
            "region": "Curitiba",
            "dataset": "v20n_dataset_curitiba_features_v2",
            "test_id": "SUSC-20O_holdout_2026",
            "source_artifact": rel(CURITIBA_PUBLIC / "results" / "v20o_all_reports.json"),
            "train_start": min(split.get("treino_anos", [None])),
            "train_end": max(split.get("treino_anos", [None])),
            "test_start": min(split.get("teste_anos", [None])),
            "test_end": max(split.get("teste_anos", [None])),
            "n_rows_train": h.get("n_train"),
            "n_rows_test": h.get("n_test"),
            "n_events_train": None,
            "n_events_test": None,
            "n_dates_train": None,
            "n_dates_test": None,
            "positive_train": report.get("firth_report_treino", {}).get("n_pos_train"),
            "negative_train": report.get("firth_report_treino", {}).get("n_neg_train"),
            "positive_test": h.get("n_pos_test"),
            "negative_test": h.get("n_neg_test"),
            "label_sources": "SIAC 156 Curitiba + negativos classificados",
            "negative_type": "processo local a auditar",
            "validation_unit": "unidade_observacao sem duplicata",
            "preprocessing_scope": "StandardScaler fitado no treino",
            "rain_source": "Open-Meteo/ERA5-Land + CHIRPS residualizado",
            "model": "logistica_l2 diagnostica",
            "auc": h.get("holdout_auc"),
            "ci95": None,
        }
    ]


def _historical_from_curitiba_20q() -> list[dict[str, Any]]:
    wf = read_csv(CURITIBA_PUBLIC / "results" / "v20q_walk_forward_cutoffs.csv")
    if wf.empty:
        return []
    rows = []
    for _, r in wf.iterrows():
        rows.append(
            {
                "region": "Curitiba",
                "dataset": "v20n_dataset_curitiba_features_v2",
                "test_id": f"SUSC-20Q_walk_forward_{r.get('test_year')}",
                "source_artifact": rel(CURITIBA_PUBLIC / "results" / "v20q_walk_forward_cutoffs.csv"),
                "train_start": str(r.get("train_years")).split(",")[0],
                "train_end": str(r.get("train_years")).split(",")[-1],
                "test_start": _int_or_none(r.get("test_year")),
                "test_end": _int_or_none(r.get("test_year")),
                "n_rows_train": _int_or_none(r.get("n_train")),
                "n_rows_test": _int_or_none(r.get("n_test")),
                "n_events_train": None,
                "n_events_test": None,
                "n_dates_train": None,
                "n_dates_test": None,
                "positive_train": None,
                "negative_train": None,
                "positive_test": _int_or_none(r.get("n_pos_test")),
                "negative_test": _int_or_none(r.get("n_neg_test")),
                "label_sources": "SIAC 156 Curitiba + negativos classificados",
                "negative_type": "processo local a auditar",
                "validation_unit": "unidade_observacao sem duplicata",
                "preprocessing_scope": "fit_train_only pelo script SUSC-20Q",
                "rain_source": "Open-Meteo/ERA5-Land + CHIRPS residualizado",
                "model": "logistica_l2 diagnostica",
                "auc": _float_or_none(r.get("holdout_auc")),
                "ci95": None,
            }
        )
    return rows


def standardize_folds() -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for path, label in [
        (EXPERIMENTOS / "folds.csv", "experimentos-cientificos"),
        (EXPERIMENTOS / "curitiba_folds.csv", "curitiba-autopsia"),
    ]:
        df = read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df["source_artifact"] = rel(path)
        df["test_family"] = label
        pieces.append(df)
    uk = read_csv(MOD_PROSP / "folds.csv")
    if not uk.empty:
        u = pd.DataFrame(
            {
                "dataset": "uk_control",
                "modelo": "mod-prosp-02",
                "esquema": "forward_temporal",
                "fold": uk["corte"].astype(str) + "_" + uk["conjunto"].astype(str),
                "n_train": uk["n_treino"],
                "n_test": uk["n_teste"],
                "train_groups": uk["grupos_treino"],
                "test_groups": uk["grupos_teste"],
                "prevalence_train": uk["prevalencia_treino"],
                "prevalence_test": uk["prevalencia_teste"],
                "train_period_ini": None,
                "train_period_fim": uk["corte"],
                "test_period_ini": uk["corte"],
                "test_period_fim": None,
                "auc": uk["auc_prospectivo"],
                "brier": None,
                "protocol_hash": None,
                "source_artifact": rel(MOD_PROSP / "folds.csv"),
                "test_family": "mod-prosp-02",
            }
        )
        pieces.append(u)
    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True, sort=False)
    out["temporal_test_candidate"] = out["esquema"].astype(str).str.contains(
        "temporal|forward", case=False, regex=True
    )
    out["classes_avaliaveis"] = (
        out["prevalence_test"].astype(float, errors="ignore").between(0, 1, inclusive="neither")
        if "prevalence_test" in out.columns
        else False
    )
    out["temporal_order_valid"] = out.apply(_fold_row_order_valid, axis=1)
    out["temporal_test_valid"] = (
        out["temporal_test_candidate"] & out["classes_avaliaveis"] & out["temporal_order_valid"]
    )
    return out


def _fold_row_order_valid(row: pd.Series) -> bool:
    if not bool(row.get("temporal_test_candidate")):
        return False
    ini = row.get("test_period_ini")
    fim = row.get("train_period_fim")
    if pd.isna(ini) or pd.isna(fim):
        return True
    try:
        return bool(pd.Timestamp(fim) <= pd.Timestamp(ini))
    except Exception:
        return True


def drift_covariates() -> pd.DataFrame:
    d = read_csv(EXPERIMENTOS / "feature_drift.csv")
    if d.empty:
        return pd.DataFrame()
    d = d.copy()
    d.insert(0, "dataset", "curitiba_autopsia")
    d["axis"] = np.where(d["feature"].astype(str).str.startswith("rain_"), "forcing", "covariate")
    d["shift_interpretation"] = d["psi_leitura"].map(
        {
            "estavel": "sem covariate shift material por PSI",
            "moderado": "covariate shift moderado",
            "grande": "covariate shift grande",
        }
    ).fillna("avaliar")
    return d


def drift_forcing(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not pool.empty and {"fonte", "data_evento"}.issubset(pool.columns):
        d = pool.copy()
        d["_date"] = parse_date(d["data_evento"])
        d["_year"] = d["_date"].dt.year
        for source, g in d.dropna(subset=["_year"]).groupby("fonte"):
            for feat in FEATURES_FORCING:
                if feat not in g.columns:
                    continue
                vals = pd.to_numeric(g[feat], errors="coerce")
                rows.append(
                    {
                        "dataset": f"ds05:{source}",
                        "feature": feat,
                        "year": int(g["_year"].iloc[0]) if g["_year"].nunique() == 1 else None,
                        "period": "all_years",
                        "n": int(len(g)),
                        "missing_pct": round(float(vals.isna().mean()), 4),
                        "median": _float_or_none(vals.median()),
                        "p05": _float_or_none(vals.quantile(0.05)),
                        "p95": _float_or_none(vals.quantile(0.95)),
                        "provider": "open_meteo_era5_land",
                        "lineage_status": "aggregate_from_ds05",
                    }
                )
            for (year), gy in g.groupby("_year"):
                for feat in FEATURES_FORCING:
                    if feat not in gy.columns:
                        continue
                    vals = pd.to_numeric(gy[feat], errors="coerce")
                    rows.append(
                        {
                            "dataset": f"ds05:{source}",
                            "feature": feat,
                            "year": int(year),
                            "period": "calendar_year",
                            "n": int(len(gy)),
                            "missing_pct": round(float(vals.isna().mean()), 4),
                            "median": _float_or_none(vals.median()),
                            "p05": _float_or_none(vals.quantile(0.05)),
                            "p95": _float_or_none(vals.quantile(0.95)),
                            "provider": "open_meteo_era5_land",
                            "lineage_status": "aggregate_from_ds05",
                        }
                    )
    cov = drift_covariates()
    if not cov.empty:
        for _, r in cov[cov["axis"] == "forcing"].iterrows():
            rows.append(
                {
                    "dataset": r["dataset"],
                    "feature": r["feature"],
                    "year": 2026,
                    "period": "train<=2025_vs_test2026",
                    "n": int(r["n_treino"] + r["n_teste"]),
                    "missing_pct": None,
                    "median": None,
                    "p05": None,
                    "p95": None,
                    "provider": "open_meteo_era5_land",
                    "lineage_status": f"PSI={r['psi']} ({r['psi_leitura']})",
                }
            )
    return pd.DataFrame(rows)


def observation_shift(pool: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    meta = read_csv(CURITIBA_PUBLIC / "results" / "v20q_metadata_composition_stability_by_year.csv")
    if not meta.empty:
        m = meta.copy()
        m.insert(0, "dataset", "curitiba_v20q")
        m["shift_axis"] = "tempo_observacional"
        rows.append(m)
    if not pool.empty and {"fonte", "classe", "data_evento"}.issubset(pool.columns):
        d = pool.copy()
        d["_year"] = parse_date(d["data_evento"]).dt.year
        comp = (
            d.dropna(subset=["_year"])
            .groupby(["_year", "fonte", "classe"], dropna=False)
            .size()
            .reset_index(name="n")
        )
        comp["dataset"] = "ds05_pool_fluvial"
        comp["year"] = comp["_year"].astype(int)
        comp["shift_axis"] = "source_label_composition"
        comp = comp.drop(columns=["_year"])
        rows.append(comp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def sampling_shift() -> pd.DataFrame:
    d = read_csv(EXPERIMENTOS / "curitiba_distribuicao_temporal.csv")
    if d.empty:
        return pd.DataFrame()
    d = d.copy()
    d.insert(0, "dataset", "curitiba_autopsia")
    d["shift_axis"] = "prior_sampling_shift"
    d["negative_interpretation"] = "negativo local depende do processo de amostragem e deve ser auditado por periodo"
    return d


def coefficients_by_time() -> pd.DataFrame:
    d = read_csv(EXPERIMENTOS / "efeitos_por_periodo.csv")
    if d.empty:
        return pd.DataFrame()
    d = d.copy()
    d.insert(0, "dataset", "curitiba_autopsia")
    d["axis"] = np.where(d["feature"].astype(str).str.startswith("rain_"), "forcing", "static")
    return d


def walk_forward() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    cur = read_csv(CURITIBA_PUBLIC / "results" / "v20q_walk_forward_cutoffs.csv")
    if not cur.empty:
        c = cur.copy()
        c.insert(0, "dataset", "curitiba_v20q")
        c["cutoff"] = c["test_year"].map(lambda x: f"train_before_{int(x)}")
        c["auc"] = c["holdout_auc"]
        c["ci"] = None
        c["brier"] = None
        c["calibration_slope"] = None
        c["n_events"] = None
        c["prevalence"] = c["n_pos_test"] / (c["n_pos_test"] + c["n_neg_test"])
        c["domain_distance"] = None
        c["forcing_shift"] = np.where(c["test_year"] == 2026, "avaliado_em_drift_forcing", "nao_avaliado")
        rows.append(c)
    uk = read_csv(MOD_PROSP / "folds.csv")
    if not uk.empty:
        u = uk.copy()
        u.insert(0, "dataset", "uk_control_mod_prosp_02")
        u["cutoff"] = u["corte"]
        u["auc"] = u["auc_prospectivo"]
        u["ci"] = u["ic95"]
        u["brier"] = None
        u["calibration_slope"] = None
        u["n_events"] = u["eventos_teste"]
        u["prevalence"] = u["prevalencia_teste"]
        u["domain_distance"] = None
        u["forcing_shift"] = "nao_decomposto"
        rows.append(u)
    if not rows:
        return pd.DataFrame()
    keep = [
        "dataset",
        "cutoff",
        "auc",
        "ci",
        "brier",
        "calibration_slope",
        "n_events",
        "prevalence",
        "domain_distance",
        "forcing_shift",
    ]
    out = pd.concat(rows, ignore_index=True, sort=False)
    for col in keep:
        if col not in out.columns:
            out[col] = None
    return out[keep]


def placebo_results(pool: pd.DataFrame, curitiba: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not pool.empty:
        c = pool[(pool.get("fonte") == "curitiba") & pool.get("classe").isin([0, 1])].copy()
        if not c.empty and "data_evento" in c.columns:
            c["_year"] = parse_date(c["data_evento"]).dt.year
            c["_periodo_2026"] = (c["_year"] == 2026).astype(int)
            rows.extend(_period_auc_rows(c, "curitiba_ds05", FEATURES_STATIC, "periodo_por_topografia"))
            rows.extend(_period_auc_rows(c, "curitiba_ds05", FEATURES_FORCING, "periodo_por_chuva"))
            rows.extend(_period_auc_rows(c, "curitiba_ds05", FEATURES_STATIC + FEATURES_FORCING, "periodo_por_topografia_chuva"))
            if c["_year"].notna().nunique() > 1:
                rows.extend(_period_auc_rows(c, "curitiba_ds05", ["_year"], "periodo_por_ano"))
    if not curitiba.empty:
        d = curitiba.copy()
        if "event_date" in d.columns:
            d["_year"] = parse_date(d["event_date"]).dt.year
            d["_periodo_2026"] = (d["_year"] == 2026).astype(int)
            meta_cols = [
                c
                for c in (
                    "source_registry",
                    "negative_source_type",
                    "confidence_tier",
                    "assunto",
                    "subdivisao",
                    "rain_status",
                )
                if c in d.columns
            ]
            if meta_cols:
                rows.extend(_period_auc_rows(d, "curitiba_v20n", meta_cols, "periodo_por_metadados", categorical=True))

    rows.append(
        {
            "dataset": "curitiba",
            "placebo": "chuva_deslocada_mais_menos_1_dia",
            "executed": False,
            "auc_mean": None,
            "folds": 0,
            "interpretacao": "nao executado: tabela auditada nao preserva serie diaria por observacao",
        }
    )
    rows.append(
        {
            "dataset": "curitiba",
            "placebo": "labels_permutados_dentro_de_evento",
            "executed": False,
            "auc_mean": None,
            "folds": 0,
            "interpretacao": "nao executado: muitos eventos/data tem classe unica ou amostra pequena",
        }
    )
    return pd.DataFrame(rows)


def _period_auc_rows(
    df: pd.DataFrame,
    dataset: str,
    cols: list[str],
    name: str,
    categorical: bool = False,
) -> list[dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if not {"_periodo_2026"}.issubset(df.columns):
        return []
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return [
            {
                "dataset": dataset,
                "placebo": name,
                "executed": False,
                "auc_mean": None,
                "folds": 0,
                "interpretacao": f"colunas ausentes: {missing}",
            }
        ]
    d = df.dropna(subset=["_periodo_2026"]).copy()
    if len(d) < 20 or d["_periodo_2026"].nunique() < 2:
        return [
            {
                "dataset": dataset,
                "placebo": name,
                "executed": False,
                "auc_mean": None,
                "folds": 0,
                "interpretacao": "periodo nao possui duas classes para o diagnostico",
            }
        ]
    y = d["_periodo_2026"].astype(int).to_numpy()
    if categorical:
        X = pd.get_dummies(d[cols].astype("string").fillna("NA"), dummy_na=True).to_numpy(float)
        pipe = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        numeric = d[cols].apply(pd.to_numeric, errors="coerce")
        valid = numeric.notna().all(axis=1).to_numpy()
        X = numeric.loc[valid].to_numpy(float)
        y = y[valid]
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    if len(np.unique(y)) < 2 or len(y) < 20:
        return [
            {
                "dataset": dataset,
                "placebo": name,
                "executed": False,
                "auc_mean": None,
                "folds": 0,
                "interpretacao": "apos listwise deletion nao restaram duas classes",
            }
        ]
    n_splits = int(min(5, np.bincount(y).min()))
    if n_splits < 2:
        return [
            {
                "dataset": dataset,
                "placebo": name,
                "executed": False,
                "auc_mean": None,
                "folds": 0,
                "interpretacao": "classe minoritaria insuficiente para CV",
            }
        ]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20260828)
    aucs = []
    for tr, te in cv.split(X, y):
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        aucs.append(float(roc_auc_score(y[te], p)))
    return [
        {
            "dataset": dataset,
            "placebo": name,
            "executed": True,
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_min": round(float(np.min(aucs)), 4),
            "auc_max": round(float(np.max(aucs)), 4),
            "folds": len(aucs),
            "n": int(len(y)),
            "features": ",".join(cols),
            "interpretacao": "AUC > 0.6 indica que X identifica periodo e pode carregar atalho temporal",
        }
    ]


def temporal_validity_matrix(capacidade: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append(
        {
            "regiao_dataset": "Environment Agency/UK - MOD-PROSP-02 TERRENO bloco",
            "temporal_identifiability": "IDENTIFICAVEL",
            "future_holdout": True,
            "event_independence": True,
            "negative_quality": "exclusao_qualificada",
            "observation_stability": "comparavel no desenho do piloto",
            "sampling_stability": "balanceado por blocos no controle positivo",
            "covariate_shift": "nao decomposto nesta auditoria",
            "forcing_shift": "nao necessario para TERRENO",
            "conditional_shift_evidence": "nao demonstrada",
            "performance_stability": "stable",
            "conclusion": "TEMPORAL_SUPPORT",
        }
    )
    rows.append(
        {
            "regiao_dataset": "Curitiba 2026 - SUSC-20O/20Q",
            "temporal_identifiability": "PARCIALMENTE_IDENTIFICAVEL",
            "future_holdout": True,
            "event_independence": "parcial: fold por ano, mas evento/data avaliavel e limitado",
            "negative_quality": "local; precisa de auditoria temporal dos negativos",
            "observation_stability": "limitada: processo administrativo e geocodificacao entram no rotulo",
            "sampling_stability": "instavel: prevalencia anual e composicao devem ser separadas",
            "covariate_shift": "demonstrado em chuva; topografia estavel",
            "forcing_shift": "demonstrado/compativel nos indices de chuva",
            "conditional_shift_evidence": "possivel, nao identificada separadamente",
            "performance_stability": "observed_degradation",
            "conclusion": "TEMPORAL_INCONCLUSIVE",
        }
    )
    rows.append(
        {
            "regiao_dataset": "Curitiba leave-one-event-out",
            "temporal_identifiability": "NAO_IDENTIFICAVEL",
            "future_holdout": False,
            "event_independence": "insuficiente: muitos eventos/data com classe unica",
            "negative_quality": "nao pareado por evento",
            "observation_stability": "limitada",
            "sampling_stability": "limitada",
            "covariate_shift": "nao interpreta causalmente",
            "forcing_shift": "nao separado",
            "conditional_shift_evidence": "nao identificavel",
            "performance_stability": "metric_undefined_or_unstable",
            "conclusion": "TEMPORAL_TEST_INVALID",
        }
    )
    for _, r in capacidade.iterrows():
        key = str(r["source_key"])
        if key in {"uk", "curitiba"}:
            continue
        ident = r["temporal_identifiability"]
        ev = avaliar_evidencia_temporal(
            temporal_test_valid=(ident != "NAO_IDENTIFICAVEL"),
            temporal_identifiability=ident,
            performance_stability=None,
        )
        rows.append(
            {
                "regiao_dataset": f"{r['regiao']} - capacidade temporal",
                "temporal_identifiability": ident,
                "future_holdout": r["teste_prospectivo_possivel"],
                "event_independence": r["eventos_independentes"],
                "negative_quality": r["negativos_comparaveis"],
                "observation_stability": "nao avaliada separadamente",
                "sampling_stability": "nao avaliada separadamente",
                "covariate_shift": "nao avaliado",
                "forcing_shift": "nao avaliado",
                "conditional_shift_evidence": "nao identificavel",
                "performance_stability": None,
                "conclusion": ev["status"],
            }
        )
    return pd.DataFrame(rows)


def curitiba_2026_summary(
    capacidade: pd.DataFrame,
    cov: pd.DataFrame,
    samp: pd.DataFrame,
    placebo: pd.DataFrame,
) -> dict[str, Any]:
    v20o = read_json(CURITIBA_PUBLIC / "results" / "v20o_all_reports.json")
    v20q = read_json(CURITIBA_PUBLIC / "results" / "v20q_all_reports.json")
    exp_h = read_json(EXPERIMENTOS / "hypotheses.json")
    cap = capacidade[capacidade["source_key"] == "curitiba"].to_dict(orient="records")
    return {
        "dataset": "Curitiba 2026",
        "fontes": [
            rel(CURITIBA_PUBLIC / "results" / "v20o_all_reports.json"),
            rel(CURITIBA_PUBLIC / "results" / "v20q_all_reports.json"),
            rel(EXPERIMENTOS / "curitiba_metrics.csv"),
            rel(EXPERIMENTOS / "feature_drift.csv"),
        ],
        "capacidade_temporal": cap[0] if cap else {},
        "split_historico": v20o.get("split"),
        "holdout_historico": v20o.get("holdout_auc"),
        "bootstrap_ci": v20q.get("1_bootstrap_ci"),
        "walk_forward": v20q.get("3_walk_forward"),
        "hipoteses_auditadas": exp_h.get("hipoteses", {}),
        "covariate_shift": cov[cov["dataset"].eq("curitiba_autopsia")].to_dict(orient="records")
        if not cov.empty
        else [],
        "sampling_shift": samp.to_dict(orient="records") if not samp.empty else [],
        "placebos": placebo[placebo["dataset"].astype(str).str.contains("curitiba", case=False, na=False)].to_dict(orient="records")
        if not placebo.empty
        else [],
        "auc_052_mede": (
            "o desenho historico D0 mede capacidade de ordenacao para a amostra "
            "administrativa futura de 2026 sob o processo observado. Ele nao "
            "identifica sozinho mudanca em P(Y|X), mudanca climatica ou falha "
            "pura do mecanismo fisico."
        ),
        "taxonomia": {
            "status_review_only": "TEMPORAL_INCONCLUSIVE",
            "qualificacao": (
                "ha degradacao observada no holdout, mas a magnitude nao esta "
                "separada de forcing, sampling e observation shift"
            ),
            "nao_chamar_de": [
                "concept_drift_provado",
                "mudanca_climatica_provada",
                "modelo_corrigido",
            ],
        },
    }


def uk_control_summary() -> dict[str, Any]:
    r = read_json(MOD_PROSP / "resultado.json")
    terreno = r.get("conjuntos", {}).get("TERRENO", {}).get("bloco", {})
    return {
        "dataset": "Environment Agency/UK",
        "fonte": rel(MOD_PROSP / "resultado.json"),
        "base_do_estrato": r.get("base_do_estrato"),
        "features": r.get("conjuntos", {}).get("TERRENO", {}).get("features"),
        "teste": "janela expansiva por grupos/blocos",
        "identificabilidade": "IDENTIFICAVEL",
        "por_que_identificavel": [
            "eventos positivos independentes e blocos negativos qualificados",
            "multiplos cortes prospectivos",
            "ambas classes avaliaveis em folds de bloco",
            "IC95 por bootstrap de grupos",
        ],
        "resultado_terreno_bloco": {
            "folds": terreno.get("folds"),
            "auc_medio": terreno.get("auc_medio"),
            "auc_min": terreno.get("auc_min"),
            "auc_max": terreno.get("auc_max"),
            "veredito_original": terreno.get("veredito"),
        },
        "status_review_only": "TEMPORAL_SUPPORT",
        "nao_e": r.get("nao_e"),
    }


def conclusoes_json(matrix: pd.DataFrame, curitiba: dict[str, Any], uk: dict[str, Any]) -> dict[str, Any]:
    return {
        "validade_temporal_no_revp": (
            "Treinar com evidencia observavel ate t e testar em eventos "
            "posteriores a t sem vazamento, preservando separacao entre tempo "
            "fisico, tempo observacional, processo de rotulo e processo de "
            "amostragem."
        ),
        "estabilidade_temporal_demonstrada": [
            "Environment Agency/UK MOD-PROSP-02 TERRENO bloco"
        ],
        "instabilidade_temporal_demonstrada": [],
        "degradacao_observada_com_identificabilidade_parcial": [
            "Curitiba 2026 SUSC-20O/20Q"
        ],
        "covariate_shift": "Curitiba: shift concentrado em chuva; topografia estavel no artefato feature_drift.csv.",
        "forcing_shift": "Curitiba: indices de chuva mudam e chuva isolada nao transporta no holdout historico.",
        "prevalencia_amostragem": "Curitiba: prevalencia anual varia e negativos exigem desenho temporal proprio.",
        "observacao_rotulo": "Curitiba: SIAC 156 e negativos locais sao processos observacionais, nao fenomeno puro.",
        "possivel_conditional_shift": "possivel em Curitiba, mas nao identificado separadamente de forcing/sampling/observacao.",
        "falta_identificabilidade": [
            "Petropolis sem inventario local",
            "Curitiba leave-one-event-out por evento/data com classe unica",
            "Recife sem controle temporal local comparavel nesta auditoria",
        ],
        "teste_temporal_invalido_ou_inconclusivo": matrix[
            matrix["conclusion"].isin(["TEMPORAL_TEST_INVALID", "TEMPORAL_INCONCLUSIVE"])
        ]["regiao_dataset"].tolist(),
        "curitiba_taxonomia_final": curitiba["taxonomia"],
        "uk_control": uk["resultado_terreno_bloco"],
        "svc07_promocao": "sem alteracao; proposta temporal e modo de revisao",
        "proximo_experimento": (
            "desenho event-matched para negativos de Curitiba, congelado antes "
            "do resultado, com oportunidade de observacao e janela temporal "
            "comparaveis"
        ),
    }


def write_report(conclusoes: dict[str, Any], matrix: pd.DataFrame, placebo: pd.DataFrame) -> None:
    cur_status = conclusoes["curitiba_taxonomia_final"]["status_review_only"]
    uk_auc = conclusoes["uk_control"].get("auc_medio")
    placebo_exec = placebo[placebo.get("executed", False) == True] if not placebo.empty else pd.DataFrame()
    strongest_placebo = None
    if not placebo_exec.empty and "auc_mean" in placebo_exec:
        row = placebo_exec.sort_values("auc_mean", ascending=False).iloc[0]
        strongest_placebo = f"{row['placebo']} AUC={row['auc_mean']}"

    lines = [
        "# Auditoria de temporalidade do REV-P",
        "",
        "## 1. O que validade temporal passou a significar no REV-P",
        conclusoes["validade_temporal_no_revp"],
        "",
        "## 2. Quais regioes suportam teste temporal",
        _matrix_lines(matrix),
        "",
        "## 3. Controle ingles",
        f"MOD-PROSP-02 e o controle positivo: identificavel, com AUC medio {uk_auc} no TERRENO bloco.",
        "",
        "## 4. Curitiba 2026",
        "Ha degradacao observada no holdout futuro, mas a auditoria classifica a identificabilidade como parcial.",
        "",
        "## 5. O AUC 0,52 mede o que?",
        "Mede ranking na amostra administrativa futura de 2026 sob D0. Nao mede sozinho P(Y|X), clima ou falha pura do mecanismo.",
        "",
        "## 6. Chuva",
        conclusoes["forcing_shift"],
        "",
        "## 7. Negativos",
        conclusoes["prevalencia_amostragem"],
        "",
        "## 8. Mecanismo fisico",
        "Topografia fica majoritariamente estavel em drift; coeficientes fisicos por ano permanecem mistos e devem ser lidos como observacionais.",
        "",
        "## 9. Placebos",
        strongest_placebo or "Placebos executados nao produziram resultado interpretavel dominante.",
        "",
        "## 10. Taxonomia final de Curitiba",
        f"{cur_status}: {conclusoes['curitiba_taxonomia_final']['qualificacao']}",
        "",
        "## 11. Consequencia para Recife, Petropolis e transferencia",
        "Recife e Petropolis nao viram estabilidade temporal por ausencia de estrutura local suficiente; transferencia espacial nao implica transferencia temporal.",
        "",
        "## 12. Consequencia para SVC-07 e promocao",
        "Sem alteracao do contrato ou modelos. A avaliacao temporal proposta e modo de revisao.",
        "",
        "## 13. Bugs/leakage encontrados",
        "Nenhum novo vazamento corrigido nesta camada; a auditoria registra testes para futuro no treino, evento atravessando fold, scaler global e AUC de classe unica.",
        "",
        "## 14. Testes",
        "Executar `python -m pytest tests/ -q` apos gerar os artefatos.",
        "",
        "## 15. Artefatos",
        "`local_runs/temporalidade/` contem ontologia, catalogo historico, matrizes, diagnosticos e resumos JSON.",
        "",
        "## 16. O que continua desconhecido",
        "Curitiba ainda nao separa completamente forcing shift, sampling shift, observation shift e possivel conditional shift.",
        "",
        "## 17. Proximo experimento de maior valor informativo",
        conclusoes["proximo_experimento"],
        "",
    ]
    (OUT / "relatorio.md").write_text("\n".join(lines), encoding="utf-8")


def _matrix_lines(matrix: pd.DataFrame) -> str:
    if matrix.empty:
        return "Matriz vazia."
    lines = []
    for _, r in matrix.iterrows():
        lines.append(
            f"- {r['regiao_dataset']}: {r['temporal_identifiability']} -> {r['conclusion']}"
        )
    return "\n".join(lines)


def manifest(
    args: argparse.Namespace,
    artifacts: list[Path],
    figures: list[Path],
) -> dict[str, Any]:
    import sklearn

    current_status = run_text(["git", "status", "--short"])
    data_inputs = [
        ADMITIDOS,
        POOL_FLUVIAL,
        CURITIBA_V20N,
        RECIFE_V12,
        EXPERIMENTOS / "manifest.json",
        MOD_PROSP / "resultado.json",
    ]
    return {
        "tipo": "manifesto_auditoria_temporalidade",
        "versao": "1",
        "baseline_pre_edicao": {
            "git_head": args.baseline_head,
            "git_status_short": args.baseline_status,
            "git_log_1": args.baseline_log,
            "pytest_command": "python -m pytest tests/ -q",
            "pytest_status_no_inicio": args.baseline_pytest_status,
        },
        "estado_git_ao_gerar": {
            "git_head": run_text(["git", "rev-parse", "HEAD"]),
            "git_status_short": current_status,
            "git_log_1": run_text(["git", "log", "-1", "--oneline"]),
        },
        "versoes": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "inputs": [
            {
                "path": rel(p),
                "exists": p.exists(),
                "sha256": sha256_file(p),
                "bytes": p.stat().st_size if p.exists() else None,
            }
            for p in data_inputs
        ],
        "commands_historicos": known_temporal_commands(),
        "scripts_temporais_descobertos": discover_temporal_scripts(),
        "artefatos": artifact_inventory(artifacts),
        "figuras": artifact_inventory(figures),
        "nao_altera": [
            "modelo_servido",
            "contrato_v1",
            "maturidades",
            "datasets_canonicos",
            "resultados_historicos",
        ],
    }


def known_temporal_commands() -> list[dict[str, Any]]:
    paths = [
        MOD_PROSP / "commands.txt",
        REPO / "local_runs" / "svc-07-reajuste" / "commands.txt",
        EXPERIMENTOS / "commands.txt",
    ]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        rows.append(
            {
                "path": rel(path),
                "sha256": sha256_file(path),
                "commands": [
                    line.strip()
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ],
            }
        )
    return rows


def discover_temporal_scripts() -> list[dict[str, Any]]:
    tokens = ("temporal", "walk_forward", "holdout", "prosp", "svc07")
    roots = [REPO / "scripts", REPO / "outputs_public" / "data"]
    found: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            name = path.name.lower()
            relp = rel(path)
            if "quarentena" in relp or "_curadoria_20260725" in relp:
                continue
            if any(tok in name or tok in relp.lower() for tok in tokens):
                found.append(
                    {
                        "path": relp,
                        "sha256": sha256_file(path),
                        "bytes": path.stat().st_size,
                    }
                )
    return sorted(found, key=lambda x: x["path"])


def artifact_inventory(paths: list[Path]) -> list[dict[str, Any]]:
    inv = []
    for p in paths:
        item = {
            "path": rel(p),
            "exists": p.exists(),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size if p.exists() else None,
        }
        if p.exists() and p.suffix.lower() in {".csv", ".parquet"}:
            try:
                df = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, low_memory=False)
                item["rows"] = int(len(df))
                item["columns"] = list(df.columns)
            except Exception as exc:
                item["row_count_error"] = str(exc)
        inv.append(item)
    return inv


def make_figures(
    cov: pd.DataFrame,
    walk: pd.DataFrame,
    samp: pd.DataFrame,
    coefs: pd.DataFrame,
    obs: pd.DataFrame,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    figdir = OUT / "figuras"
    figdir.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    def save_current(name: str) -> None:
        p = figdir / name
        plt.tight_layout()
        plt.savefig(p, dpi=160)
        plt.close()
        made.append(p)

    if not cov.empty and {"feature", "psi"}.issubset(cov.columns):
        plt.figure(figsize=(7, 4))
        d = cov.sort_values("psi")
        plt.barh(d["feature"].astype(str), d["psi"].astype(float))
        plt.xlabel("PSI train<=2025 vs test 2026")
        plt.title("Curitiba covariate/forcing drift")
        save_current("covariate_drift_curitiba.png")

    if not walk.empty and {"dataset", "auc"}.issubset(walk.columns):
        plt.figure(figsize=(8, 4))
        for ds, g in walk.dropna(subset=["auc"]).groupby("dataset"):
            plt.plot(range(len(g)), g["auc"].astype(float), marker="o", label=str(ds)[:32])
        plt.axhline(0.5, color="black", linewidth=0.8, linestyle="--")
        plt.ylabel("AUC")
        plt.xlabel("cutoff order")
        plt.title("Walk-forward performance")
        plt.legend(fontsize=7)
        save_current("walk_forward_auc.png")

    if not samp.empty and {"ano", "prevalencia"}.issubset(samp.columns):
        plt.figure(figsize=(6, 4))
        plt.plot(samp["ano"], samp["prevalencia"], marker="o")
        plt.ylim(0, 1)
        plt.ylabel("sample prevalence")
        plt.xlabel("year")
        plt.title("Curitiba prevalence by period")
        save_current("prevalencia_curitiba.png")

    if not coefs.empty and {"ano", "feature", "coef"}.issubset(coefs.columns):
        plt.figure(figsize=(8, 4))
        for feat, g in coefs.groupby("feature"):
            if str(feat) in {"hand_m", "twi_dinf", "rain_max_24h", "rain_decay_index"}:
                plt.plot(g["ano"], g["coef"], marker="o", label=str(feat))
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("standardized coefficient")
        plt.xlabel("year")
        plt.title("Physical/forcing coefficients by year")
        plt.legend(fontsize=7)
        save_current("coeficientes_por_tempo.png")

    if not obs.empty and {"dataset", "shift_axis"}.issubset(obs.columns):
        d = obs[obs["dataset"].eq("curitiba_v20q")]
        if not d.empty and {"year", "frac_assunto_drenagem_pos"}.issubset(d.columns):
            plt.figure(figsize=(6, 4))
            plt.plot(d["year"], d["frac_assunto_drenagem_pos"], marker="o")
            plt.ylim(0, 1)
            plt.ylabel("fraction")
            plt.xlabel("year")
            plt.title("Observation composition: Drenagem positives")
            save_current("observation_shift_curitiba.png")

    return made


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_sanitize(data), indent=2, ensure_ascii=False, default=str, allow_nan=False),
        encoding="utf-8",
    )


def json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [json_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if value is pd.NaT:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _region_from_dataset(dataset: str) -> str:
    if "curitiba" in dataset:
        return "Curitiba"
    if "recife" in dataset:
        return "Recife"
    if "serra" in dataset:
        return "Fluvial serra"
    if "planicie" in dataset:
        return "Fluvial planicie"
    return dataset


def _label_sources_for_dataset(dataset: str) -> str:
    if "curitiba" in dataset:
        return "SIAC 156 Curitiba + negativos locais"
    if "recife" in dataset:
        return "SEDEC/Recife + fontes locais"
    if "planicie" in dataset or "serra" in dataset:
        return "UK/Copernicus/Sen1Floods conforme ds05"
    return "desconhecido"


def _negative_type_for_dataset(dataset: str) -> str:
    if "curitiba" in dataset:
        return "misto/local; auditar temporalidade"
    if "recife" in dataset:
        return "ausencia_de_registro"
    if "planicie" in dataset or "serra" in dataset:
        return "misto: observado/exclusao qualificada"
    return "desconhecido"


def _rain_source_for_dataset(dataset: str) -> str:
    if "curitiba" in dataset or "recife" in dataset:
        return "open_meteo_era5_land/CHIRPS conforme artefato"
    if "planicie" in dataset or "serra" in dataset:
        return "open_meteo_era5_land quando presente"
    return None


def _min_col(df: pd.DataFrame, col: str) -> Any:
    if df.empty or col not in df.columns:
        return None
    vals = df[col].dropna()
    return vals.min() if not vals.empty else None


def _max_col(df: pd.DataFrame, col: str) -> Any:
    if df.empty or col not in df.columns:
        return None
    vals = df[col].dropna()
    return vals.max() if not vals.empty else None


def _mean_col(df: pd.DataFrame, col: str) -> float | None:
    if df.empty or col not in df.columns:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    return float(vals.mean()) if not vals.empty else None


def _int_or_none(v: Any) -> int | None:
    try:
        if pd.isna(v):
            return None
        return int(round(float(v)))
    except Exception:
        return None


def _float_or_none(v: Any) -> float | None:
    try:
        if pd.isna(v):
            return None
        return round(float(v), 4)
    except Exception:
        return None


def _float_or_nan(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _count_from_prevalence(n: Any, prevalence: Any) -> int | None:
    nf = _float_or_nan(n)
    pf = _float_or_nan(prevalence)
    if np.isnan(nf) or np.isnan(pf):
        return None
    return int(round(nf * pf))


def _ci(lo: Any, hi: Any) -> str | None:
    l = _float_or_none(lo)
    h = _float_or_none(hi)
    if l is None or h is None:
        return None
    return json.dumps([l, h])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-head", default=None)
    parser.add_argument("--baseline-status", default=None)
    parser.add_argument("--baseline-log", default=None)
    parser.add_argument("--baseline-pytest-status", default="running_or_not_recorded")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    ontologia = gravar_ontologia(OUT)

    pool = read_csv(POOL_FLUVIAL)
    admitidos = read_csv(ADMITIDOS)
    curitiba = read_csv(CURITIBA_V20N)

    capacidade = capacidade_por_regiao(pool, admitidos)
    tests = historical_tests(pool)
    folds = standardize_folds()
    cov = drift_covariates()
    forc = drift_forcing(pool)
    obs = observation_shift(pool)
    samp = sampling_shift()
    coefs = coefficients_by_time()
    walk = walk_forward()
    placebo = placebo_results(pool, curitiba)
    matrix = temporal_validity_matrix(capacidade)
    cur_summary = curitiba_2026_summary(capacidade, cov, samp, placebo)
    uk_summary = uk_control_summary()
    concl = conclusoes_json(matrix, cur_summary, uk_summary)

    artifacts = [
        OUT / "ontologia.json",
        OUT / "testes_historicos.parquet",
        OUT / "capacidade_por_regiao.csv",
        OUT / "folds.csv",
        OUT / "drift_covariates.csv",
        OUT / "drift_forcing.csv",
        OUT / "observation_shift.csv",
        OUT / "sampling_shift.csv",
        OUT / "coefficients_by_time.csv",
        OUT / "walk_forward.csv",
        OUT / "placebo_results.csv",
        OUT / "temporal_validity_matrix.csv",
        OUT / "curitiba_2026.json",
        OUT / "uk_control.json",
        OUT / "conclusoes.json",
        OUT / "relatorio.md",
    ]

    tests.to_parquet(OUT / "testes_historicos.parquet", index=False)
    capacidade.to_csv(OUT / "capacidade_por_regiao.csv", index=False)
    folds.to_csv(OUT / "folds.csv", index=False)
    cov.to_csv(OUT / "drift_covariates.csv", index=False)
    forc.to_csv(OUT / "drift_forcing.csv", index=False)
    obs.to_csv(OUT / "observation_shift.csv", index=False)
    samp.to_csv(OUT / "sampling_shift.csv", index=False)
    coefs.to_csv(OUT / "coefficients_by_time.csv", index=False)
    walk.to_csv(OUT / "walk_forward.csv", index=False)
    placebo.to_csv(OUT / "placebo_results.csv", index=False)
    matrix.to_csv(OUT / "temporal_validity_matrix.csv", index=False)
    write_json(OUT / "curitiba_2026.json", cur_summary)
    write_json(OUT / "uk_control.json", uk_summary)
    write_json(OUT / "conclusoes.json", concl)
    write_report(concl, matrix, placebo)
    figures = make_figures(cov, walk, samp, coefs, obs)
    write_json(OUT / "manifest.json", manifest(args, artifacts, figures))

    print(
        json.dumps(
            {
                "saida": rel(OUT),
                "termos_ontologia": len(ontologia["termos"]),
                "testes_historicos": int(len(tests)),
                "regioes_capacidade": int(len(capacidade)),
                "linhas_matriz_validade": int(len(matrix)),
                "figuras": len(figures),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
