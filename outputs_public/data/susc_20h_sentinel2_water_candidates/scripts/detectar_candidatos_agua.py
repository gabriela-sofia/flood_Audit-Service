"""Via B — candidatos a lâmina d'água a partir de bandas cruas Sentinel-2 (pré/pós-evento).

Generaliza o processamento ad hoc que adjudicou o candidato de Petrópolis/Valparaíso (v14) e
adiciona o refinamento indicado pela revisão de literatura (seção 3): em vez de decidir por um
índice relativo só, exigir **concordância de 2 dos 3 índices** (NDWI, MNDWI, AWEI) cruzando o
limiar de mudança pré→pós, mantendo a reflectância absoluta como filtro físico obrigatório.

Cadeia de decisão (todas as condições, nesta ordem):

1. **Filtro físico obrigatório** (o que separou água de nuvem no v14): no dia do evento,
   `B08 < 0,15` **e** `B11 < 0,15`; e **não** era assim antes. Nuvem sobe nas três bandas, água
   desce em NIR/SWIR. Pixel que não passa aqui não é candidato, por mais índice que concorde.
2. **Três índices, calculados pré e pós**:
   - NDWI  = (B03 − B08) / (B03 + B08)                       — McFeeters 1996
   - MNDWI = (B03 − B11) / (B03 + B11)                       — Xu 2006
   - AWEI_nsh = 4·(B03 − B11) − (0,25·B08 + 2,75·B12)        — Feyisa et al. 2014,
     *Remote Sensing of Environment* 140:23-35, doi:10.1016/j.rse.2013.08.029
3. **Consenso**: pelo menos 2 dos 3 índices têm de subir mais que o próprio limiar de mudança.
4. **Agregação em cluster**: componentes conexos (8-vizinhos) com tamanho mínimo.

Sobre a banda B12: AWEI_nsh exige SWIR2. Com só B03/B08/B11 o "2 de 3" degeneraria exatamente
no par NDWI+MNDWI que a literatura considera insuficiente sozinho, então o script **falha
fechado** se B12 faltar. Existe `--allow-two-index-fallback` para rodar mesmo assim, e nesse
caso todo cluster sai marcado `NDWI_MNDWI_ONLY_WEAKER` — nunca silenciosamente.

Os limiares aqui são **portas de detecção**, não features: nenhum valor calculado por este
script pode virar entrada de modelo (regra fixa do projeto).

Uso:
    python detectar_candidatos_agua.py --before-dir <dir_pre> --after-dir <dir_pos>
                                      --out-csv <candidatos.csv> [--min-cluster-px 20]
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from scipy import ndimage
from scipy.ndimage import label as cc_label

BAND_PATTERNS = {
    "B03": re.compile(r"(?<![0-9])B0?3(?![0-9])", re.I),
    "B08": re.compile(r"(?<![0-9])B0?8(?![0-9])", re.I),
    "B11": re.compile(r"(?<![0-9])B11(?![0-9])", re.I),
    "B12": re.compile(r"(?<![0-9])B12(?![0-9])", re.I),
}

# Limiares default. Todos parametrizáveis; nenhum é variavel de modelo.
NIR_SWIR_MAX = 0.15      # reflectância absoluta no dia do evento (critério físico do v14)
NDWI_DELTA = 0.10        # subida mínima pré→pós
MNDWI_DELTA = 0.10
AWEI_DELTA = 0.20        # AWEI tem escala ~4x a de reflectância
MIN_CLUSTER_PX = 20
CONSENSUS_REQUIRED = 2


@dataclass
class DetectionConfig:
    nir_swir_max: float = NIR_SWIR_MAX
    ndwi_delta: float = NDWI_DELTA
    mndwi_delta: float = MNDWI_DELTA
    awei_delta: float = AWEI_DELTA
    min_cluster_px: int = MIN_CLUSTER_PX
    consensus_required: int = CONSENSUS_REQUIRED
    allow_two_index_fallback: bool = False


@dataclass
class DetectionResult:
    clusters: list = field(default_factory=list)
    base_consenso: str = ""
    n_pixels_physical_gate: int = 0
    n_pixels_consensus: int = 0
    n_clusters_before_size_filter: int = 0
    config: dict = field(default_factory=dict)
    indices_available: list = field(default_factory=list)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.full(num.shape, np.nan, dtype="float64")
    ok = np.isfinite(den) & (np.abs(den) > 1e-9)
    out[ok] = num[ok] / den[ok]
    return out


def ndwi(b03: np.ndarray, b08: np.ndarray) -> np.ndarray:
    """McFeeters 1996."""
    return _safe_ratio(b03 - b08, b03 + b08)


def mndwi(b03: np.ndarray, b11: np.ndarray) -> np.ndarray:
    """Xu 2006."""
    return _safe_ratio(b03 - b11, b03 + b11)


def awei_nsh(b03: np.ndarray, b08: np.ndarray, b11: np.ndarray, b12: np.ndarray) -> np.ndarray:
    """AWEI sem sombra, Feyisa et al. 2014 (doi:10.1016/j.rse.2013.08.029)."""
    return 4.0 * (b03 - b11) - (0.25 * b08 + 2.75 * b12)


def physical_water_gate(
    b08_after: np.ndarray, b11_after: np.ndarray,
    b08_before: np.ndarray, b11_before: np.ndarray,
    nir_swir_max: float = NIR_SWIR_MAX,
) -> np.ndarray:
    """Reflectância absoluta baixa em NIR e SWIR no dia do evento, e não antes."""
    dark_after = (b08_after < nir_swir_max) & (b11_after < nir_swir_max)
    dark_before = (b08_before < nir_swir_max) & (b11_before < nir_swir_max)
    return dark_after & ~dark_before


def consensus_mask(index_changes: dict[str, tuple[np.ndarray, float]], required: int = CONSENSUS_REQUIRED):
    """Conta quantos índices subiram mais que o próprio limiar; devolve (máscara, contagem)."""
    votes = None
    for _, (delta, threshold) in index_changes.items():
        v = (delta > threshold) & np.isfinite(delta)
        votes = v.astype("int8") if votes is None else votes + v.astype("int8")
    if votes is None:
        raise ValueError("nenhum índice fornecido para o consenso")
    return votes >= required, votes


def detectar_candidatos_agua(
    bands_before: dict[str, np.ndarray],
    bands_after: dict[str, np.ndarray],
    config: DetectionConfig | None = None,
    transform=None,
    crs=None,
) -> DetectionResult:
    """Aplica filtro físico + consenso de índices e agrega em clusters."""
    cfg = config or DetectionConfig()
    for required in ("B03", "B08", "B11"):
        for name, d in (("antes", bands_before), ("depois", bands_after)):
            if required not in d:
                raise ValueError(f"banda {required} ausente no conjunto '{name}'")

    has_b12 = "B12" in bands_before and "B12" in bands_after
    if not has_b12 and not cfg.allow_two_index_fallback:
        raise ValueError(
            "B12 (SWIR2) ausente: AWEI_nsh (Feyisa et al. 2014) não pode ser calculado, e o "
            "consenso 2-de-3 degeneraria no par NDWI+MNDWI que a literatura considera "
            "insuficiente sozinho. Baixe também B12 (Raw, 32-bit) ou rode com "
            "--allow-two-index-fallback, que marca todo cluster como NDWI_MNDWI_ONLY_WEAKER."
        )

    gate = physical_water_gate(
        bands_after["B08"], bands_after["B11"],
        bands_before["B08"], bands_before["B11"],
        cfg.nir_swir_max,
    )

    changes = {
        "ndwi": (ndwi(bands_after["B03"], bands_after["B08"])
                 - ndwi(bands_before["B03"], bands_before["B08"]), cfg.ndwi_delta),
        "mndwi": (mndwi(bands_after["B03"], bands_after["B11"])
                  - mndwi(bands_before["B03"], bands_before["B11"]), cfg.mndwi_delta),
    }
    if has_b12:
        changes["awei_nsh"] = (
            awei_nsh(bands_after["B03"], bands_after["B08"], bands_after["B11"], bands_after["B12"])
            - awei_nsh(bands_before["B03"], bands_before["B08"], bands_before["B11"], bands_before["B12"]),
            cfg.awei_delta,
        )

    agree, votes = consensus_mask(changes, cfg.consensus_required)
    candidate = gate & agree

    labels, n_found = cc_label(candidate, structure=np.ones((3, 3), dtype=int))
    result = DetectionResult(
        base_consenso="NDWI_MNDWI_AWEI" if has_b12 else "NDWI_MNDWI_ONLY_WEAKER",
        n_pixels_physical_gate=int(gate.sum()),
        n_pixels_consensus=int(candidate.sum()),
        n_clusters_before_size_filter=int(n_found),
        config=cfg.__dict__.copy(),
        indices_available=list(changes),
    )

    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs is not None else None

    # Agregação vetorizada (scipy.ndimage), não um np.where por cluster: com dado real e
    # ruidoso (milhares de componentes) o loop ingênuo é O(n_clusters x tamanho_da_imagem) e
    # trava. ndimage.sum/mean/center_of_mass fazem a mesma conta em poucas passadas O(imagem).
    if n_found:
        all_idx = np.arange(1, n_found + 1)
        sizes = ndimage.sum(candidate, labels, index=all_idx)
        keep = sizes >= cfg.min_cluster_px
        kept_idx = all_idx[keep]
    else:
        kept_idx = np.array([], dtype=int)

    if len(kept_idx):
        sizes_kept = ndimage.sum(candidate, labels, index=kept_idx)
        centroids = ndimage.center_of_mass(candidate, labels, index=kept_idx)
        votes_mean = ndimage.mean(votes.astype("float64"), labels, index=kept_idx)
        b08_depois_media = ndimage.mean(bands_after["B08"], labels, index=kept_idx)
        b11_depois_media = ndimage.mean(bands_after["B11"], labels, index=kept_idx)
        b08_antes_media = ndimage.mean(bands_before["B08"], labels, index=kept_idx)
        b11_antes_media = ndimage.mean(bands_before["B11"], labels, index=kept_idx)

        for i, lab in enumerate(kept_idx):
            r, c = float(centroids[i][0]), float(centroids[i][1])
            entry = {
                "cluster_id": f"S2WC_{int(lab):05d}",
                "n_pixels": int(sizes_kept[i]),
                "centroide_linha": round(r, 2),
                "centroide_coluna": round(c, 2),
                "base_consenso": result.base_consenso,
                "votos_indice_media": round(float(votes_mean[i]), 3),
                "b08_depois_media": round(float(b08_depois_media[i]), 4),
                "b11_depois_media": round(float(b11_depois_media[i]), 4),
                "b08_antes_media": round(float(b08_antes_media[i]), 4),
                "b11_antes_media": round(float(b11_antes_media[i]), 4),
                "adjudicado": "false",
            }
            if transform is not None:
                x, y = transform * (c + 0.5, r + 0.5)
                entry["easting"], entry["northing"] = round(x, 2), round(y, 2)
                if to_wgs84 is not None:
                    lon, lat = to_wgs84.transform(x, y)
                    entry["lon"], entry["lat"] = round(lon, 6), round(lat, 6)
            result.clusters.append(entry)

    result.clusters.sort(key=lambda e: -e["n_pixels"])
    return result


def resolve_band_files(directory: Path | str) -> dict[str, Path]:
    """Casa arquivos da pasta com B03/B08/B11/B12 pelo nome."""
    directory = Path(directory)
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".tif", ".tiff"):
            continue
        for band, pattern in BAND_PATTERNS.items():
            if pattern.search(path.stem) and band not in found:
                found[band] = path
    return found


def load_bands(directory: Path | str, scale: str | float = "auto"):
    """Lê as bandas de uma pasta; devolve (dict de arrays, transform, crs)."""
    files = resolve_band_files(directory)
    if not files:
        raise FileNotFoundError(f"nenhum GeoTIFF de banda reconhecido em {directory}")
    arrays, transform, crs, shape = {}, None, None, None
    for band, path in files.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float64")
            if transform is None:
                transform, crs, shape = src.transform, src.crs, arr.shape
            elif arr.shape != shape:
                raise ValueError(
                    f"banda {band} tem grade {arr.shape}, diferente de {shape} — "
                    "reamostre antes; este script não reamostra por conta própria"
                )
        factor = 1.0
        if scale == "auto":
            finite = arr[np.isfinite(arr)]
            if finite.size and np.nanpercentile(finite, 99) > 1.5:
                factor = 1 / 10000.0  # DN L2A de 0-10000 em vez de reflectância 0-1
        elif scale not in (None, "none"):
            factor = float(scale)
        arrays[band] = arr * factor
    return arrays, transform, crs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before-dir", required=True, help="pasta com as bandas do dia de referência")
    p.add_argument("--after-dir", required=True, help="pasta com as bandas do dia do evento")
    p.add_argument("--out-csv", required=True)
    p.add_argument("--json-out")
    p.add_argument("--scale", default="auto")
    p.add_argument("--nir-swir-max", type=float, default=NIR_SWIR_MAX)
    p.add_argument("--ndwi-delta", type=float, default=NDWI_DELTA)
    p.add_argument("--mndwi-delta", type=float, default=MNDWI_DELTA)
    p.add_argument("--awei-delta", type=float, default=AWEI_DELTA)
    p.add_argument("--min-cluster-px", type=int, default=MIN_CLUSTER_PX)
    p.add_argument("--allow-two-index-fallback", action="store_true")
    args = p.parse_args(argv)

    before, transform, crs = load_bands(args.before_dir, args.scale)
    after, _, _ = load_bands(args.after_dir, args.scale)
    cfg = DetectionConfig(
        nir_swir_max=args.nir_swir_max, ndwi_delta=args.ndwi_delta,
        mndwi_delta=args.mndwi_delta, awei_delta=args.awei_delta,
        min_cluster_px=args.min_cluster_px,
        allow_two_index_fallback=args.allow_two_index_fallback,
    )
    res = detectar_candidatos_agua(before, after, cfg, transform=transform, crs=crs)

    import csv

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if res.clusters:
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(res.clusters[0]))
            writer.writeheader()
            writer.writerows(res.clusters)
    else:
        out_csv.write_text("cluster_id,n_pixels\n", encoding="utf-8")

    summary = {
        "n_clusters_listed": len(res.clusters),
        "base_consenso": res.base_consenso,
        "indices_available": res.indices_available,
        "n_pixels_physical_gate": res.n_pixels_physical_gate,
        "n_pixels_consensus": res.n_pixels_consensus,
        "n_clusters_before_size_filter": res.n_clusters_before_size_filter,
        "config": res.config,
        "note": "clusters são CANDIDATOS; adjudicação SUSC-20A é passo humano separado",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
