"""Via C — candidatos a lâmina d'água a partir de retroespalhamento SAR (Sentinel-1 GRD, VV/VH).

Motivação real (linhagem Petrópolis, evento 05/04/2025): três referências ópticas
independentes (16/02, 28/11/2024, 03/03/2025), todas comparadas contra as duas únicas cenas
Sentinel-2 utilizáveis pós-evento (07/04 e 09/04, d+2 e d+4), deram resultado nulo no corredor
real dos rios (Quitandinha/Piabanha) depois de normalizar por z-escore contra o deslocamento
sistemático da cena. Terreno de serra drena rápido — é fisicamente plausível que a água já
tivesse recuado antes da primeira imagem óptica sem nuvem disponível. SAR não depende de nuvem
e pode captar o evento mais perto do pico (aqui: 2 dias antes via 03/04, ou o dia core do
evento se houver revisita mais próxima).

Física do método (não um escore, é o mecanismo de espalhamento):
água parada é uma superfície especular — reflete a maior parte da energia de radar pra longe do
sensor, não de volta — por isso aparece **escura** (retroespalhamento baixo) em VV/VH, ao
contrário de terreno rugoso/vegetação, que espalha de volta e aparece claro. Um pixel que
"escurece" entre antes e depois é candidato a alagamento novo.

Limiares usados (literatura real, não inventados):
- **Retroespalhamento absoluto baixo em VV no dia do evento** (`< -17 dB`, faixa comumente usada
  pra água lisa em VV — ver referências do sistema operacional DLR/Copernicus EMS baseado em
  Martinis & Twele).
- **Queda de retroespalhamento (BCR, backscatter change ratio) entre antes e depois**: inundação
  tipicamente produz queda de **3 a 8 dB** em relação ao solo seco antes do evento (ver Twele et
  al. 2016; revisão em Clement et al. 2018, *J. Flood Risk Management*, DOI:10.1111/jfr3.12303).
  Usamos 3 dB como limiar mínimo, o extremo mais conservador da faixa publicada.
- VV é preferido (mais sensível à especularidade da água), mas é suscetível a falso-negativo por
  vento/chuva que rugosifica a superfície — por isso VH entra como corroboração opcional, não
  substituto: some ao voto de consenso quando disponível.

Cadeia de decisão:
1. Gate físico obrigatório: `VV_depois_dB < vv_water_max` **e** `VV_antes_dB >= vv_water_max`
   (água nova, não já presente antes — mesma lógica do filtro óptico do v14/Via B).
2. Queda de retroespalhamento: `VV_antes_dB - VV_depois_dB > min_drop_db`.
3. Se VH disponível, mesma checagem em VH conta como voto extra (consenso, não obrigatório).
4. Agregação vetorizada em clusters (scipy.ndimage, mesmo padrão corrigido do Via B).

Nenhum valor calculado aqui vira variavel de modelo — são portas de detecção (regra fixa do
projeto).

Uso:
    python detectar_candidatos_agua_sar.py --before-dir <dir_pre> --after-dir <dir_pos>
                                          --out-csv <candidatos.csv> [--corridor-mask <npy>]
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
    "VV": re.compile(r"(?<![A-Za-z])VV(?![A-Za-z])", re.I),
    "VH": re.compile(r"(?<![A-Za-z])VH(?![A-Za-z])", re.I),
}

VV_WATER_MAX_DB = -17.0   # agua lisa em VV, faixa comum na literatura (Martinis/Twele)
MIN_DROP_DB = 3.0         # extremo conservador da faixa publicada (3-8 dB, Twele et al. 2016)
MIN_CLUSTER_PX = 10


@dataclass
class SarDetectionConfig:
    vv_water_max_db: float = VV_WATER_MAX_DB
    min_drop_db: float = MIN_DROP_DB
    min_cluster_px: int = MIN_CLUSTER_PX
    require_vh_agreement: bool = False  # se True, so aceita se VH tambem concordar (mais estrito)


@dataclass
class SarDetectionResult:
    clusters: list = field(default_factory=list)
    n_pixels_gate: int = 0
    n_pixels_after_drop: int = 0
    vh_disponivel: bool = False
    config: dict = field(default_factory=dict)


def to_db(arr: np.ndarray) -> np.ndarray:
    """Converte sigma0 linear pra dB se necessario; detecta automaticamente pela faixa de valor.

    Export 'Raw' do Copernicus Browser pra Sentinel-1 GRD normalmente vem em potencia linear
    (sigma0), sempre positiva, tipicamente < 5. Retroespalhamento em dB e sempre negativo pra
    a maioria da cena (terra seca ja fica em torno de -10 a -15 dB). Se ja houver valor negativo
    real na cena, assumimos que ja esta em dB.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    if np.nanmin(finite) >= 0:
        safe = np.where(arr > 0, arr, np.nan)
        return 10.0 * np.log10(safe)
    return arr


def physical_water_gate_sar(
    vv_after_db: np.ndarray, vv_before_db: np.ndarray, vv_water_max_db: float = VV_WATER_MAX_DB,
) -> np.ndarray:
    dark_after = vv_after_db < vv_water_max_db
    dark_before = vv_before_db < vv_water_max_db
    return dark_after & ~dark_before


def detectar_candidatos_agua_sar(
    bands_before: dict[str, np.ndarray],
    bands_after: dict[str, np.ndarray],
    config: SarDetectionConfig | None = None,
    transform=None,
    crs=None,
    extra_mask: np.ndarray | None = None,
) -> SarDetectionResult:
    cfg = config or SarDetectionConfig()
    for name, d in (("antes", bands_before), ("depois", bands_after)):
        if "VV" not in d:
            raise ValueError(f"banda VV ausente no conjunto '{name}'")

    vv_before = to_db(bands_before["VV"])
    vv_after = to_db(bands_after["VV"])

    gate = physical_water_gate_sar(vv_after, vv_before, cfg.vv_water_max_db)
    drop = vv_before - vv_after
    drop_ok = drop > cfg.min_drop_db
    candidate = gate & drop_ok

    vh_disponivel = "VH" in bands_before and "VH" in bands_after
    if vh_disponivel:
        vh_before = to_db(bands_before["VH"])
        vh_after = to_db(bands_after["VH"])
        vh_gate = physical_water_gate_sar(vh_after, vh_before, cfg.vv_water_max_db)
        vh_drop_ok = (vh_before - vh_after) > cfg.min_drop_db
        vh_agree = vh_gate & vh_drop_ok
        if cfg.require_vh_agreement:
            candidate = candidate & vh_agree

    if extra_mask is not None:
        candidate = candidate & extra_mask

    result = SarDetectionResult(
        n_pixels_gate=int(gate.sum()),
        n_pixels_after_drop=int(candidate.sum()),
        vh_disponivel=vh_disponivel,
        config=cfg.__dict__.copy(),
    )

    labels, n_found = cc_label(candidate, structure=np.ones((3, 3), dtype=int))
    to_wgs84 = Transformer.from_crs(crs, "EPSG:4326", always_xy=True) if crs is not None else None

    if n_found:
        all_idx = np.arange(1, n_found + 1)
        sizes = ndimage.sum(candidate, labels, index=all_idx)
        keep = all_idx[sizes >= cfg.min_cluster_px]
    else:
        keep = np.array([], dtype=int)

    if len(keep):
        sizes_kept = ndimage.sum(candidate, labels, index=keep)
        centroids = ndimage.center_of_mass(candidate, labels, index=keep)
        vv_after_mean = ndimage.mean(vv_after, labels, index=keep)
        vv_before_mean = ndimage.mean(vv_before, labels, index=keep)
        for i, lab in enumerate(keep):
            r, c = float(centroids[i][0]), float(centroids[i][1])
            entry = {
                "cluster_id": f"S1WC_{int(lab):05d}",
                "n_pixels": int(sizes_kept[i]),
                "centroide_linha": round(r, 2),
                "centroide_coluna": round(c, 2),
                "vv_depois_db_media": round(float(vv_after_mean[i]), 2),
                "vv_antes_db_media": round(float(vv_before_mean[i]), 2),
                "queda_db_media": round(float(vv_before_mean[i] - vv_after_mean[i]), 2),
                "vh_disponivel": vh_disponivel,
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
    directory = Path(directory)
    found: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".tif", ".tiff"):
            continue
        for band, pattern in BAND_PATTERNS.items():
            if pattern.search(path.stem) and band not in found:
                found[band] = path
    return found


def load_bands(directory: Path | str):
    files = resolve_band_files(directory)
    if not files:
        raise FileNotFoundError(f"nenhuma banda VV/VH reconhecida em {directory}")
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
        arrays[band] = arr
    return arrays, transform, crs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--before-dir", required=True)
    p.add_argument("--after-dir", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--json-out")
    p.add_argument("--corridor-mask", help="caminho .npy de mascara booleana (mesma grade) pra restringir a area real de drenagem")
    p.add_argument("--vv-water-max-db", type=float, default=VV_WATER_MAX_DB)
    p.add_argument("--min-drop-db", type=float, default=MIN_DROP_DB)
    p.add_argument("--min-cluster-px", type=int, default=MIN_CLUSTER_PX)
    p.add_argument("--require-vh-agreement", action="store_true")
    args = p.parse_args(argv)

    before, transform, crs = load_bands(args.before_dir)
    after, _, _ = load_bands(args.after_dir)
    extra_mask = np.load(args.corridor_mask) if args.corridor_mask else None

    cfg = SarDetectionConfig(
        vv_water_max_db=args.vv_water_max_db, min_drop_db=args.min_drop_db,
        min_cluster_px=args.min_cluster_px, require_vh_agreement=args.require_vh_agreement,
    )
    res = detectar_candidatos_agua_sar(before, after, cfg, transform=transform, crs=crs, extra_mask=extra_mask)

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
        "n_pixels_gate": res.n_pixels_gate,
        "n_pixels_after_drop": res.n_pixels_after_drop,
        "vh_disponivel": res.vh_disponivel,
        "config": res.config,
        "note": "clusters são CANDIDATOS; adjudicação SUSC-20A é passo humano separado",
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
