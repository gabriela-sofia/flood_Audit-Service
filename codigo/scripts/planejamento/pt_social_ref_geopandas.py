"""Referência da equivalência social: roda no ambiente que TEM geopandas (revp-susc), em processo separado.

Injeta `social_projecao.reprojetar` implementado com `geopandas.GeoSeries.to_crs` (o código original do serviço) e
chama `social07_servico_medidas.consultar` sem nenhuma outra mudança. Grava a saída por AOI e as coordenadas/áreas
de todos os setores reprojetados. Não importa nada do ambiente do serviço; só lê arquivos.

    <python do revp-susc> -I pt_social_ref_geopandas.py --aois A.json --saida S.json --coords C.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "social"))


def _reprojetar_gpd(geometrias, origem=4326, destino=6933):
    import geopandas as gpd
    return list(gpd.GeoSeries(list(geometrias), crs=origem).to_crs(destino))


def main() -> int:
    import geopandas
    import numpy as np
    import pyproj
    import shapely
    from shapely.geometry import shape
    ap = argparse.ArgumentParser()
    ap.add_argument("--aois", required=True)
    ap.add_argument("--saida", required=True)
    ap.add_argument("--coords", required=True)
    a = ap.parse_args()
    mod = types.ModuleType("social_projecao")
    mod.reprojetar = _reprojetar_gpd
    sys.modules["social_projecao"] = mod
    import social07_servico_medidas as s7
    casos = json.loads(Path(a.aois).read_text(encoding="utf-8"))
    saida = {"versoes": {"geopandas": geopandas.__version__, "shapely": shapely.__version__, "pyproj": pyproj.__version__,
                         "proj": pyproj.proj_version_str, "python": sys.version.split()[0], "executavel": sys.executable},
             "casos": {}}
    for c in casos:
        try:
            saida["casos"][c["nome"]] = {"ok": True, "resposta": s7.consultar(c["regiao"], c["aoi"])}
        except (ValueError, KeyError) as e:
            saida["casos"][c["nome"]] = {"ok": False, "erro": f"{type(e).__name__}: {e}"}
    arrays = {}
    for regiao in sorted({c["regiao"] for c in casos} & {"curitiba", "rs_emsr720", "am_piloto"}):
        _, dados = s7._carregar(str(s7.PACOTE), s7.sha(s7.PACOTE / "manifesto.json"))
        geoms = [shape(f["geometry"]) for f in dados[regiao]["features"]]
        proj = _reprojetar_gpd(geoms)
        arrays[f"{regiao}_coords"] = shapely.get_coordinates(np.asarray(proj, dtype=object))
        arrays[f"{regiao}_areas"] = np.array([g.area for g in proj])
    np.savez_compressed(a.coords, **arrays)
    Path(a.saida).write_text(json.dumps(saida, ensure_ascii=False, default=str), encoding="utf-8")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
