"""Equivalência do caminho social sem geopandas (`social_projecao.reprojetar`) com o original (`GeoSeries.to_crs`).

A referência roda em OUTRO processo, no ambiente local que tem geopandas (miniconda `revp-susc`): nenhum binário de um
ambiente é carregado no outro. Os dois lados chamam o mesmo `social07_servico_medidas.consultar` sobre o mesmo pacote
com hash conferido; só a reprojeção muda.

Casos (fixos, sem olhar resultado): AOI dos testes HTTP, cidade inteira, AOI fora, AOI cruzando a borda, AOIs
aleatórias com semente fixa e lado de 0,002° a 0,08° dentro da extensão de cada região (curitiba, rs_emsr720,
am_piloto), AOI inválida. Compara: coordenadas de todos os setores reprojetados (diferença máxima em metros), áreas
(diferença relativa), e a resposta inteira por caso — chaves, textos e inteiros iguais; números reais com tolerância
relativa 1e-9 (a versão do PROJ difere entre os ambientes e isso é declarado).
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pt_comum import LINHAGEM, REPO, gravar, rel, sha

PY_REF = Path.home() / "miniconda3" / "envs" / "revp-susc" / "python.exe"
SAIDA = LINHAGEM / "social"
TOL_REL = 1e-9


def casos() -> list[dict]:
    import social07_servico_medidas as s7
    from shapely.geometry import shape
    _, dados = s7._carregar(str(s7.PACOTE), s7.sha(s7.PACOTE / "manifesto.json"))
    out = [{"nome": "curitiba_aoi_teste_http", "regiao": "curitiba",
            "aoi": {"lat_min": -25.609, "lon_min": -49.359, "lat_max": -25.601, "lon_max": -49.351}},
           {"nome": "curitiba_fora", "regiao": "curitiba", "aoi": {"lon_min": 0, "lat_min": 0, "lon_max": 1, "lat_max": 1}},
           {"nome": "curitiba_invalida", "regiao": "curitiba", "aoi": {"lon_min": -49.2, "lat_min": -25.4, "lon_max": -49.3, "lat_max": -25.3}},
           {"nome": "regiao_nao_coberta", "regiao": "petropolis", "aoi": {"lon_min": -43.2, "lat_min": -22.5, "lon_max": -43.1, "lat_max": -22.4}}]
    rng = np.random.default_rng(20261019)
    for regiao, n in (("curitiba", 40), ("rs_emsr720", 12), ("am_piloto", 12)):
        b = np.array([shape(f["geometry"]).bounds for f in dados[regiao]["features"]])
        x0, y0, x1, y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        out.append({"nome": f"{regiao}_inteira", "regiao": regiao,
                    "aoi": {"lon_min": float(x0), "lat_min": float(y0), "lon_max": float(x1), "lat_max": float(y1)}})
        out.append({"nome": f"{regiao}_borda", "regiao": regiao,
                    "aoi": {"lon_min": float(x0 - .02), "lat_min": float((y0 + y1) / 2 - .02), "lon_max": float(x0 + .02), "lat_max": float((y0 + y1) / 2 + .02)}})
        for k in range(n):
            lado = float(rng.uniform(.002, .08))
            cx, cy = float(rng.uniform(x0, x1)), float(rng.uniform(y0, y1))
            out.append({"nome": f"{regiao}_aleatoria_{k:02d}", "regiao": regiao,
                        "aoi": {"lon_min": cx - lado / 2, "lat_min": cy - lado / 2, "lon_max": cx + lado / 2, "lat_max": cy + lado / 2}})
    return out


def comparar(a, b, caminho="", dif=None):
    """Lista de divergências; acumula a maior diferença relativa entre reais."""
    dif = dif if dif is not None else {"max_rel": 0.0, "divergencias": []}
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            dif["divergencias"].append(f"{caminho}: chaves {sorted(set(a) ^ set(b))}")
        for k in set(a) & set(b):
            comparar(a[k], b[k], f"{caminho}.{k}", dif)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            dif["divergencias"].append(f"{caminho}: tamanho {len(a)} vs {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            comparar(x, y, f"{caminho}[{i}]", dif)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, int) and isinstance(b, int):
        if a != b:
            dif["divergencias"].append(f"{caminho}: {a!r} vs {b!r}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        r = abs(a - b) / max(abs(a), abs(b), 1e-300)
        dif["max_rel"] = max(dif["max_rel"], r)
        if r > TOL_REL:
            dif["divergencias"].append(f"{caminho}: {a!r} vs {b!r} (rel {r:.2e})")
    elif a != b:
        dif["divergencias"].append(f"{caminho}: {str(a)[:60]!r} vs {str(b)[:60]!r}")
    return dif


def executar() -> dict:
    import pyproj
    import shapely
    import social07_servico_medidas as s7
    from shapely.geometry import shape
    from social_projecao import reprojetar
    if not PY_REF.exists():
        raise FileNotFoundError(f"ambiente de referencia ausente: {PY_REF}")
    cs = casos()
    with tempfile.TemporaryDirectory(dir=LINHAGEM.parent) as tmp:
        t = Path(tmp)
        (t / "aois.json").write_text(json.dumps(cs), encoding="utf-8")
        p = subprocess.run([str(PY_REF), "-I", str(REPO / "scripts/planejamento/pt_social_ref_geopandas.py"),
                            "--aois", str(t / "aois.json"), "--saida", str(t / "ref.json"), "--coords", str(t / "ref.npz")],
                           capture_output=True, text=True, timeout=1800, cwd=REPO)
        if "===END===" not in p.stdout:
            raise RuntimeError(f"referencia falhou: {p.stderr[-2000:]}")
        ref = json.loads((t / "ref.json").read_text(encoding="utf-8"))
        npz = dict(np.load(t / "ref.npz"))
    resultados, geo = {}, {}
    for regiao in sorted({c["regiao"] for c in cs if c["regiao"] in ("curitiba", "rs_emsr720", "am_piloto")}):
        _, dados = s7._carregar(str(s7.PACOTE), s7.sha(s7.PACOTE / "manifesto.json"))
        proj = reprojetar([shape(f["geometry"]) for f in dados[regiao]["features"]])
        co = shapely.get_coordinates(np.asarray(proj, dtype=object))
        ar = np.array([g.area for g in proj])
        rc, ra = npz[f"{regiao}_coords"], npz[f"{regiao}_areas"]
        geo[regiao] = {"setores": len(proj), "vertices": int(len(co)), "mesmo_numero_de_vertices": co.shape == rc.shape,
                       "dif_max_coordenada_m": float(np.abs(co - rc).max()) if co.shape == rc.shape else None,
                       "dif_rel_max_area": float((np.abs(ar - ra) / ra).max()),
                       "area_total_km2": float(ar.sum() / 1e6), "validos": int(sum(g.is_valid for g in proj))}
    for c in cs:
        try:
            meu = {"ok": True, "resposta": s7.consultar(c["regiao"], c["aoi"])}
        except (ValueError, KeyError) as e:
            meu = {"ok": False, "erro": f"{type(e).__name__}: {e}"}
        d = comparar(ref["casos"][c["nome"]], meu)
        ctx = (meu.get("resposta") or {}).get("exposicao", {}).get("contexto_aoi") or {}
        resultados[c["nome"]] = {"regiao": c["regiao"], "ok": meu["ok"], "erro": meu.get("erro"),
                                 "setores_intersectados": (meu.get("resposta") or {}).get("setores_intersectados"),
                                 "unidades_no_contexto": len(ctx.get("unidades_intersectadas") or []),
                                 "max_rel": d["max_rel"], "divergencias": d["divergencias"][:5], "n_divergencias": len(d["divergencias"])}
    n_div = sum(r["n_divergencias"] for r in resultados.values())
    out = {"tipo": "equivalencia_social_sem_geopandas", "casos": len(cs),
           "referencia": ref["versoes"], "servico": {"shapely": shapely.__version__, "pyproj": pyproj.__version__, "proj": pyproj.proj_version_str},
           "binarios_misturados": False, "pacote_sha256": sha(s7.PACOTE / "manifesto.json"), "geometria_por_regiao": geo,
           "casos_com_divergencia": [k for k, r in resultados.items() if r["n_divergencias"]], "divergencias_total": n_div,
           "max_rel_global": max(r["max_rel"] for r in resultados.values()),
           "erros_iguais": all(ref["casos"][k]["ok"] == r["ok"] for k, r in resultados.items()),
           "casos_com_interseccao": sum(1 for r in resultados.values() if (r["setores_intersectados"] or 0) > 0),
           "veredito": "equivalente" if n_div == 0 and all(g["mesmo_numero_de_vertices"] for g in geo.values()) else "divergente",
           "codigo": {rel(REPO / "scripts/social/social_projecao.py"): sha(REPO / "scripts/social/social_projecao.py")}}
    gravar(SAIDA / "equivalencia_casos.json", resultados)
    gravar(SAIDA / "equivalencia.json", out)
    return out
