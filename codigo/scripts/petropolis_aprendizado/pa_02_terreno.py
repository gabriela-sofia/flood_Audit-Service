"""Atributos de encosta pela mesma cadeia no doador (Idai) e no destino (Petropolis).

A parte hidrologica (preenchimento, declividade, area contribuinte especifica
D-infinito) e a do servico: `_derivar_terreno` de `geoprocessamento_sob_demanda`,
sem reimplementacao. Curvaturas e ponteiro D8 usam o mesmo binario WhiteboxTools;
relevo, TPI e estatisticas de vizinhanca sao numpy, identicos nas duas regioes.

Atributos (todos anteriores a qualquer evento; nenhum usa imagem pos-evento):
  decl_m3        declividade media 3x3 (graus)
  decl_max5      declividade maxima 5x5 (graus)  -- tolera erro posicional
  curv_plan_m3   curvatura em planta media 3x3 (WhiteboxTools)
  curv_perf_m3   curvatura em perfil media 3x3 (WhiteboxTools)
  logsca_m3      log10 da area contribuinte especifica D-inf, media 3x3
  relevo7        amplitude altimetrica 7x7 (~210 m)
  tpi11          posicao topografica 11x11 (~330 m): elevacao - media local

HAND e TWI ficam de fora de proposito: sao as variaveis do modelo de inundacao,
e a tarefa de encosta nao herda os coeficientes nem as entradas da agua.
Elevacao absoluta fica de fora porque nao e transferivel entre Idai e Petropolis.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from pa_comum import FONTES, ROOT, TERRENO, Etapa, gravar_json, rel, sha256

sys.path.insert(0, str(ROOT / "scripts" / "servico"))
sys.path.insert(0, str(ROOT / "scripts"))

ATRIBUTOS = ["decl_m3", "decl_max5", "curv_plan_m3", "curv_perf_m3", "logsca_m3", "relevo7", "tpi11"]
BLOCOS = {"declive": ["decl_m3", "decl_max5"], "curvatura": ["curv_plan_m3", "curv_perf_m3"],
          "fluxo": ["logsca_m3"], "relevo": ["relevo7", "tpi11"]}
REGIOES = {"petropolis": "fabdem_petropolis_epsg31983_30m.tif", "idai": "fabdem_idai_epsg32736_30m.tif"}


def _media(a, k):
    from scipy.ndimage import uniform_filter
    v = np.isfinite(a)
    num = uniform_filter(np.where(v, a, 0.0), size=k, mode="nearest")
    den = uniform_filter(v.astype("float64"), size=k, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~v] = np.nan
    return out


def _max(a, k):
    from scipy.ndimage import maximum_filter
    out = maximum_filter(np.where(np.isfinite(a), a, -np.inf), size=k, mode="nearest")
    out[~np.isfinite(a)] = np.nan
    return out


def _min(a, k):
    from scipy.ndimage import minimum_filter
    out = minimum_filter(np.where(np.isfinite(a), a, np.inf), size=k, mode="nearest")
    out[~np.isfinite(a)] = np.nan
    return out


def _ler(p):
    import rasterio
    with rasterio.open(p) as r:
        a = r.read(1).astype("float64")
        nd = r.nodata
        perfil = r.profile
    if nd is not None:
        a[a == nd] = np.nan
    a[a < -1e30] = np.nan
    return a, perfil


def _wbt_extra(dem, destino) -> dict:
    """Curvaturas e D8 com o mesmo binario e a mesma disciplina de repeticao do servico."""
    import whitebox
    wbt = whitebox.WhiteboxTools()
    log: list[str] = []
    wbt.verbose = True
    wbt.default_callback = log.append
    wbt.set_working_dir(str(destino))
    feitos = {}
    for nome, saida, kw in [("plan_curvature", "curv_plan.tif", {}),
                            ("profile_curvature", "curv_perf.tif", {}),
                            ("d8_pointer", "d8_pointer_esri.tif", {"esri_pntr": True})]:
        alvo = destino / saida
        entrada = destino / "dem_filled.tif" if nome == "d8_pointer" else dem
        for tentativa in range(1, 4):
            if alvo.exists():
                alvo.unlink()
            getattr(wbt, nome)(str(entrada), str(alvo), **kw)
            for _ in range(20):
                if alvo.exists():
                    break
                time.sleep(0.5)
            if alvo.exists():
                feitos[nome] = {"saida": rel(alvo), "tentativas": tentativa}
                break
        else:
            raise RuntimeError(f"WhiteboxTools.{nome} falhou: {' | '.join(log)[-800:]}")
    return feitos


def derivar(regiao: str) -> dict:
    import rasterio
    from geoprocessamento_sob_demanda import _derivar_terreno

    dem = FONTES / "fabdem" / REGIOES[regiao]
    destino = TERRENO / regiao
    destino.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    canon = _derivar_terreno(dem, destino)
    extra = _wbt_extra(dem, destino)

    z, perfil = _ler(dem)
    decl, _ = _ler(destino / "slope_deg_wbt.tif")
    sca, _ = _ler(destino / "facc_dinf.tif")
    cp, _ = _ler(destino / "curv_plan.tif")
    cf, _ = _ler(destino / "curv_perf.tif")
    validos = np.isfinite(z) & np.isfinite(decl) & np.isfinite(sca)
    for a in (decl, sca, cp, cf):
        a[~validos] = np.nan
    z = np.where(validos, z, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        logsca = np.log10(np.clip(sca, 1e-3, None))
    pilha = {
        "decl_m3": _media(decl, 3),
        "decl_max5": _max(decl, 5),
        "curv_plan_m3": _media(cp, 3),
        "curv_perf_m3": _media(cf, 3),
        "logsca_m3": _media(logsca, 3),
        "relevo7": _max(z, 7) - _min(z, 7),
        "tpi11": z - _media(z, 11),
    }
    arq = destino / "atributos_encosta.tif"
    perfil.update(count=len(ATRIBUTOS), dtype="float32", nodata=np.nan, compress="deflate",
                  tiled=True, predictor=3)
    with rasterio.open(arq, "w", **perfil) as d:
        for i, nome in enumerate(ATRIBUTOS, 1):
            d.write(pilha[nome].astype("float32"), i)
            d.set_band_description(i, nome)
    resumo = {n: {"p1": float(np.nanpercentile(v, 1)), "p50": float(np.nanpercentile(v, 50)),
                  "p99": float(np.nanpercentile(v, 99)), "nan": int(np.isnan(v).sum())}
              for n, v in pilha.items()}
    return {"regiao": regiao, "dem": rel(dem), "dem_sha256": sha256(dem),
            "cadeia_canonica": {k: (rel(v) if isinstance(v, str) else v) for k, v in canon.items()},
            "wbt_extra": extra, "atributos": rel(arq), "atributos_sha256": sha256(arq),
            "celulas_validas": int(validos.sum()), "forma": list(z.shape), "resumo": resumo,
            "parede_s": round(time.time() - t0, 1)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--regioes", nargs="*", default=list(REGIOES))
    a = p.parse_args()
    e = Etapa("pa_02_terreno")
    reg = {"atributos": ATRIBUTOS, "blocos": BLOCOS,
           "excluidos": {"hand_m/twi": "variaveis da agua; separacao de tarefas",
                         "elevacao": "nao transferivel entre regioes",
                         "aspecto": "orientacao regional; fora do nucleo transferivel"}}
    for r in a.regioes:
        print(f"[terreno] {r}", flush=True)
        reg[r] = derivar(r)
        e.entrada(FONTES / "fabdem" / REGIOES[r])
        e.saida(TERRENO / r / "atributos_encosta.tif")
    gravar_json(TERRENO / "terreno.json", reg)
    e.saida(TERRENO / "terreno.json")
    e.fechar(regioes=a.regioes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
