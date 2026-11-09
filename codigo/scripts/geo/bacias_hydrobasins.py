"""Bacias HydroBASINS v1c no acervo: unidade hidrologica e conectividade montante-jusante.

-------------------------------------------------------------------------
A LACUNA
-------------------------------------------------------------------------

Ate 09/10/2026 a independencia entre papeis era medida por tile de 1 grau e
componente evento-tile. Tile nao e bacia: dois tiles diferentes podem estar no
mesmo rio, um a montante do outro, e um ID de bacia diferente tambem nao
garante independencia -- sub-bacias do mesmo sistema fluvial dividem a agua que
passa por elas. O BR03 usava HydroBASINS pelo Earth Engine; o disco nao tinha.

-------------------------------------------------------------------------
A FONTE (conferida em 09/10/2026)
-------------------------------------------------------------------------

  arquivo   hybas_sa_lev01-12_v1c.zip (334.160.720 bytes), formato padrao
            (sem lagos inseridos: a topologia do formato com lagos e
            declaradamente inconsistente em lagoas e reservatorios, TechDoc 2.3)
  origem    https://data.hydrosheds.org/file/hydrobasins/standard/
  campos    HYBAS_ID, NEXT_DOWN (0 = sem jusante), NEXT_SINK, MAIN_BAS (exutorio do
            sistema fluvial), DIST_SINK, DIST_MAIN, SUB_AREA, UP_AREA, PFAF_ID,
            ENDO (2 = sumidouro endorreico; NEXT_DOWN > 0 com ENDO = 2 e ligacao
            virtual), COAST (bacias costeiras aglomeradas), ORDER, SORT
  licenca   HydroSHEDS v1 License Agreement (Apendice A do TechDoc v1.4): uso e
            obra derivada livres, comercial ou nao; o dado nao pode ser
            distribuido como produto avulso; atribuicao do Exhibit B obrigatoria
  citacao   Lehner & Grill 2013, Hydrol. Process. 27(15): 2171-2186
  derivado  HydroSHEDS 15" (~500 m), de SRTM: e OUTRA drenagem que a do ter-01
            (FABDEM 30 m); a fronteira de bacia pode passar a centenas de metros
            da divisoria que o HAND do ter-01 enxerga

-------------------------------------------------------------------------
O QUE ESTE MODULO FAZ
-------------------------------------------------------------------------

  verificar  tamanho conferido com o servidor, CRC de todos os membros, SHA-256
  converter  cada nivel vira parquet (atributos + WKB + caixa) com manifesto;
             topologia conferida: NEXT_DOWN aponta para ID existente, sem ciclo
  localizar  ponto -> bacia em varios niveis; 0 poligonos = fora (mar ou fora da
             America do Sul), > 1 = ambiguo (borda); nunca escolhe por conveniencia
  relacao    mesma bacia, montante, jusante, mesmo sistema fluvial (MAIN_BAS) ou
             desconectadas pela superficie; ligacao virtual endorreica marcada

nao_e: nao e mancha de inundacao nem delimitacao a 30 m; nao muda particao.

Uso:
    python scripts/geo/bacias_hydrobasins.py --verificar
    python scripts/geo/bacias_hydrobasins.py --converter
    python scripts/geo/bacias_hydrobasins.py --ponto -51.07 -12.07
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
RAIZ = RUNS / "hydrobasins" / "v1c"
ZIP = RAIZ / "bruto" / "hybas_sa_lev01-12_v1c.zip"
TABELAS = RAIZ / "tabelas"
URL = "https://data.hydrosheds.org/file/hydrobasins/standard/hybas_sa_lev01-12_v1c.zip"
BYTES_DO_SERVIDOR = 334_160_720
# Regioes adquiridas. Cada uma com seu zip, tamanho conferido no servidor e tabelas proprias; IDs nao colidem
# (o primeiro digito do HYBAS_ID e a regiao). Europa entrou em 10/10/2026 pelas reservas EMSR868/EMSR871.
REGIOES = {
    "sa": {"zip": ZIP, "tabelas": TABELAS, "bytes": BYTES_DO_SERVIDOR},
    "eu": {"zip": RAIZ / "bruto_eu" / "hybas_eu_lev01-12_v1c.zip", "tabelas": RAIZ / "tabelas_eu",
           "bytes": 360_873_795},
}
NIVEIS = (4, 5, 6, 7, 8, 10, 12)
CAMPOS = ("HYBAS_ID", "NEXT_DOWN", "NEXT_SINK", "MAIN_BAS", "DIST_SINK", "DIST_MAIN", "SUB_AREA", "UP_AREA",
          "PFAF_ID", "ENDO", "COAST", "ORDER", "SORT")
FONTE = {
    "produto": "HydroBASINS v1c, formato padrao (sem lagos), America do Sul",
    "url": URL, "tecdoc": "https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf",
    "licenca": ("HydroSHEDS v1 License Agreement (Apendice A, TechDoc v1.4, abr/2022): uso, reproducao e obra "
                "derivada livres para fins comerciais e nao comerciais; proibido distribuir como produto avulso; "
                "atribuicao do Exhibit B obrigatoria; aceite pelo uso"),
    "atribuicao": ("This product incorporates data from the HydroSHEDS version 1 database which is (c) World "
                   "Wildlife Fund, Inc. (2006-2022) and has been used herein under license. WWF has not evaluated the "
                   "data as altered and incorporated within, and therefore gives no warranty regarding its accuracy, "
                   "completeness, currency or suitability for any particular purpose. Portions of the HydroSHEDS v1 "
                   "database incorporate data which are the intellectual property rights of (c) USGS (2006-2008), NASA "
                   "(2000-2005), ESRI (1992-1998), CIAT (2004-2006), UNEP-WCMC (1993), WWF (2004), Commonwealth of "
                   "Australia (2007), and Her Royal Majesty and the British Crown and are used under license. The "
                   "HydroSHEDS v1 database and more information are available at https://www.hydrosheds.org."),
    "citacao": "Lehner, B., Grill, G. (2013). Hydrological Processes 27(15): 2171-2186. doi:10.1002/hyp.9740",
    "resolucao": "derivado do HydroSHEDS 15 arc-seg (~500 m), SRTM; subdivisao em confluencias >= 100 km2",
    "uso_no_revp": "unidade hidrologica e conectividade para independencia e contexto; nao e rotulo",
}


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return "sha256:" + h.hexdigest()


def _sha_json(o) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(o, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                                                 default=str).encode("utf-8")).hexdigest()


def verificar(zip_: Path = ZIP, bytes_esperados: int = BYTES_DO_SERVIDOR) -> dict:
    """Integridade antes de abrir qualquer membro: tamanho do servidor, CRC de cada membro, hash local."""
    if not zip_.exists():
        return {"estado": "ausente", "arquivo": str(zip_)}
    tamanho = zip_.stat().st_size
    with zipfile.ZipFile(zip_) as z:
        ruim = z.testzip()
        membros = [(i.filename, i.file_size) for i in z.infolist()]
    shp = sorted(n for n, _ in membros if n.lower().endswith(".shp"))
    perigosos = [n for n, _ in membros if n.startswith(("/", "\\")) or ".." in Path(n).parts]
    return {"estado": "integro" if (ruim is None and tamanho == bytes_esperados and not perigosos) else "falhou",
            "bytes": tamanho, "bytes_do_servidor": bytes_esperados, "crc_primeiro_membro_ruim": ruim,
            "membros": len(membros), "shapefiles": shp, "caminhos_perigosos": perigosos, "sha256": _sha(zip_)}


def _membro(nivel: int, regiao: str = "sa") -> str:
    return f"hybas_{regiao}_lev{nivel:02d}_v1c.shp"


def topologia(t: pd.DataFrame) -> dict:
    """NEXT_DOWN aponta para ID existente (ou 0)? Ha ciclo? Quantas ligacoes virtuais endorreicas?"""
    ids = set(t.HYBAS_ID)
    orfaos = int((~t.NEXT_DOWN.isin(ids) & (t.NEXT_DOWN != 0)).sum())
    prox = dict(zip(t.HYBAS_ID, t.NEXT_DOWN))
    ciclos = 0
    for i in t.HYBAS_ID:
        visto, x = set(), i
        while x and x in prox:
            if x in visto:
                ciclos += 1
                break
            visto.add(x)
            x = prox[x]
    return {"poligonos": int(len(t)), "next_down_sem_destino": orfaos, "ciclos": ciclos,
            "exutorios": int((t.NEXT_DOWN == 0).sum()), "ligacoes_virtuais_endorreicas": int(((t.ENDO == 2)
                                                                                              & (t.NEXT_DOWN > 0)).sum()),
            "main_bas_inexistente": int((~t.MAIN_BAS.isin(ids)).sum()),
            "sub_area_km2": {"mediana": float(t.SUB_AREA.median()), "p95": float(t.SUB_AREA.quantile(0.95))}}


def converter(niveis=NIVEIS, zip_: Path = ZIP, saida: Path = TABELAS, regiao: str = "sa") -> dict:
    import fiona
    import shapely
    from shapely.geometry import shape
    if regiao != "sa" or zip_ != ZIP:
        r = REGIOES[regiao]
        zip_, saida = r["zip"], r["tabelas"]
    ver = verificar(zip_, REGIOES[regiao]["bytes"])
    if ver["estado"] != "integro":
        raise RuntimeError(f"zip nao passou na verificacao: {ver}")
    saida.mkdir(parents=True, exist_ok=True)
    registro = {}
    for nivel in (1, *niveis):
        arq = saida / f"nivel_{nivel:02d}.parquet"
        caminho = f"zip://{zip_.as_posix()}!{_membro(nivel, regiao)}"
        linhas, geoms = [], []
        with fiona.open(caminho) as src:
            crs = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
            for f in src:
                p = dict(f["properties"])
                linhas.append({c: p.get(c) for c in CAMPOS})
                geoms.append(shape(f["geometry"]))
        t = pd.DataFrame(linhas)
        g = np.array(geoms, dtype=object)
        validos = shapely.is_valid(g)
        b = shapely.bounds(g)
        t["lon_min"], t["lat_min"], t["lon_max"], t["lat_max"] = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        t["geometria_valida"] = validos
        t["wkb"] = shapely.to_wkb(g)
        t.to_parquet(arq.with_suffix(".part"), index=False)
        arq.with_suffix(".part").replace(arq)
        registro[nivel] = {"arquivo": arq.name, "sha256": _sha(arq), "crs": crs, "topologia": topologia(t),
                           "geometrias_invalidas": int((~validos).sum())}
        print(nivel, registro[nivel]["topologia"]["poligonos"], flush=True)
    man = {"tipo": "hydrobasins_tabelas", "regiao": regiao, "fonte": {**FONTE, "url": FONTE["url"].replace("_sa_", f"_{regiao}_")},
           "zip": ver, "niveis": registro,
           "codigo_sha256": _sha(Path(__file__)), "quando_utc": _agora(),
           "nao_e": "nao redistribui o dado bruto; tabelas locais derivadas para consulta"}
    man["hash_resumo"] = _sha_json({k: v for k, v in man.items() if k != "quando_utc"})
    (saida / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def adicionar_nivel(regiao: str, nivel: int) -> dict:
    """Acrescenta um nivel as tabelas de uma regiao sem reescrever os ja convertidos (que outras versoes citam)."""
    import fiona
    import shapely
    from shapely.geometry import shape
    r = REGIOES[regiao]
    man_arq = r["tabelas"] / "manifesto.json"
    man = json.loads(man_arq.read_text(encoding="utf-8"))
    if str(nivel) in {str(k) for k in man["niveis"]}:
        return man
    linhas, geoms = [], []
    with fiona.open(f"zip://{r['zip'].as_posix()}!{_membro(nivel, regiao)}") as src:
        for f in src:
            linhas.append({c: dict(f["properties"]).get(c) for c in CAMPOS})
            geoms.append(shape(f["geometry"]))
    t = pd.DataFrame(linhas)
    g = np.array(geoms, dtype=object)
    b = shapely.bounds(g)
    t["lon_min"], t["lat_min"], t["lon_max"], t["lat_max"] = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    t["geometria_valida"] = shapely.is_valid(g)
    t["wkb"] = shapely.to_wkb(g)
    arq = r["tabelas"] / f"nivel_{nivel:02d}.parquet"
    t.to_parquet(arq, index=False)
    man.setdefault("niveis_acrescentados", {})[str(nivel)] = {
        "arquivo": arq.name, "sha256": _sha(arq), "topologia": topologia(t), "quando_utc": _agora(),
        "por_que": "contorno continental para separar fora da cobertura de fora de toda bacia"}
    man_arq.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return man


def _agora() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class Bacias:
    """Niveis carregados sob demanda, de todas as regioes convertidas; localizacao vetorizada por STRtree.

    `tabelas` explicito (um diretorio) mantem o uso antigo e os testes. Sem ele, carrega toda regiao de
    REGIOES que tenha manifesto -- e um ponto fora da caixa de todas elas fica `fora_da_cobertura`, que NAO e
    `fora_de_toda_bacia` (mar ou lacuna dentro de uma regiao coberta).
    """

    def __init__(self, tabelas: Path | None = None):
        dirs = [Path(tabelas)] if tabelas else [r["tabelas"] for r in REGIOES.values()
                                                 if (r["tabelas"] / "manifesto.json").exists()]
        self.dirs = dirs
        mans = [json.loads((d / "manifesto.json").read_text(encoding="utf-8")) for d in dirs]
        self.tabelas = dirs[0]
        self.man = mans[0] if len(mans) == 1 else {
            "hash_resumo": _sha_json(sorted(m["hash_resumo"] for m in mans)),
            "regioes": {m.get("regiao", "sa"): m["hash_resumo"] for m in mans}}
        self.caixas = []
        for d in dirs:
            n1 = d / "nivel_01.parquet"
            if n1.exists():
                c = pd.read_parquet(n1, columns=["lon_min", "lat_min", "lon_max", "lat_max"])
                self.caixas.append((c.lon_min.min(), c.lat_min.min(), c.lon_max.max(), c.lat_max.max()))
        self._nivel: dict = {}

    def nivel(self, n: int):
        if n not in self._nivel:
            import shapely
            t = pd.concat([pd.read_parquet(d / f"nivel_{n:02d}.parquet") for d in self.dirs
                           if (d / f"nivel_{n:02d}.parquet").exists()], ignore_index=True)
            g = shapely.from_wkb(t.wkb.to_numpy())
            self._nivel[n] = (t.drop(columns="wkb"), g, shapely.STRtree(g))
        return self._nivel[n]

    def coberto(self, lon, lat) -> np.ndarray:
        lon, lat = np.asarray(lon, float), np.asarray(lat, float)
        if not self.caixas:
            return np.ones(len(lon), bool)
        return np.any([(lon >= c[0]) & (lon <= c[2]) & (lat >= c[1]) & (lat <= c[3]) for c in self.caixas], axis=0)

    def localizar(self, lon, lat, n: int) -> pd.DataFrame:
        """HYBAS_ID por ponto. `n_poligonos` 0 = fora de toda bacia (mar/fora da regiao); > 1 = ambiguo."""
        import shapely
        t, g, arvore = self.nivel(n)
        pts = shapely.points(np.asarray(lon, float), np.asarray(lat, float))
        par = arvore.query(pts, predicate="intersects")
        cont = np.bincount(par[0], minlength=len(pts))
        primeiro = np.full(len(pts), -1)
        # para o ambiguo guarda o menor HYBAS_ID (deterministico), mas o estado fica "ambiguo"
        ordem = np.lexsort((t.HYBAS_ID.to_numpy()[par[1]], par[0]))
        p0, p1 = par[0][ordem], par[1][ordem]
        prim = np.r_[True, p0[1:] != p0[:-1]] if len(p0) else np.array([], bool)
        primeiro[p0[prim]] = p1[prim]
        ok = primeiro >= 0
        out = pd.DataFrame({"n_poligonos": cont})
        for c in ("HYBAS_ID", "MAIN_BAS", "NEXT_DOWN", "ENDO", "COAST", "SUB_AREA", "UP_AREA", "PFAF_ID"):
            v = np.zeros(len(pts), dtype=t[c].dtype) if t[c].dtype.kind in "iu" else np.full(len(pts), np.nan)
            v[ok] = t[c].to_numpy()[primeiro[ok]]
            out[c] = v
        cob = self.coberto(lon, lat)
        out["estado_bacia"] = np.select([(cont == 0) & ~cob, cont == 0, cont == 1],
                                        ["fora_da_cobertura", "fora_de_toda_bacia", "unica"], "ambigua_na_borda")
        out.loc[cont == 0, ["HYBAS_ID", "MAIN_BAS"]] = 0
        return out

    def jusante(self, n: int, hybas_id: int, limite: int = 10000) -> list[int]:
        t, _, _ = self.nivel(n)
        prox = dict(zip(t.HYBAS_ID, t.NEXT_DOWN))
        cadeia, x = [], hybas_id
        while x and x in prox and len(cadeia) < limite:
            cadeia.append(int(x))
            x = prox[x]
        return cadeia

    def relacao(self, n: int, a: int, b: int) -> str:
        """Como duas bacias do mesmo nivel se ligam pela superficie."""
        if a == b:
            return "mesma_bacia"
        if not a or not b:
            return "fora_de_toda_bacia"
        if b in self.jusante(n, a)[1:]:
            return "a_montante_de_b"
        if a in self.jusante(n, b)[1:]:
            return "b_montante_de_a"
        t, _, _ = self.nivel(n)
        mb = dict(zip(t.HYBAS_ID, t.MAIN_BAS))
        if mb.get(a) == mb.get(b):
            return "mesmo_sistema_fluvial"
        return "desconectadas_pela_superficie"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verificar", action="store_true")
    ap.add_argument("--converter", action="store_true")
    ap.add_argument("--regiao", default="sa", choices=sorted(REGIOES))
    ap.add_argument("--ponto", nargs=2, type=float)
    a = ap.parse_args()
    if a.verificar:
        r = REGIOES[a.regiao]
        print(json.dumps(verificar(r["zip"], r["bytes"]), ensure_ascii=False, indent=1))
    if a.converter:
        t0 = time.time()
        m = converter(regiao=a.regiao)
        print(json.dumps({n: r["topologia"] for n, r in m["niveis"].items()}, ensure_ascii=False))
        print(round(time.time() - t0), "s")
    if a.ponto:
        b = Bacias()
        for n in NIVEIS:
            print(n, b.localizar([a.ponto[0]], [a.ponto[1]], n).iloc[0].to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
