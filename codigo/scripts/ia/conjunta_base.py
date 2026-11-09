"""Base conjunta v1: pontos fisicos MT03/MT05 e chips SAR Sen1Floods11 no mesmo espaco de atributos.

Linhagem `aprendizado_conjunto_v1` (16/10/2026). Nao treina e nao le reserva.

Duas tarefas, um vocabulario fisico:
  fisica     pontos MT03 (43 componentes, 130 eventos); alvo = inundado/seco OBSERVADO
             no evento; escore avaliado como ordem DENTRO do evento. Nao e probabilidade
             territorial: a amostra e condicionada a ocorrencia do evento e a fonte.
  observacao chips Sen1Floods11 (rotulo manual sobre S1/S2); alvo = agua nao permanente
             na visada posterior (JRC permanente, sem leitura e sem rotulo = -1).

Atributos fisicos do encoder (calculaveis num lugar sem rotulo, do mesmo raster ter-01):
  f_hand      log1p(HAND D-inf)                     acumulacao (proxy)
  f_sca       clip(TWI + ln tan declive, -5, 25)    concentracao (proxy = ln SCA; truncada no recorte)
  f_declive   log1p(declive)                        conexao/gradiente (proxy)
  f_dist_dren log1p(dist. a drenagem D-inf)         conexao (proxy)
  f_dist_canal log1p(dist. a canal >= 10 km2)       conexao ao canal de porte (proxy)
  z_relevo    log1p(declive mediano do raster da unidade)  contexto sem rotulo
Fora do encoder: f_construido e f_areia entram so na cabeca fisica (nao existem por pixel
no acervo dos chips). Fora de tudo: elevacao relativa a amostra/chip (depende do recorte),
SAR anterior (mediana anual, nao e par datado) e qualquer dado posterior na consulta fisica.

Distancias sao calculadas na grade nativa do terreno exatamente como em `mt02_atributos`
(EDT sobre a unidade inteira) e so depois reprojetadas (bilinear, NaN fora) para a grade
do SAR posterior. Pixel sem terreno vira -1 no alvo de observacao em TODOS os bracos.

Uso:
    python scripts/ia/conjunta_base.py montar
    python scripts/ia/conjunta_base.py verificar
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "sar", "neural", "servico", "manutencao"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

VERSAO = "aprendizado_conjunto_v1"
SAIDA = REPO / "local_runs" / "aprendizado-conjunto" / "v1"
BASE = SAIDA / "base"
CAMPANHA_SAR = REPO / "local_runs" / "sar-independencia" / "campanha_2026_10_06"
TER = REPO / "local_runs" / "ter-01-cadeia-harmonizada"
CONTEXTO = REPO / "local_runs" / "mecanismos-transferiveis" / "v1" / "contexto_unidades.parquet"
TERRENO = ("f_hand", "f_sca", "f_declive", "f_dist_dren", "f_dist_canal", "z_relevo")
COBERTURA = ("f_construido", "f_areia")
SEM_DADO = -32768.0
CANAL_KM2 = 10.0
FORMA = (512, 512)


def caracteristicas_pontos(d: pd.DataFrame) -> pd.DataFrame:
    """Mesmas transformacoes do MT05 (`mt05_transferencia_sem_amostra.caracteristicas`)."""
    x = pd.DataFrame(index=d.index)
    x["f_hand"] = np.log1p(d.hand_m.clip(lower=0))
    x["f_sca"] = d.ln_sca.clip(-5, 25)
    x["f_declive"] = np.log1p(d.slope_deg.clip(lower=0))
    x["f_dist_dren"] = np.log1p(d.dist_drenagem_m)
    x["f_dist_canal"] = np.log1p(d.dist_canal10_m)
    x["z_relevo"] = np.log1p(d.declive_p50)
    x["f_construido"] = d.construido_v2
    x["f_areia"] = d.areia_v2 / 100
    return x


def terreno_nativo(pasta: Path, declive_p50: float) -> tuple[dict, dict]:
    """Atributos na grade nativa do ter-01; distancias como `mt02_atributos.distancias_da_unidade`."""
    import rasterio
    from scipy import ndimage

    def ler(nome):
        with rasterio.open(pasta / f"{nome}.tif") as f:
            a = f.read(1).astype(np.float64)
            a[(a == SEM_DADO) | ~np.isfinite(a)] = np.nan
            return a, f.transform, f.crs, f.res[0]

    hand, tr, crs, res = ler("hand_dinf")
    twi, *_ = ler("twi_dinf")
    slope, *_ = ler("slope_deg_wbt")
    # ln SCA como `mt01_populacao.derivar`: TWI + ln tan(declive com piso 0,01 grau), nao ln(dinf_sca)
    ln_sca = twi + np.log(np.tan(np.radians(np.clip(slope, 0.01, None))))
    with rasterio.open(pasta / "streams_dinf.tif") as s:
        canal = s.read(1) == 1
    with rasterio.open(pasta / "dinf_accum_cells.tif") as s:
        grande = s.read(1) * res * res / 1e6 >= CANAL_KM2
    dist = []
    for mascara in (canal, grande):
        dist.append(ndimage.distance_transform_edt(~mascara) * res if mascara.any()
                    else np.full(hand.shape, np.nan))
    with np.errstate(divide="ignore", invalid="ignore"):
        f = {"f_hand": np.log1p(np.clip(hand, 0, None)),
             "f_sca": np.clip(ln_sca, -5, 25),
             "f_declive": np.log1p(np.clip(slope, 0, None)),
             "f_dist_dren": np.log1p(dist[0]),
             "f_dist_canal": np.log1p(dist[1]),
             "z_relevo": np.full(hand.shape, np.log1p(declive_p50))}
    meta = {"transform": tr, "crs": crs, "res_m": float(res), "forma": list(hand.shape),
            "tem_canal_10km2": bool(grande.any()), "tem_drenagem": bool(canal.any())}
    return f, meta


def reprojetar(f: dict, meta: dict, destino: dict) -> np.ndarray:
    from rasterio.warp import Resampling, reproject
    out = np.full((len(TERRENO),) + FORMA, np.nan, np.float32)
    for i, k in enumerate(TERRENO):
        reproject(f[k].astype(np.float32), out[i], src_transform=meta["transform"], src_crs=meta["crs"],
                  dst_transform=destino["transform"], dst_crs=destino["crs"],
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
    return out


def pasta_terreno(nome: str) -> Path | None:
    for d in (TER / f"n1chip_{nome}_LabelHand", TER / f"n1chip_{nome}"):
        if d.exists():
            return d
    return None


def carregar_pontos() -> pd.DataFrame:
    import mt03_experimento as e3
    d = e3.carregar()
    x = caracteristicas_pontos(d)
    cols = ["ponto_id", "unidade", "evento_id", "fonte", "classe", "papel", "processo_ponto", "lat", "lon",
            "unidade_terreno", "familia_mdt", "resolucao_nativa_m", "precisao_data", "janela_inicio", "janela_fim",
            "hand_m"]
    return pd.concat([d[cols].reset_index(drop=True), x.reset_index(drop=True)], axis=1)


def montar() -> dict:
    t0 = time.time()
    BASE.mkdir(parents=True, exist_ok=True)
    pts = carregar_pontos()
    pts.to_parquet(BASE / "pontos.parquet", index=False)
    print("pontos", pts.shape, round(time.time() - t0), flush=True)

    inv = json.loads((CAMPANHA_SAR / "inventario.json").read_text(encoding="utf-8"))
    cache = CAMPANHA_SAR / "cache"
    X = np.load(cache / "x.npy", mmap_mode="r")
    S = np.load(cache / "sensor.npy", mmap_mode="r")
    Y = np.load(cache / "y.npy", mmap_mode="r")
    NPV = np.load(cache / "npv.npy", mmap_mode="r")
    ctx = pd.read_parquet(CONTEXTO).set_index("unidade_terreno")
    ev_unid = pts[pts.fonte == "sen1floods11"].groupby("evento_id").unidade.agg(lambda s: sorted(set(s)))
    pares = inv["pares"]
    n = len(pares)
    ter = np.lib.format.open_memmap(BASE / "chips_terreno.npy", mode="w+", dtype=np.float16,
                                    shape=(n, len(TERRENO)) + FORMA)
    sar = np.lib.format.open_memmap(BASE / "chips_sar.npy", mode="w+", dtype=np.float16, shape=(n, 2) + FORMA)
    alvo = np.lib.format.open_memmap(BASE / "chips_alvo.npy", mode="w+", dtype=np.int8, shape=(n,) + FORMA)
    from rasterio import Affine
    from fontes_sar import normalizar_fixo
    linhas = []
    for i, r in enumerate(pares):
        evento_mt = f"S1F11:{r['local']}"
        un = ev_unid.get(evento_mt)
        if un is None or len(un) != 1:
            raise ValueError(f"evento SAR sem componente unico: {evento_mt} {un}")
        g = r["grade"]
        dest = {"transform": Affine(*g["transform"][:6]), "crs": g["crs"]}
        pasta = pasta_terreno(r["nome"])
        sensor = np.asarray(S[i])
        y = np.asarray(Y[i]).astype(np.int8)
        alvo_obs = y.copy()
        alvo_obs[~(np.asarray(NPV[i]) & sensor)] = -1  # mesma regra de ci.alvo
        if pasta is None:
            t = np.full((len(TERRENO),) + FORMA, np.nan, np.float32)
            meta = {"res_m": None, "tem_canal_10km2": None, "tem_drenagem": None}
            unid_ter, fam = None, None
        else:
            unid_ter = pasta.name
            fam = str(ctx.loc[unid_ter, "familia_mdt"]) if unid_ter in ctx.index else None
            p50 = float(ctx.loc[unid_ter, "declive_p50"]) if unid_ter in ctx.index else np.nan
            f, meta = terreno_nativo(pasta, p50)
            t = reprojetar(f, meta, dest)
        valido_ter = np.isfinite(t[[0, 1, 2, 3]]).all(0)
        alvo_obs[~valido_ter] = -1
        z = normalizar_fixo(np.asarray(X[i, :2], np.float32))
        z[:, ~sensor] = 0
        ter[i], sar[i], alvo[i] = t.astype(np.float16), z.astype(np.float16), alvo_obs
        linhas.append({
            "indice": i, "nome": r["nome"], "evento_sar": r["evento"], "evento_mt": evento_mt, "componente": un[0],
            "pais": r["local"], "data_pos": r["data_pos_catalogo"], "instrumento": "Sentinel-1 GRD VV/VH posterior; rotulo manual Sen1Floods11",
            "resolucao_sar_m": 10.0, "resolucao_terreno_m": meta["res_m"], "familia_mdt": fam,
            "unidade_terreno": unid_ter, "tem_canal_10km2": meta["tem_canal_10km2"],
            "fracao_terreno_valido": float(valido_ter.mean()), "fracao_sensor_valido": float(sensor.mean()),
            "pixels_positivos": int((alvo_obs == 1).sum()), "pixels_negativos": int((alvo_obs == 0).sum()),
            "pixels_desconhecidos": int((alvo_obs == -1).sum()),
            "papel": "supervisao_observacao" if (alvo_obs != -1).any() else "sem_pixel_rotulado",
            "bounds": g["bounds"], "crs": g["crs"],
            "janela": "visada posterior unica do catalogo; anterior = mediana anual (nao usada)",
            "hashes_fonte": r["hashes"]})
        if i % 50 == 0:
            print(i, r["nome"], round(time.time() - t0), flush=True)
    for a in (ter, sar, alvo):
        a.flush()
    del ter, sar, alvo
    chips = pd.DataFrame(linhas)
    # correspondencia ponto <-> chip (mesmo rotulo manual de origem)
    s1 = pts[pts.fonte == "sen1floods11"]
    pc = []
    for r in linhas:
        x0, y0, x1, y1 = r["bounds"]
        m = (s1.lon >= x0) & (s1.lon <= x1) & (s1.lat >= y0) & (s1.lat <= y1)
        pc.append(int(m.sum()))
    chips["pontos_fisicos_no_chip"] = pc
    chips.drop(columns=["hashes_fonte"]).to_parquet(BASE / "chips.parquet", index=False)
    man = manifesto(pts, chips, inv)
    (BASE / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("fim", round(time.time() - t0), flush=True)
    return man["contagens"]


def manifesto(pts: pd.DataFrame, chips: pd.DataFrame, inv: dict) -> dict:
    ok = chips[chips.papel == "supervisao_observacao"]
    arquivos = ["pontos.parquet", "chips.parquet", "chips_terreno.npy", "chips_sar.npy", "chips_alvo.npy"]
    return {
        "versao": VERSAO, "quando_utc": agora_utc(),
        "tipo": "base_conjunta_fisica_observacao",
        "nao_e": "nao e reserva confirmatoria, nao e rotulo novo, nao e probabilidade territorial",
        "regime": "desenvolvimento retrospectivo: eventos e rotulos ja examinados no projeto",
        "fontes": {"pontos": "mt03_experimento.carregar (MT03-A/MT05, populacao comum filtrada por elevation_local_m historico)",
                   "chips": str(CAMPANHA_SAR / "inventario.json"), "terreno": str(TER),
                   "contexto": str(CONTEXTO),
                   "inventario_sha256": sha256_arquivo(CAMPANHA_SAR / "inventario.json"),
                   "cache_sar_manifesto_sha256": sha256_arquivo(CAMPANHA_SAR / "cache" / "cache.json")},
        "arquivos_sha256": {a: sha256_arquivo(BASE / a) for a in arquivos},
        "atributos_encoder": list(TERRENO), "atributos_so_fisica": list(COBERTURA),
        "excluidos": {"elevation_local_m": "calculada sobre a amostra rotulada do evento",
                      "elevacao_relativa_chip": "depende do recorte do chip",
                      "sar_anterior": "mediana anual Sen1Floods11; nao e par datado; nao usado",
                      "sar_na_fisica": "radar posterior nunca entra na cabeca fisica",
                      "variaveis_sociais": "fora do escore fisico"},
        "papeis": {"ponto_positivo": "positivo_episodico observado no evento",
                   "ponto_negativo": "negativo_observado no mesmo evento (nunca ausencia de registro)",
                   "pixel_-1": "desconhecido, agua permanente JRC, sem leitura SAR, sem rotulo ou sem terreno"},
        "alinhamento": "terreno ter-01 nativo (30 m nos chips) -> grade do SAR posterior (10 m), bilinear; distancias na grade nativa",
        "observabilidade": "Sen1Floods11 ve agua aberta em SAR/optico; CEMS/GFD/UFO tem instrumentos e quadros de negativo diferentes; nao equivalentes",
        "contagens": {
            "pontos": int(len(pts)), "pontos_eventos": int(pts.evento_id.nunique()),
            "pontos_componentes": int(pts.unidade.nunique()),
            "pontos_por_fonte": pts.groupby("fonte").agg(linhas=("ponto_id", "size"), eventos=("evento_id", "nunique"),
                                                         componentes=("unidade", "nunique")).to_dict("index"),
            "chips_inventario": int(len(inv["pares"])), "chips_com_pixel_rotulado": int(len(ok)),
            "chips_sem_terreno": int(chips.unidade_terreno.isna().sum()),
            "chips_eventos_sar": int(ok.evento_sar.nunique()), "chips_componentes": int(ok.componente.nunique()),
            "chips_por_componente": ok.groupby("componente").nome.size().to_dict(),
            "chips_por_evento": ok.groupby("evento_mt").nome.size().to_dict(),
            "pixels_positivos": int(ok.pixels_positivos.sum()), "pixels_negativos": int(ok.pixels_negativos.sum()),
            "chips_com_pontos_fisicos": int((chips.pontos_fisicos_no_chip > 0).sum()),
            "pontos_fisicos_em_chips": int(chips.pontos_fisicos_no_chip.sum()),
        },
        "unidade_inferencial": "componente de dependencia (43 na fisica; 10 eventos SAR em 10 componentes na observacao); pixel/chip/semente nao sao unidades",
        "regras": ["desconhecido nunca vira negativo", "sem negativo por ausencia de registro",
                   "sem reamostragem para aumentar amostra: recortes aleatorios sao aumento de treino, avaliacao no chip inteiro",
                   "predicao da rede nunca vira referencia", "anterior SAR nao e tratado como par"],
    }


def verificar(n_chips: int = 40) -> dict:
    """Confere o alinhamento: atributo do chip na posicao do ponto x valor do ponto (mesma origem)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import rowcol
    pts = pd.read_parquet(BASE / "pontos.parquet")
    chips = pd.read_parquet(BASE / "chips.parquet")
    ter = np.load(BASE / "chips_terreno.npy", mmap_mode="r")
    s1 = pts[pts.fonte == "sen1floods11"]
    ctx = pd.read_parquet(CONTEXTO).set_index("unidade_terreno")
    dif_nat, dif_sar, n = {k: [] for k in TERRENO}, {k: [] for k in TERRENO}, 0
    for _, c in chips[chips.unidade_terreno.notna()].head(n_chips).iterrows():
        sub = s1[s1.unidade_terreno == c.unidade_terreno]
        if sub.empty:
            continue
        f, meta = terreno_nativo(TER / c.unidade_terreno, float(ctx.loc[c.unidade_terreno, "declive_p50"]))
        px, py = Transformer.from_crs(4326, meta["crs"], always_xy=True).transform(sub.lon.to_numpy(), sub.lat.to_numpy())
        li, co = map(np.asarray, rowcol(meta["transform"], px, py))
        dentro = (li >= 0) & (li < meta["forma"][0]) & (co >= 0) & (co < meta["forma"][1])
        with rasterio.open(Path(json.loads((CAMPANHA_SAR / "inventario.json").read_text(encoding="utf-8"))
                                ["pares"][int(c.indice)]["caminhos"]["depois"])) as fs_:
            li2, co2 = map(np.asarray, rowcol(fs_.transform, sub.lon.to_numpy(), sub.lat.to_numpy()))
        dentro2 = (li2 >= 0) & (li2 < 512) & (co2 >= 0) & (co2 < 512)
        for j, k in enumerate(TERRENO):
            v = sub[k].to_numpy()
            a = f[k][li[dentro], co[dentro]]
            dif_nat[k] += list(np.abs(a - v[dentro])[np.isfinite(a) & np.isfinite(v[dentro])])
            b = np.asarray(ter[int(c.indice), j], np.float32)[li2[dentro2], co2[dentro2]]
            dif_sar[k] += list(np.abs(b - v[dentro2])[np.isfinite(b) & np.isfinite(v[dentro2])])
        n += int(dentro.sum())
    r = {"pontos_conferidos": n, "chips": n_chips,
         "nativo_max_abs": {k: float(np.max(v)) if v else None for k, v in dif_nat.items()},
         "nativo_frac_exata_1e-6": {k: float(np.mean(np.array(v) < 1e-6)) if v else None for k, v in dif_nat.items()},
         "grade_sar_mediana_abs": {k: float(np.median(v)) if v else None for k, v in dif_sar.items()},
         "grade_sar_p95_abs": {k: float(np.percentile(v, 95)) if v else None for k, v in dif_sar.items()},
         "leitura": "nativo deve ser exato (mesmo raster e mesma regra); grade SAR difere por reamostragem bilinear 30->10 m e float16"}
    (BASE / "verificacao_alinhamento.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["montar", "verificar"])
    a = ap.parse_args()
    print(json.dumps(montar() if a.acao == "montar" else verificar(), ensure_ascii=False, indent=1, default=str)[:6000])
