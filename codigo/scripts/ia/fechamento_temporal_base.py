"""Etapa B: quadro lugar x instante da pilha S1 com referencia so do passado.

Linhagem `fechamento_aprendizado_transferencia_v1` (17/10/2026). Nao treina.

Por bloco admitido em (b) da pilha (200 blocos, 26 componentes):
  pixels    ate 512 pixels observaveis nao permanentes (classe 1 da pilha), sorteio com semente
            por bloco, sem olhar rotulo nem deteccao
  atributos terreno ter-01 (os 6 do encoder), arvore/construido WorldCover, angulo de incidencia,
            JRC occurrence (so calibracao e qualidade)
  instante  data, UTC da passagem, janela (fora/transicao/evento), orbita e passagem fixas do bloco
  sinal     VV e VH em dB (NaN quando sem leitura: ausencia nunca vira zero)
  referencia P75 de VV do proprio pixel nas datas FORA estritamente anteriores (passado) e o numero
            delas; P75 retrospectivo (todas as datas fora) fica como sensibilidade declarada
  forcamento ERA5 0,25 grau via Open-Meteo: p72 (72 h anteriores) e ante3_14 ((T-14d, T-3d]),
            do quadro v2_era5; e reanalise (disponivel dias depois), nao previsao em tempo real
Calibracao: por bloco, ate 256 pixels JRC >= 90 (agua) e 256 JRC = 0 e max = 0 (terra) observaveis,
VV nas datas fora; o limiar e escolhido no TREINO de cada dobra, nunca no destino.

Niveis de alvo (contagem no manifesto):
  1 referencia independente contemporanea   nenhuma neste quadro
  2 anotacao admitida com limites            rotulos CEMS/GFD/Sen1/UFO dos pontos (tarefa fisica)
  3 pseudoalvo instrumental                  deteccao SAR por regra (este quadro)

Uso:
    python scripts/ia/fechamento_temporal_base.py montar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "sar", "servico", "metodo", "neural"):
    sys.path.insert(0, str(REPO / "scripts" / s))
import conjunta_base as cb  # noqa: E402
import conjunta_campanha as cc  # noqa: E402
import fechamento_diagnostico as fd  # noqa: E402
import pilha_s1_mesmo_lugar as ps  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

BASE = fd.SAIDA / "base_temporal"
PIXELS_POR_BLOCO = 512
CALIB_POR_CLASSE = 256
MIN_PASSADO = 10
SEMENTE = 20261017


def p75_passado(vv: np.ndarray, fora: np.ndarray, minimo: int = MIN_PASSADO) -> tuple[np.ndarray, np.ndarray]:
    """vv [datas, pixels] em ordem temporal. P75 das datas fora ESTRITAMENTE anteriores; NaN sem historia."""
    n, p = vv.shape
    ref = np.full((n, p), np.nan, np.float32)
    cont = np.zeros((n, p), np.int16)
    hist = []
    for i in range(n):
        if hist:
            h = np.stack(hist)
            k = np.isfinite(h).sum(0)
            cont[i] = k
            with np.errstate(all="ignore"):
                q = np.nanpercentile(h, 75, axis=0) if len(hist) else np.full(p, np.nan)
            ref[i] = np.where(k >= minimo, q, np.nan)
        if fora[i]:
            hist.append(vv[i])
    return ref, cont


def montar() -> dict:
    BASE.mkdir(parents=True, exist_ok=True)
    blocos = cc.ler(fd.PILHA / "blocos.json")["blocos"]
    der = pd.read_csv(fd.PILHA / "blocos_derivados.csv").set_index("bloco")
    quadro = pd.read_parquet(fd.QUADRO)
    ctx = pd.read_parquet(cb.CONTEXTO).set_index("unidade_terreno")
    usados = [b for b in blocos if b["bloco"] in der.index and bool(der.loc[b["bloco"], "em_b"])]
    linhas, calib, info = [], [], []
    for aoi in sorted({b["aoi"] for b in usados}):
        f, _ = cb.terreno_nativo(cb.TER / aoi, float(ctx.loc[aoi, "declive_p50"]))
        for b in [x for x in usados if x["aoi"] == aoi]:
            nome = b["bloco"]
            with np.load(fd.PILHA / "pilhas" / f"{nome}.npz", allow_pickle=True) as z:
                vv, vh = z["vv"].astype(np.float32), z["vh"].astype(np.float32)
                datas_p = [str(x) for x in z["datas"]]
                arvore, constr, jocc, jmax, ang = (z[k].astype(np.float32) for k in ("arvore", "construido", "jrc_occ", "jrc_max", "angulo"))
            with np.load(fd.PILHA / "frequencia" / f"{nome}.npz", allow_pickle=True) as z:
                classe = z["classe"]
                datas = [str(x) for x in z["datas"]]; janela = [str(x) for x in z["janela"]]; utc = [str(x) for x in z["quando_utc"]]
            idx = [datas_p.index(d) for d in datas]
            vv, vh = vv[idx], vh[idx]
            for a in (vv, vh):
                a[a == ps.NODATA] = np.nan
                a /= 100.
            ordem = np.argsort(pd.to_datetime(utc).values)
            vv, vh = vv[ordem], vh[ordem]
            datas, janela, utc = [datas[i] for i in ordem], [janela[i] for i in ordem], [utc[i] for i in ordem]
            fora = np.array([j == "fora" for j in janela])
            sl = np.s_[b["linha0"]:b["linha0"] + b["altura"], b["coluna0"]:b["coluna0"] + b["largura"]]
            ter = np.stack([f[k][sl] for k in cb.TERRENO], -1).reshape(-1, len(cb.TERRENO))
            leitura = np.isfinite(vv.reshape(len(datas), -1)).mean(0)
            ok = (classe.ravel() == 1) & (leitura >= .8) & np.isfinite(ter[:, :4]).all(1)
            rng = np.random.default_rng([SEMENTE, blocos.index(b)])
            cand = np.flatnonzero(ok)
            pix = np.sort(rng.choice(cand, min(PIXELS_POR_BLOCO, len(cand)), replace=False)) if len(cand) else cand
            v = vv.reshape(len(datas), -1)[:, pix]; h = vh.reshape(len(datas), -1)[:, pix]
            refp, npass = p75_passado(v, fora)
            with np.errstate(all="ignore"):
                refr = np.nanpercentile(v[fora], 75, axis=0) if fora.sum() else np.full(len(pix), np.nan)
            q = quadro[quadro.bloco == nome].set_index("data").reindex(datas)
            nd, npx = len(datas), len(pix)
            linhas.append(pd.DataFrame({
                "bloco": nome, "componente": b["componente"], "aoi": aoi,
                "pixel": np.tile(pix, nd).astype(np.int32), "data": np.repeat(datas, npx),
                "passagem_utc": np.repeat(utc, npx), "janela": np.repeat(janela, npx),
                "vv_db": v.ravel(), "vh_db": h.ravel(), "p75_passado_db": refp.ravel(), "n_fora_passado": npass.ravel(),
                "p75_retro_db": np.tile(refr, nd).astype(np.float32),
                "p72": np.repeat(q.p72.to_numpy(float), npx), "ante3_14": np.repeat(q.ante3_14.to_numpy(float), npx),
                **{k: np.tile(ter[pix, i], nd) for i, k in enumerate(cb.TERRENO)},
                "arvore": np.tile(arvore.ravel()[pix], nd), "construido": np.tile(constr.ravel()[pix], nd),
                "angulo": np.tile(ang.ravel()[pix], nd), "jrc_occ": np.tile(jocc.ravel()[pix], nd)}))
            # amostra de calibracao (agua JRC >= 90 e terra JRC = 0 e max = 0), so datas fora
            obs = ps.observabilidade(arvore, constr, np.expm1(f["f_declive"][sl]), jocc)  # declive em graus
            for tipo, m in (("agua", obs["valido"] & (jocc >= 90)), ("terra", obs["observavel"] & (jocc == 0) & (jmax == 0))):
                c = np.flatnonzero(m.ravel())
                if not len(c):
                    continue
                c = np.sort(rng.choice(c, min(CALIB_POR_CLASSE, len(c)), replace=False))
                vf = vv.reshape(len(datas), -1)[fora][:, c]
                calib.append(pd.DataFrame({"bloco": nome, "componente": b["componente"], "tipo": tipo,
                                           "vv_db": vf.ravel()}).dropna())
            info.append({"bloco": nome, "componente": b["componente"], "pixels": int(npx), "datas": nd,
                         "datas_fora": int(fora.sum()), "pixels_elegiveis": int(len(cand))})
        print(aoi, len(info), flush=True)
    t = pd.concat(linhas, ignore_index=True)
    for k in ("vv_db", "vh_db", "p75_passado_db", "p75_retro_db", "arvore", "construido", "angulo", "jrc_occ", *cb.TERRENO):
        t[k] = t[k].astype(np.float32)
    t.to_parquet(BASE / "lugar_instante.parquet", index=False)
    cal = pd.concat(calib, ignore_index=True)
    cal.to_parquet(BASE / "calibracao_jrc.parquet", index=False)
    man = {"tipo": "quadro_lugar_x_instante_pilha_S1", "versao": "fechamento_aprendizado_transferencia_v1", "quando_utc": agora_utc(),
           "nao_e": "nao e inundacao validada nem referencia independente; deteccao SAR e pseudoalvo instrumental",
           "linhas": len(t), "blocos": int(t.bloco.nunique()), "componentes": int(t.componente.nunique()),
           "pixels": int(t.groupby("bloco").pixel.nunique().sum()), "datas_bloco": int(t.groupby(["bloco", "data"]).ngroups),
           "vv_ausente": int(t.vv_db.isna().sum()),
           "referencia_passado_ausente_por_falta_de_historia": int(t.p75_passado_db.isna().sum()),
           "forcamento_ausente": int(t.p72.isna().sum()),
           "niveis_de_alvo": {"1_referencia_independente_contemporanea": 0,
                              "2_anotacao_admitida": "rotulos dos pontos fisicos (510.348 pontos; tarefa fisica)",
                              "3_pseudoalvo_instrumental": int(t.vv_db.notna().sum())},
           "calibracao": cal.groupby(["componente", "tipo"]).size().unstack(fill_value=0).to_dict("index"),
           "regra_passado": f"P75 de VV nas datas fora estritamente anteriores; minimo {MIN_PASSADO}; senao recusa",
           "forcamento": "ERA5 0,25 grau (Open-Meteo); reanalise, nao disponivel em tempo real",
           "blocos_info": info,
           "fontes": {str(p.relative_to(REPO)): sha256_arquivo(p) for p in (
               fd.PILHA / "blocos.json", fd.PILHA / "blocos_derivados.csv", fd.QUADRO, cb.CONTEXTO)},
           "arquivos": {k: sha256_arquivo(BASE / k) for k in ("lugar_instante.parquet", "calibracao_jrc.parquet")}}
    cc.escrever(BASE / "manifesto.json", man)
    return {k: man[k] for k in ("linhas", "blocos", "componentes", "pixels", "datas_bloco", "vv_ausente",
                                "referencia_passado_ausente_por_falta_de_historia", "forcamento_ausente")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["montar"])
    print(json.dumps(montar(), indent=1))
