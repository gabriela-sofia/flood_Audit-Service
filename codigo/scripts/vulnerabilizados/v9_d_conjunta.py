"""v9 Frentes C (multimodal) e D (aprendizagem conjunta): encoder de terreno compartilhado entre
  S  suscetibilidade antecedente: patches de terreno 32x32 dos pontos externos fora_BR, logito = Q + cabeca (sem radar,
     sem pos-evento, sem mascara de inundacao -- nada do rotulo entra na inferencia)
  O  observacao de agua visivel: chips Sen1Floods11 (splits oficiais treino/validacao/teste; Bolivia trancado) a 30 m,
     radar S1 co-evento (VV, VH) + terreno, com bandeira de presenca da modalidade e modality dropout (p = 0,3);
     rotulo LabelHand (anotacao manual) reamostrado por media para 30 m (agua se >= 0,5 dos pixels validos)
preparar   chips -> npz (terreno no contrato do encoder; radar normalizado pelo treino; rotulo; validade)
observacao C-multimodal: O_fusao (radar+terreno, dropout), O_radar (terreno zerado), O_terreno (radar ausente);
           avaliado em teste com radar presente e ausente (AUC/IoU por pixel)
conjunta   D: S_sep (so S), O_sep (so O), stopgrad (encoder treinado por S; O sobre atributos destacados),
           conjunta (S + O no mesmo encoder), quebrado (conjunta com rotulos O de OUTRO chip), mesmo numero de passos;
           cosseno entre gradientes de S e O no encoder registrado; 3 sementes; S avaliada nas coortes (subamostra C)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_c_patches import CAMADAS, LOG, D as DC, encoder, normalizacao, prever as prever_s, Modelo as ModeloS
from v9_comum import COORTES, REPO, TER, V9, agora, auc, congelar9, etapa, g9

D = V9 / "d_conjunta"
CH = D / "chips"
for p in ("sar", "terreno"):
    sys.path.insert(0, str(REPO / "scripts" / p))
SPLITS = ("treino", "validacao", "teste")
RECORTE = 64


def preparar():
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    import dados_sen1floods11 as s11
    import substrato_multimodal as sub
    media, desvio = normalizacao()
    CH.mkdir(parents=True, exist_ok=True)
    meta = []
    for sp in SPLITS:
        for nome in s11.nomes_do_split(sp):
            alvo = CH / f"{nome}.npz"
            pasta = TER / f"n1chip_{nome}_LabelHand"
            if not (pasta / "dem_filled.tif").exists():
                meta.append({"chip": nome, "split": sp, "estado": "sem_terreno"})
                continue
            if not alvo.exists():
                x, val = sub.carregar(pasta.name)
                radar, rv = x[:2], val[:2].all(0)
                pil = []
                for k, c in enumerate(CAMADAS):
                    with rasterio.open(pasta / f"{c}.tif") as s:
                        a = s.read(1).astype("float32")
                        v = np.isfinite(a) & (a != s.nodata) & (a > -1e4)
                        if c == "hand_dinf":
                            v &= a >= 0
                        if c in LOG:
                            a = np.log1p(np.where(v & (a > 0), a, 0))
                        pil.append((np.where(v, (a - media[k]) / desvio[k], 0), v))
                        perfil = (s.transform, s.crs, s.height, s.width)
                ter = np.stack([p for p, _ in pil]).astype("float32")
                tv = np.all([v for _, v in pil], 0)
                with rasterio.open(s11.caminho(nome, "rotulo")) as src:
                    lab = src.read(1).astype("float32")
                    agua = np.where(lab == 1, 1.0, np.where(lab == 0, 0.0, np.nan)).astype("float32")
                    dest = np.full((perfil[2], perfil[3]), np.nan, "float32")
                    reproject(agua, dest, src_transform=src.transform, src_crs=src.crs, dst_transform=perfil[0], dst_crs=perfil[1],
                              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
                lv = np.isfinite(dest)
                np.savez_compressed(alvo, terreno=ter, tv=tv, radar=radar.astype("float32"), rv=rv, rotulo=np.where(lv, dest >= .5, 0).astype("uint8"), lv=lv)
            z = np.load(alvo)
            meta.append({"chip": nome, "split": sp, "estado": "ok", "pais": nome.split("_")[0], "pixels_rotulados": int(z["lv"].sum()),
                         "fracao_agua": float(z["rotulo"][z["lv"]].mean()) if z["lv"].any() else None, "radar_valido": float(z["rv"].mean())})
    m = pd.DataFrame(meta)
    tr = [np.load(CH / f"{c}.npz") for c in m[(m.split == "treino") & (m.estado == "ok")].chip]
    vals = [np.concatenate([z["radar"][i][z["rv"]] for z in tr]) for i in range(2)]
    g9(D / "chips.csv", m)
    g9(D / "normalizacao_radar.json", {"media": [float(v.mean()) for v in vals], "desvio": [float(v.std()) for v in vals], "origem": "so chips de treino"})
    etapa("d_preparar", chips=int((m.estado == "ok").sum()))
    print(m.groupby(["split", "estado"]).size())


class Observador:
    """Ramo de radar + atributos do encoder de terreno -> decodificador -> logito por pixel (30 m)."""
    def __init__(self, semente):
        import torch
        import torch.nn as nn
        torch.manual_seed(semente)
        self.radar = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.fusao = nn.Sequential(nn.Conv2d(64 + 32 + 1, 64, 3, padding=1), nn.ReLU(),
                                   nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(), nn.ConvTranspose2d(32, 16, 2, stride=2), nn.ReLU(), nn.Conv2d(16, 1, 1))

    def params(self):
        return list(self.radar.parameters()) + list(self.fusao.parameters())

    def __call__(self, feat_ter, radar, presente):
        import torch
        r = self.radar(radar) * presente[:, None, None, None]
        p = presente[:, None, None, None].expand(-1, 1, *feat_ter.shape[2:])
        return self.fusao(torch.cat([feat_ter, r, p], 1)).squeeze(1)

    def modo(self, t):
        self.radar.train(t)
        self.fusao.train(t)


def lote_chips(chips, rng, nr, n=8, rotulos_trocados=None):
    T, R, Y, V = [], [], [], []
    for _ in range(n):
        i = rng.integers(len(chips))
        z = chips[i]
        h, w = z["terreno"].shape[1:]
        if h < RECORTE or w < RECORTE:
            continue
        a, b = rng.integers(0, h - RECORTE + 1), rng.integers(0, w - RECORTE + 1)
        lab_z = chips[rotulos_trocados[i]] if rotulos_trocados is not None else z
        hl, wl = lab_z["rotulo"].shape
        a2, b2 = (a, b) if rotulos_trocados is None else (rng.integers(0, hl - RECORTE + 1), rng.integers(0, wl - RECORTE + 1))
        T.append(z["terreno"][:, a:a + RECORTE, b:b + RECORTE])
        R.append(np.where(z["rv"][a:a + RECORTE, b:b + RECORTE], (z["radar"][:, a:a + RECORTE, b:b + RECORTE] - nr[0][:, None, None]) / nr[1][:, None, None], 0))
        Y.append(lab_z["rotulo"][a2:a2 + RECORTE, b2:b2 + RECORTE])
        V.append(lab_z["lv"][a2:a2 + RECORTE, b2:b2 + RECORTE])
    return [np.stack(v).astype("float32") for v in (T, R, Y, V)]


def carregar_chips(split):
    m = pd.read_csv(D / "chips.csv")
    return [dict(np.load(CH / f"{c}.npz")) for c in m[(m.split == split) & (m.estado == "ok")].chip]


def avaliar_O(enc, obs, chips, nr, presente, terreno_zero=False):
    import torch
    enc.eval()
    obs.modo(False)
    ys, ss = [], []
    with torch.no_grad():
        for z in chips:
            h, w = z["terreno"].shape[1:]
            h4, w4 = h // 4 * 4, w // 4 * 4
            t = torch.tensor(z["terreno"][None, :, :h4, :w4])
            if terreno_zero:
                t = torch.zeros_like(t)
            r = torch.tensor(np.where(z["rv"], (z["radar"] - nr[0][:, None, None]) / nr[1][:, None, None], 0)[None, :, :h4, :w4], dtype=torch.float32)
            s = obs(enc(t), r, torch.tensor([float(presente)])).numpy()[0]
            v = z["lv"][:h4, :w4]
            ys.append(z["rotulo"][:h4, :w4][v])
            ss.append(s[v])
    y, s = np.concatenate(ys), np.concatenate(ss)
    pred = s > 0
    iou = float((pred & (y == 1)).sum() / max(1, (pred | (y == 1)).sum()))
    return {"auc": auc(y, s), "iou": iou, "pixels": int(len(y)), "fracao_agua": float(y.mean())}


def treinar(braco, semente, passos, S, chips_tr, nr, terreno_zero=False, radar_nunca=False):
    """braco in {S_sep, O_sep, stopgrad, conjunta, quebrado, O_fusao, O_radar, O_terreno}."""
    import torch
    torch.manual_seed(semente)
    torch.set_num_threads(8)
    rng = np.random.default_rng(semente)
    enc = encoder(True, semente)
    cab = torch.nn.Linear(128, 1)
    torch.nn.init.zeros_(cab.weight)
    torch.nn.init.zeros_(cab.bias)
    obs = Observador(semente)
    usa_S = braco in ("S_sep", "stopgrad", "conjunta", "quebrado")
    usa_O = braco not in ("S_sep",)
    trocados = rng.permutation(len(chips_tr)) if braco == "quebrado" else None
    params = list(enc.parameters()) + list(cab.parameters()) + obs.params()
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    P, q, y, w, idx = S
    cos = []
    t0 = time.time()
    for it in range(passos):
        enc.train()
        obs.modo(True)
        perdas = {}
        if usa_S:
            b = rng.choice(idx, 128, replace=False)
            z = enc(torch.tensor(P[b], dtype=torch.float32))
            emb = torch.cat([z.mean((2, 3)), z.std((2, 3))], 1)
            logit = cab(emb).squeeze(-1) + torch.tensor(q[b])
            perdas["S"] = (torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(y[b]), reduction="none") * torch.tensor(w[b])).mean()
        if usa_O:
            T, R, Y, V = lote_chips(chips_tr, rng, nr, rotulos_trocados=trocados)
            T = torch.tensor(T)
            if terreno_zero:
                T = torch.zeros_like(T)
            feat = enc(T)
            if braco == "stopgrad":
                feat = feat.detach()
            pres = torch.tensor((rng.random(len(R)) > .3).astype("float32")) if not radar_nunca else torch.zeros(len(R))
            if braco == "O_radar":
                pres = torch.ones(len(R))
            s = obs(feat, torch.tensor(R), pres)
            perdas["O"] = (torch.nn.functional.binary_cross_entropy_with_logits(s, torch.tensor(Y), reduction="none") * torch.tensor(V)).sum() / max(1.0, float(V.sum()))
        if usa_S and usa_O and braco != "stopgrad" and it % 50 == 0:
            gs = torch.autograd.grad(perdas["S"], list(enc.parameters()), retain_graph=True, allow_unused=True)
            go = torch.autograd.grad(perdas["O"], list(enc.parameters()), retain_graph=True, allow_unused=True)
            a = torch.cat([g.flatten() for g in gs if g is not None])
            bb = torch.cat([g.flatten() for g in go if g is not None])
            cos.append(float(torch.nn.functional.cosine_similarity(a, bb, 0)))
        perda = sum(perdas.values())
        opt.zero_grad()
        perda.backward()
        opt.step()
    return enc, cab, obs, {"segundos": time.time() - t0, "cosseno_gradientes": cos}


def experimento(modo):
    import torch
    nr0 = json.loads((D / "normalizacao_radar.json").read_text(encoding="utf-8"))
    nr = (np.array(nr0["media"], "float32"), np.array(nr0["desvio"], "float32"))
    chips_tr, chips_te = carregar_chips("treino"), carregar_chips("teste")
    from v9_c_patches import dados_externos
    tr, x, Q, P, q, y, w, fd = dados_externos()
    S = (P, q, y, w, np.arange(len(y)))
    piloto = D / "piloto.json"
    if not piloto.exists():
        t0 = time.time()
        treinar("conjunta", 0, 20, S, chips_tr, nr)
        g9(piloto, {"s_por_passo_conjunto": (time.time() - t0) / 20, "quando_utc": agora()})
    s_passo = json.loads(piloto.read_text(encoding="utf-8"))["s_por_passo_conjunto"]
    passos = int(min(3000, max(400, 900 / s_passo)))
    bracos = ["O_fusao", "O_radar", "O_terreno"] if modo == "observacao" else ["S_sep", "O_sep", "stopgrad", "conjunta", "quebrado"]
    congelar9(D / f"protocolo_{modo}.json", {"versao": f"v9-d-{modo}-1", "bracos": bracos, "sementes": [1, 2, 3], "passos": passos,
                                             "dropout_modalidade": .3, "recorte": RECORTE, "lr": 3e-4, "wd": 1e-4, "encoder": "encoder_v2_autosup",
                                             "proibido": "cabeca S nunca recebe radar/pos-evento/mascara"})
    linhas = []
    for braco in bracos:
        for s in (1, 2, 3):
            enc, cab, obs, info = treinar(braco, s, passos, S, chips_tr, nr, terreno_zero=(braco == "O_radar"), radar_nunca=(braco == "O_terreno"))
            torch.save({"enc": enc.state_dict(), "cab": cab.state_dict(), "obs_radar": obs.radar.state_dict(), "obs_fusao": obs.fusao.state_dict()},
                       D / f"modelo_{modo}_{braco}_s{s}.pt")
            r = {"modo": modo, "braco": braco, "semente": s, "passos": passos, **{k: v for k, v in info.items() if k != "cosseno_gradientes"},
                 "cosseno_medio": float(np.mean(info["cosseno_gradientes"])) if info["cosseno_gradientes"] else None,
                 "cosseno_frac_negativo": float(np.mean(np.array(info["cosseno_gradientes"]) < 0)) if info["cosseno_gradientes"] else None}
            if braco != "S_sep":
                for pres in (1, 0):
                    o = avaliar_O(enc, obs, chips_te, nr, pres, terreno_zero=(braco == "O_radar"))
                    r.update({f"O_teste_radar{pres}_{k}": v for k, v in o.items()})
            if braco in ("S_sep", "stopgrad", "conjunta", "quebrado"):
                ms = ModeloS("ft_total", s)
                ms.enc, ms.cab = enc, cab
                for c in COORTES:
                    ids = pd.read_parquet(DC / f"ids_{c}.parquet")
                    pc = pd.read_parquet(DC / f"predicoes_{c}.parquet", columns=["ponto_id", "Q_sel_ext"]) if (DC / f"predicoes_{c}.parquet").exists() else None
                    if pc is None:
                        continue
                    pc = ids[["ponto_id", "classe_territorial", "estado_primaria"]].merge(pc, on="ponto_id")
                    sc = prever_s(ms, np.load(DC / f"patches_{c}.npy", mmap_mode="r"), pc.Q_sel_ext.to_numpy("float32"))
                    yy = pc.estado_primaria.eq("positivo").to_numpy()
                    f = pc.classe_territorial.eq("fcu").to_numpy()
                    r[f"S_{c}_todas"], r[f"S_{c}_fcu"] = auc(yy, sc), auc(yy[f], sc[f])
            linhas.append(r)
            g9(D / f"resultados_{modo}.csv", pd.DataFrame(linhas))
            etapa(f"d_{modo}", braco=braco, semente=s, segundos=info["segundos"])
            print(braco, s, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True)
    g9(D / f"resultado_{modo}.json", {"quando_utc": agora(), "linhas": len(linhas)})


def reavaliar_radar():
    """Correcao registrada: o braco O_radar foi TREINADO com terreno zerado, mas a primeira avaliacao lhe deu o terreno
    real (AUC 0,30 era artefato). Reavalia os pesos salvos com o terreno zerado, sem retreinar."""
    import torch
    nr0 = json.loads((D / "normalizacao_radar.json").read_text(encoding="utf-8"))
    nr = (np.array(nr0["media"], "float32"), np.array(nr0["desvio"], "float32"))
    chips_te = carregar_chips("teste")
    linhas = []
    for s in (1, 2, 3):
        e = torch.load(D / f"modelo_observacao_O_radar_s{s}.pt", map_location="cpu")
        enc, obs = encoder(False, s), Observador(s)
        enc.load_state_dict(e["enc"])
        obs.radar.load_state_dict(e["obs_radar"])
        obs.fusao.load_state_dict(e["obs_fusao"])
        o = avaliar_O(enc, obs, chips_te, nr, 1, terreno_zero=True)
        linhas.append({"braco": "O_radar", "semente": s, **{f"O_teste_radar1_{k}": v for k, v in o.items()}})
        print(linhas[-1], flush=True)
    g9(D / "reavaliacao_O_radar.csv", pd.DataFrame(linhas))
    etapa("d_reavaliar_radar", nota="avaliacao inicial do O_radar usava terreno real; corrigida")


if __name__ == "__main__":
    a = sys.argv[1]
    {"preparar": preparar, "reavaliar_radar": reavaliar_radar}.get(a, lambda: experimento(a))()
