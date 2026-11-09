"""v9 Frente C: representacao espacial de terreno em patches (encoder mascarado existente) para suscetibilidade.

extrair  patches 32x32 (960 m a 30 m) nas 5 camadas do contrato do encoder (dem, declive, HAND, SCA, acumulacao; log1p
         em SCA/acumulacao; normalizacao do encoder_v2_autosup; invalido -> 0 + mascara). Externo: os 36.331 pontos
         fora_BR (840 diretorios de terreno). Coortes: particao teste, todos os positivos + ate 20.000 negativos
         sorteados (semente fixa; AUC nao enviesada por subamostragem uniforme de negativos; declarado).
piloto   mede s/lote de forward/backward e de inferencia; define amostras por epoca para caber no orcamento.
treinar  familias (todas como correcao residual sobre o logito do Q_sel_ext, camada final iniciada em zero, e as
         sondas tambem sem o Q):
           sonda_pre / sonda_aleatorio   encoder pre-treinado ou aleatorio CONGELADO + linear sobre media/desvio (128)
           ft_parcial                    pre-treinado, so o nivel 3 + cabeca treinam
           ft_total                      pre-treinado, tudo treina
           aleatorio_total               mesmo encoder do zero, tudo treina (supervisionado sem pre-treino)
         grade lr {3e-4, 1e-3} x wd {1e-4, 1e-2}; parada antecipada na dobra 0 de unidades externas; 2 melhores x 3
         sementes. Pre-treino: chips Sen1Floods11 sem rotulo (inclui o terreno, nao o rotulo, de S1F11:USA, unidade
         de retencao -- declarado). Nenhum dado brasileiro no treino nem na selecao.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v9_comum import COORTES, REPO, TER, V9, agora, auc, congelar9, etapa, g9

D = V9 / "c_espacial"
LADO = 32
CAMADAS = ("dem_filled", "slope_deg_wbt", "hand_dinf", "dinf_sca", "dinf_accum_cells")
LOG = ("dinf_sca", "dinf_accum_cells")
ENC = REPO / "local_runs/terreno-encoder/encoder_v2_autosup.pt"
ENC_MAN = REPO / "local_runs/terreno-encoder/encoder_v2_autosup_manifesto.json"
NEG_COORTE = 20000


def normalizacao():
    m = json.loads(ENC_MAN.read_text(encoding="utf-8"))["normalizacao"]
    assert tuple(m["camadas"]) == CAMADAS
    return np.array(m["media"], "float32"), np.array(m["desvio"], "float32")


def patches_no_dir(pasta: Path, lon, lat, media, desvio):
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import rowcol
    pil = []
    for c in CAMADAS:
        with rasterio.open(pasta / f"{c}.tif") as s:
            a = s.read(1).astype("float32")
            v = np.isfinite(a) & (a != s.nodata) & (a > -1e4)
            if c == "hand_dinf":
                v &= a >= 0
            if c in LOG:
                a = np.log1p(np.where(v & (a > 0), a, 0))
            pil.append((a, v))
            tr_, crs = s.transform, s.crs
    h, w = pil[0][0].shape
    px, py = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    lin, col = (np.asarray(t) for t in rowcol(tr_, px, py))
    out = np.zeros((len(lon), len(CAMADAS), LADO, LADO), "float16")
    frac = np.zeros(len(lon), "float32")
    r = LADO // 2
    for i, (li, co) in enumerate(zip(lin, col)):
        r0, c0 = li - r, co - r
        a0, b0 = max(r0, 0), max(c0, 0)
        a1, b1 = min(r0 + LADO, h), min(c0 + LADO, w)
        if a1 <= a0 or b1 <= b0:
            continue
        ok = np.ones((a1 - a0, b1 - b0), bool)
        for k, (a, v) in enumerate(pil):
            ok &= v[a0:a1, b0:b1]
        for k, (a, v) in enumerate(pil):
            blk = (a[a0:a1, b0:b1] - media[k]) / desvio[k]
            out[i, k, a0 - r0:a1 - r0, b0 - c0:b1 - c0] = np.where(ok, blk, 0)
        frac[i] = ok.sum() / LADO ** 2
    return out, frac


def extrair():
    from v8_04_campanha import coorte, externo
    media, desvio = normalizacao()
    D.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    alvo = D / "patches_externo.npy"
    if not alvo.exists():
        tr, _, _ = externo()
        P = np.zeros((len(tr), len(CAMADAS), LADO, LADO), "float16")
        F = np.zeros(len(tr), "float32")
        for u, g in tr.groupby("unidade_terreno"):
            p, f = patches_no_dir(TER / u, g.lon.to_numpy(), g.lat.to_numpy(), media, desvio)
            P[g.index.to_numpy()], F[g.index.to_numpy()] = p, f
        np.save(alvo, P)
        g9(D / "ids_externo.parquet", tr[["ponto_id", "unidade", "evento_id", "classe", "fonte"]].assign(fracao_valida=F))
        print("externo", len(tr), round(time.time() - t0), flush=True)
    for c in COORTES:
        arq = D / f"patches_{c}.npy"
        if arq.exists():
            continue
        d = coorte(c)
        te = d[d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])]
        pos, neg = te[te.estado_primaria.eq("positivo")], te[te.estado_primaria.eq("negativo")]
        neg = neg.sample(min(NEG_COORTE, len(neg)), random_state=20261023)
        q = pd.concat([pos, neg]).reset_index(drop=True)
        tiles = pd.read_parquet(REPO / "local_runs/transferencia-vulnerabilizados/v8_validacao_integrada/ponte" / f"{c}.parquet", columns=["ponto_id", "tile"])
        q = q.drop(columns=[k for k in ["tile"] if k in q]).merge(tiles, on="ponto_id")
        P = np.zeros((len(q), len(CAMADAS), LADO, LADO), "float16")
        F = np.zeros(len(q), "float32")
        lonlat = pd.read_parquet(REPO / "local_runs/transferencia-vulnerabilizados/v8_validacao_integrada/contrato/contrato_coortes_brasileiras.parquet",
                                 columns=["ponto_id", "coorte", "lon", "lat"]).query("coorte == @c").drop(columns="coorte")
        q = q.merge(lonlat, on="ponto_id")
        for t, g in q.groupby("tile"):
            p, f = patches_no_dir(TER / t, g.lon.to_numpy(), g.lat.to_numpy(), media, desvio)
            P[g.index.to_numpy()], F[g.index.to_numpy()] = p, f
        np.save(arq, P)
        g9(D / f"ids_{c}.parquet", q[["ponto_id", "grupo", "classe_territorial", "estado_primaria"]].assign(fracao_valida=F, peso_negativo=len(te[te.estado_primaria.eq("negativo")]) / max(1, len(neg))))
        print(c, len(q), round(time.time() - t0), flush=True)
    etapa("c_extrair", segundos=time.time() - t0)


def encoder(pre: bool, semente: int):
    import torch
    sys.path.insert(0, str(REPO / "scripts/terreno"))
    from modelo_encoder import AutoencoderMascarado
    torch.manual_seed(semente)
    ae = AutoencoderMascarado(5, 16)
    if pre:
        ae.load_state_dict(torch.load(ENC, map_location="cpu"))
    return ae.encoder


class Modelo:
    """encoder -> media/desvio (128) -> linear; logit = [q] + linear."""
    def __init__(self, familia, semente, residual=True):
        import torch.nn as nn
        self.familia, self.residual = familia, residual
        self.enc = encoder(familia not in ("sonda_aleatorio", "aleatorio_total"), semente)
        self.cab = nn.Linear(128, 1)
        nn.init.zeros_(self.cab.weight)
        nn.init.zeros_(self.cab.bias)
        treina = {"sonda_pre": [], "sonda_aleatorio": [], "ft_parcial": list(self.enc.n3.parameters()),
                  "ft_total": list(self.enc.parameters()), "aleatorio_total": list(self.enc.parameters())}[familia]
        for p in self.enc.parameters():
            p.requires_grad = False
        for p in treina:
            p.requires_grad = True
        self.params = treina + list(self.cab.parameters())

    def emb(self, x):
        import torch
        z = self.enc(x)
        return torch.cat([z.mean((2, 3)), z.std((2, 3))], 1)

    def __call__(self, x, q):
        s = self.cab(self.emb(x)).squeeze(-1)
        return s + q if self.residual else s

    def modo(self, treino):
        self.enc.train(treino and self.familia in ("ft_parcial", "ft_total", "aleatorio_total"))
        self.cab.train(treino)

    def estado(self):
        return {"enc": self.enc.state_dict(), "cab": self.cab.state_dict()}


def prever(m, P, q, lote=1024):
    import torch
    m.modo(False)
    out = []
    with torch.no_grad():
        for i in range(0, len(P), lote):
            out.append(m(torch.tensor(P[i:i + lote], dtype=torch.float32), torch.tensor(q[i:i + lote], dtype=torch.float32)).numpy())
    return np.concatenate(out)


def treinar_um(familia, cfg, P, q, y, w, itr, iva, semente, amostras_epoca, epocas_max=10, epocas_fixas=None, residual=True):
    import torch
    torch.set_num_threads(8)
    m = Modelo(familia, semente, residual)
    opt = torch.optim.AdamW(m.params, lr=cfg["lr"], weight_decay=cfg["wd"])
    rng = np.random.default_rng(semente)
    melhor, est, pac, hist = -np.inf, None, 0, []
    for ep in range(epocas_fixas or epocas_max):
        m.modo(True)
        sel = rng.choice(itr, min(amostras_epoca, len(itr)), replace=False)
        for i in range(0, len(sel), 128):
            b = sel[i:i + 128]
            logit = m(torch.tensor(P[b], dtype=torch.float32), torch.tensor(q[b], dtype=torch.float32))
            perda = (torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(y[b], dtype=torch.float32), reduction="none")
                     * torch.tensor(w[b], dtype=torch.float32)).mean()
            opt.zero_grad()
            perda.backward()
            opt.step()
        if iva is not None and epocas_fixas is None:
            a = auc(y[iva], prever(m, P[iva], q[iva]))
            hist.append(a)
            if a > melhor + 1e-4:
                melhor, est, pac = a, {k: {kk: vv.clone() for kk, vv in v.items()} for k, v in m.estado().items()}, 0
            else:
                pac += 1
                if pac >= 3:
                    break
    if est is not None:
        m.enc.load_state_dict(est["enc"])
        m.cab.load_state_dict(est["cab"])
    return m, {"epocas": len(hist) if hist else (epocas_fixas or epocas_max), "melhor_val": melhor, "historico": hist}


def dados_externos():
    import pickle as pk
    from v8_04_campanha import externo
    from v9_b_tabular import pesos
    tr, x, _ = externo()
    with (V9 / "p0_regularizacao/modelos.pkl").open("rb") as f:
        Q = pk.load(f)["Q_sel_ext"]
    P = np.load(D / "patches_externo.npy", mmap_mode="r")
    unid = np.array(sorted(tr.unidade.unique()))
    dobra = dict(zip(unid, np.random.default_rng(20261023).permutation(len(unid)) % 3))
    return tr, x, Q, P, Q.escore(x).astype("float32"), tr.classe.to_numpy().astype("float32"), pesos(tr).astype("float32"), tr.unidade.map(dobra).to_numpy()


def piloto():
    import torch
    tr, x, Q, P, q, y, w, fd = dados_externos()
    out = {}
    for fam in ["sonda_pre", "ft_total"]:
        m = Modelo(fam, 0)
        opt = torch.optim.AdamW(m.params, lr=1e-3)
        t0 = time.time()
        for i in range(10):
            b = np.arange(i * 128, (i + 1) * 128)
            logit = m(torch.tensor(P[b], dtype=torch.float32), torch.tensor(q[b], dtype=torch.float32))
            perda = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.tensor(y[b]))
            opt.zero_grad()
            perda.backward()
            opt.step()
        tl = (time.time() - t0) / 10
        t0 = time.time()
        prever(m, P[:4096], q[:4096])
        ti = (time.time() - t0) / 4096
        torch.save(m.estado(), D / f"piloto_{fam}.pt")
        out[fam] = {"s_por_lote128_treino": tl, "s_por_amostra_inferencia": ti}
    orc_epoca_s = 360
    amostras = int(min(len(tr), orc_epoca_s / out["ft_total"]["s_por_lote128_treino"] * 128))
    g9(D / "piloto.json", {"quando_utc": agora(), **out, "orcamento_epoca_s": orc_epoca_s, "amostras_por_epoca": amostras,
                           "estimativa_total_h": (amostras / 128 * out["ft_total"]["s_por_lote128_treino"] * 10 * 21 + 600) / 3600,
                           "nota": "piloto atravessou leitura, forward/backward, validacao e salvamento"})
    etapa("c_piloto", amostras=amostras)
    print(out, amostras)


def treinar():
    import torch
    if not (D / "piloto.json").exists():
        piloto()
    pil = json.loads((D / "piloto.json").read_text(encoding="utf-8"))
    ame = pil["amostras_por_epoca"]
    prot = congelar9(D / "protocolo.json", {"versao": "v9-c-1", "lado": LADO, "camadas": list(CAMADAS), "encoder": ENC.name,
                                            "familias": ["sonda_pre", "sonda_aleatorio", "ft_parcial", "ft_total", "aleatorio_total"],
                                            "grade": [{"lr": a, "wd": b} for a in (3e-4, 1e-3) for b in (1e-4, 1e-2)], "sementes_finais": [1, 2, 3],
                                            "amostras_por_epoca": ame, "parada": "paciencia 3, max 10 epocas, AUC na dobra 0 externa",
                                            "residual": "logit = Q + cabeca (cabeca iniciada em zero); sondas tambem sem Q"})
    tr, x, Q, P, q, y, w, fd = dados_externos()
    itr, iva = np.where(fd != 0)[0], np.where(fd == 0)[0]
    tudo = np.arange(len(y))
    grade = prot["grade"]
    busca, finais = [], {}
    t0 = time.time()
    for fam in prot["familias"]:
        notas = []
        for cfg in grade:
            m, info = treinar_um(fam, cfg, P, q, y, w, itr, iva, 0, ame)
            notas.append((cfg, info["melhor_val"], info["epocas"]))
            busca.append({"familia": fam, **cfg, "auc_val_pontos": info["melhor_val"], "epocas": info["epocas"], "residual": True})
            g9(D / "busca.csv", pd.DataFrame(busca))
        cfg, nota, ep = max(notas, key=lambda t: t[1])
        finais[fam] = {"config": cfg, "auc_val": nota, "epocas": max(1, ep)}
        for s in (1, 2, 3):
            m, _ = treinar_um(fam, cfg, P, q, y, w, tudo, None, s, ame, epocas_fixas=max(1, ep))
            torch.save(m.estado(), D / f"modelo_{fam}_s{s}.pt")
        if fam in ("sonda_pre", "sonda_aleatorio"):
            for s in (1, 2, 3):
                m, _ = treinar_um(fam, cfg, P, q, y, w, tudo, None, s, ame, epocas_fixas=max(1, ep), residual=False)
                torch.save(m.estado(), D / f"modelo_{fam}_semQ_s{s}.pt")
        etapa("c_familia", familia=fam, auc_val=nota, segundos=time.time() - t0)
        print(fam, cfg, nota, round(time.time() - t0), flush=True)
    linhas = []
    for c in COORTES:
        ids = pd.read_parquet(D / f"ids_{c}.parquet")
        Pc = np.load(D / f"patches_{c}.npy", mmap_mode="r")
        from v8_04_campanha import caracteristicas, coorte
        d = coorte(c)
        xx = caracteristicas(d).set_index(d.ponto_id).loc[ids.ponto_id].reset_index(drop=True)
        qc = Q.escore(xx).astype("float32")
        yc = ids.estado_primaria.eq("positivo").astype(int).to_numpy()
        pred = ids[["ponto_id", "grupo", "classe_territorial", "fracao_valida"]].assign(y=yc, Q_sel_ext=qc)
        b = pd.read_parquet(V9 / f"b_tabular/predicoes_{c}.parquet", columns=["ponto_id", "MLP_res", "GB_mono", "A0_hand", "menos_declive"]) if (V9 / f"b_tabular/predicoes_{c}.parquet").exists() else None
        if b is not None:
            pred = pred.merge(b, on="ponto_id", how="left")
        for fam in finais:
            for suf, res in [("", True), ("_semQ", False)] if fam in ("sonda_pre", "sonda_aleatorio") else [("", True)]:
                ss = []
                for s in (1, 2, 3):
                    m = Modelo(fam, s, res)
                    e = torch.load(D / f"modelo_{fam}{suf}_s{s}.pt", map_location="cpu")
                    m.enc.load_state_dict(e["enc"])
                    m.cab.load_state_dict(e["cab"])
                    ss.append(prever(m, Pc, qc))
                pred[f"{fam}{suf}"] = np.mean(ss, 0)
                for s, v in zip((1, 2, 3), ss):
                    pred[f"{fam}{suf}|s{s}"] = v
        g9(D / f"predicoes_{c}.parquet", pred)
        cols = [k for k in pred.columns if k not in ("ponto_id", "grupo", "classe_territorial", "fracao_valida", "y") and "|s" not in k]
        for pop in ["todas", "fcu"]:
            qq = pred if pop == "todas" else pred[pred.classe_territorial.eq(pop)]
            linhas.append({"coorte": c, "populacao": pop, "positivos": int(qq.y.sum()), "n": len(qq), **{k: auc(qq.y, qq[k]) for k in cols if qq[k].notna().all()}})
        print(c, flush=True)
    g9(D / "auc_coortes.csv", pd.DataFrame(linhas))
    g9(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "finais": finais, "piloto": pil})
    etapa("c_espacial", segundos=time.time() - t0)
    print(pd.DataFrame(linhas).round(3).to_string())


if __name__ == "__main__":
    {"extrair": extrair, "piloto": piloto, "treinar": treinar}[sys.argv[1]]()
