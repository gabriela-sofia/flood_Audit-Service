"""v8: campanha de transferencia e poucos rotulos sob o contrato fisico Q (ponte MT01/MT11 -> coortes FCU).

DESENVOLVIMENTO/REANALISE: RMPA/2024, Manaus/2021, Curitiba/2023 (particoes v3, IDs aninhados v4) e Recife/RMR/2022
(so sem rotulo local) ja foram examinados. Nada aqui e confirmacao.

Atributos (mesmos nomes e transformacoes do Q): f_hand, f_sca (ln_sca_twi), f_declive, f_construido (GHSL), f_areia
(SoilGrids; NaN imputado pela mediana do ajuste, regra do proprio contrato Q), f_dist_dren, f_dist_canal.

Sem rotulo local (B=0): regras HAND, -declive, TWI; melhor regra fisica escolhida nos dados EXTERNOS (fora_BR, MT03-A),
nunca no destino; Q congelado (MT05: contem EMSR720/RS-2024 e GFD brasileiro); Q_ext (fora_BR) terreno/base/conexao;
Q_extAS (fora_AS); Q_perm (rotulos permutados); Q_ext_sem_areia (rota parcial); GBM_ext monotonico (busca de 4
configuracoes por validacao agrupada por unidade nos dados externos); A1_v3 (linha territorial, 5 atributos WorldCover).

Poucos rotulos (B = 25/100/400 x 3 politicas x 6 sementes; IDs v4), mesmos 7 atributos, logistica balanceada com prior
gaussiano e precisao tau em {1, 25} (pre-declarada, sem selecao):
  L_estrito      escala local (dos B pontos), prior 0
  L_escext       escala externa (Q_ext), prior 0              <- prior neutro com a MESMA escala e tau do P_ext
  P_ext          escala externa, prior = Q_ext conexao
  P_perm         escala externa, prior = Q_perm (controle de desenvolvimento)
  P_ext_escloc   escala local, prior Q_ext reexpresso na escala local (b * sd_loc / sd_ext)
  GBM_local      HistGB monotonico (prof. 2, 100 iteracoes, folha 5), so com duas classes e B >= 100
Contagem separada: ajustes com duas classes, copias do prior (uma classe) e ajustes impossiveis.
Subcomandos: protocolo | executar | (analise em v8_04b)
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
from v8_comum import MT, TV, V3A, V4, V7, V8, agora, congelar8, etapa, g8, ler, sha_hex

D = V8 / "campanha"
P = V8 / "ponte"
FEAT = ["f_hand", "f_sca", "f_declive", "f_construido", "f_areia", "f_dist_dren", "f_dist_canal"]
MONO = [-1, 1, -1, 0, 0, 0, -1]
TAUS = [1.0, 25.0]
COORTES_B = ["rmpa", "manaus", "curitiba"]
COORTES_0 = ["recife", "rmr"]
GRADE_GBM = [{"max_depth": d, "max_iter": it} for d in (2, 3) for it in (100, 300)]

PROTOCOLO = {"versao": "v8-campanha-1", "natureza": "desenvolvimento/reanalise; nenhuma confirmacao",
             "atributos": FEAT, "monotonia": dict(zip(FEAT, MONO)), "taus": TAUS, "orcamentos": [25, 100, 400],
             "politicas": ["urbano_uniforme", "fcu_comunidades", "baixa_renda_setores"], "sementes": 6,
             "bracos_B": ["L_estrito", "L_escext", "P_ext", "P_perm", "P_ext_escloc", "GBM_local"],
             "bracos_0": ["A0_hand", "menos_declive", "twi", "melhor_fisica_ext", "Q_congelado", "Q_ext_terreno", "Q_ext_base",
                          "Q_ext_conexao", "Q_extAS_conexao", "Q_perm_conexao", "Q_ext_sem_areia", "GBM_ext", "A1_v3"],
             "escolhas_so_externas": "melhor regra fisica e configuracao do GBM_ext por AUC dentro do evento media por unidade nos dados fora_BR",
             "avaliacao": "particao teste v3 (estados positivo/negativo); populacoes fcu, baixa_renda, comparador_urbano, todas; agrupada e macro FCU (>=5/5)",
             "contaminacao": "Q_congelado viu EMSR720 (mesmo episodio da RMPA) e 73 eventos GFD brasileiros"}


def externo():
    """Amostra MT03-A com marcas fora_BR (mesmo procedimento do v8_01)."""
    import mt05_transferencia_sem_amostra as mt
    d = mt.e3.carregar()
    tr = mt.e1.amostrar(d, mt.e3.CONFIG["teto_ajuste_por_evento_classe"], mt.e3.CONFIG["semente"])
    geo = pd.read_parquet(V7 / "v7_02_contextos_tempo/geografia_pontos.parquet", columns=["ponto_id", "exterior_auditado"])
    tr = tr.merge(geo, on="ponto_id", how="left")
    ext = tr.groupby("evento_id").exterior_auditado.apply(lambda s: bool(s.fillna(False).all()))
    tr = tr[tr.evento_id.isin(ext.index[ext])].reset_index(drop=True)
    return tr, mt.caracteristicas(tr), mt


def auc_evento_unidade(tr, s):
    from mt01_experimento import auc
    v = []
    for _, gu in tr.assign(_s=s).groupby("unidade"):
        a = [auc(g.classe.to_numpy(), g._s.to_numpy()) for _, g in gu.groupby("evento_id")]
        a = [x for x in a if np.isfinite(x)]
        if a:
            v.append(np.mean(a))
    return float(np.mean(v))


def pesos_evento(tr):
    n = tr.groupby(["evento_id", "classe"]).classe.transform("size")
    return (.5 / n / tr.evento_id.nunique()).to_numpy()


def gbm(cfg, folha=50, taxa=.05):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=cfg["max_depth"], max_iter=cfg["max_iter"], learning_rate=taxa,
                                          min_samples_leaf=folha, monotonic_cst=MONO, random_state=0)


def escolhas_externas():
    tr, x, mt = externo()
    regras = {"A0_hand": -tr.hand_m.to_numpy(), "menos_declive": -tr.slope_deg.to_numpy(), "twi": tr.twi_dinf.to_numpy()}
    notas = {k: auc_evento_unidade(tr, np.nan_to_num(v, nan=np.nanmedian(v))) for k, v in regras.items()}
    unid = np.array(sorted(tr.unidade.unique()))
    dobra = dict(zip(unid, np.random.default_rng(20261022).permutation(len(unid)) % 3))
    f = tr.unidade.map(dobra).to_numpy()
    cv = []
    for cfg in GRADE_GBM:
        s = np.zeros(len(tr))
        for k in range(3):
            te = f == k
            m = gbm(cfg).fit(x.loc[~te, FEAT], tr.classe[~te], sample_weight=pesos_evento(tr[~te]))
            s[te] = m.predict_proba(x.loc[te, FEAT])[:, 1]
        cv.append({**cfg, "auc_evento_unidade": auc_evento_unidade(tr, s)})
    melhor_cfg = max(cv, key=lambda r: r["auc_evento_unidade"])
    gext = gbm(melhor_cfg).fit(x[FEAT], tr.classe, sample_weight=pesos_evento(tr))
    lam = ler(mt.e3.SAIDA / "candidatas_congeladas.json")["lam"]
    sem_areia = mt.e1.Logistica([c for c in FEAT if c != "f_areia"], (), lam_p=lam["p"], lam_i=lam["i"])
    sem_areia.fit(x, tr.classe.to_numpy(float), tr.evento_id.to_numpy())
    return {"regras": notas, "melhor_fisica": max(notas, key=notas.get), "gbm_cv": cv, "gbm_cfg": melhor_cfg}, gext, sem_areia


def coorte(nome):
    from v8_02_ponte_fisica import FONTES
    cols = ["ponto_id", "hand_m", "slope_deg", "twi_dinf", "classe_territorial", "CD_FCU", "CD_SETOR"]
    extra = ["estado_primaria", "particao", "grupo"] if nome in COORTES_B else []
    d = pd.read_parquet(FONTES[nome], columns=cols + extra)
    d = d.merge(pd.read_parquet(P / f"{nome}.parquet"), on="ponto_id").merge(pd.read_parquet(P / "cobertura_solo_30m" / f"{nome}.parquet"), on="ponto_id")
    if nome == "recife":
        r = pd.read_parquet(TV / "v6_evidencia_robusta/recife_avaliacao/referencia.parquet", columns=["ponto_id", "estado_E2_primario"])
        d = d.merge(r.rename(columns={"estado_E2_primario": "estado_primaria"}), on="ponto_id")
    if nome == "rmr":
        r = pd.read_parquet(TV / "v6_evidencia_robusta/rmr_avaliacao/referencia_a02.parquet")
        d = d.merge(r.rename(columns={"estado": "estado_primaria"}), on="ponto_id")
    if nome in COORTES_0:
        d["particao"] = "teste"
        d["grupo"] = np.where(d.classe_territorial.eq("fcu"), "FCU:" + d.CD_FCU.astype(str), "SET:" + d.CD_SETOR.astype(str))
    d = d[d.hand_m.notna() & d.slope_deg.notna() & d.twi_dinf.notna()].reset_index(drop=True)
    d = d.rename(columns={"ln_sca_twi": "ln_sca"})
    return d


def caracteristicas(d):
    import mt05_transferencia_sem_amostra as mt
    return mt.caracteristicas(d)


def escala_ext(modelo):
    pad = modelo.pad
    return pad.mu[FEAT], pad.sd[FEAT], pad.med[FEAT]


def z_externa(x, mu, sd, med):
    return ((x[FEAT].fillna(med) - mu) / sd).to_numpy()


def z_local(xa, xt):
    med = xa[FEAT].median()
    xa2, xt2 = xa[FEAT].fillna(med), xt[FEAT].fillna(med)
    mu, sd = xa2.mean(), xa2.std(ddof=0).replace(0, 1).fillna(1)
    za, zt = ((xa2 - mu) / sd).fillna(0).to_numpy(), ((xt2 - mu) / sd).fillna(0).to_numpy()
    return za, zt, sd


def executar():
    import v5_aprendizagem as v5
    from comum_tv import auc_mann_whitney
    prot = congelar8(D / "protocolo.json", PROTOCOLO)
    t0 = time.time()
    with (V8 / "q_externo/modelos.pkl").open("rb") as f:
        Q = pickle.load(f)
    with (MT / "v5/sem_amostra_v1/candidatas/modelos.pkl").open("rb") as f:
        QC = pickle.load(f)
    esc, gext, sem_areia = escolhas_externas()
    g8(D / "escolhas_externas.json", {"quando_utc": agora(), **esc, "protocolo_quando": prot["quando_utc"]})
    with (D / "modelos_externos_v8.pkl").open("wb") as f:
        pickle.dump({"GBM_ext": gext, "Q_ext_sem_areia": sem_areia}, f)
    qext = Q["Q_conexao|fora_BR"]
    mu, sd, med = escala_ext(qext)
    b_ext = np.array([dict(zip(qext.principais, qext.b))[c] for c in FEAT])
    qperm = Q["Q_conexao|permutado"]
    b_perm = np.array([dict(zip(qperm.principais, qperm.b))[c] for c in FEAT])
    ids = pd.read_csv(V4 / "ids_orcamentos.csv")
    fits, metr, comu = [], [], []
    for nome in COORTES_B + COORTES_0:
        d = coorte(nome)
        x = caracteristicas(d)
        te = d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])
        dt, xt = d[te].reset_index(drop=True), x[te].reset_index(drop=True)
        y = dt.estado_primaria.eq("positivo").astype(int).to_numpy()
        pred = dt[["ponto_id", "grupo", "classe_territorial"]].assign(y=y)
        sem = {"A0_hand": -dt.hand_m.to_numpy(), "menos_declive": -dt.slope_deg.to_numpy(), "twi": dt.twi_dinf.to_numpy()}
        sem["melhor_fisica_ext"] = sem[esc["melhor_fisica"]]
        sem["Q_congelado"] = QC["Q_conexao"].escore(xt)
        for b in ["terreno", "base", "conexao"]:
            sem[f"Q_ext_{b}"] = Q[f"Q_{b}|fora_BR"].escore(xt)
        sem["Q_extAS_conexao"] = Q["Q_conexao|fora_AS"].escore(xt)
        sem["Q_perm_conexao"] = Q["Q_conexao|permutado"].escore(xt)
        sem["Q_ext_sem_areia"] = sem_areia.escore(xt)
        sem["GBM_ext"] = gext.predict_proba(xt[FEAT])[:, 1]
        a1 = TV / f"v3/aprendizagem/predicoes_internacional_completo_{nome}.parquet"
        if a1.exists():
            sem["A1_v3"] = dt[["ponto_id"]].merge(pd.read_parquet(a1, columns=["ponto_id", "A1"]), on="ponto_id", how="left").A1.to_numpy()
        for k, v in sem.items():
            pred[k] = v.astype("float32")
        imput = {"areia_imputada_frac": float(xt.f_areia.isna().mean()), "areia_imputada_fcu": float(xt.f_areia[dt.classe_territorial.eq("fcu")].isna().mean())}
        etapa("v8_04_coorte_sem_rotulo", coorte=nome, pontos=len(dt), positivos=int(y.sum()), **imput)
        if nome in COORTES_B:
            pool = d.particao.eq("adaptacao")
            xa_all, da_all = x[pool], d[pool]
            idx_ad = dict(zip(da_all.ponto_id, da_all.index))
            zt_ext = z_externa(xt, mu, sd, med)
            acum = {}
            for (pol, B, sem_), q in ids[ids.territorio.eq(nome)].groupby(["politica", "B", "semente"]):
                ii = [idx_ad[p] for p in q.ponto_id if p in idx_ad]
                da, xa = d.loc[ii], x.loc[ii]
                ya = da.estado_primaria.eq("positivo").astype(int).to_numpy()
                obs = da.estado_primaria.isin(["positivo", "negativo"]).to_numpy()
                da, xa, ya = da[obs], xa[obs], ya[obs]
                duas = len(np.unique(ya)) == 2
                za_ext = z_externa(xa, mu, sd, med)
                za_loc, zt_loc, sd_loc = z_local(xa, xt)
                bracos = {}
                for tau in TAUS:
                    for nomeb, za, ztt, prior in [("L_estrito", za_loc, zt_loc, None), ("L_escext", za_ext, zt_ext, None),
                                                  ("P_ext", za_ext, zt_ext, b_ext), ("P_perm", za_ext, zt_ext, b_perm),
                                                  ("P_ext_escloc", za_loc, zt_loc, b_ext * sd_loc.to_numpy() / sd.to_numpy())]:
                        th, aprendeu = v5.ajustar(za, ya, prior, tau, balancear=True)
                        bracos[f"{nomeb}|t{tau:g}"] = (v5.escore(th, ztt), th, aprendeu)
                if duas and B >= 100:
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    w = np.where(ya == 1, len(ya) / (2 * ya.sum()), len(ya) / (2 * (len(ya) - ya.sum())))
                    gl = HistGradientBoostingClassifier(max_depth=2, max_iter=100, learning_rate=.1, min_samples_leaf=5,
                                                        monotonic_cst=MONO, random_state=0).fit(xa[FEAT], ya, sample_weight=w)
                    bracos["GBM_local"] = (gl.predict_proba(xt[FEAT])[:, 1], None, True)
                for k, (s, th, aprendeu) in bracos.items():
                    fits.append({"coorte": nome, "politica": pol, "B": int(B), "semente": int(sem_), "braco": k,
                                 "consultados": int(len(q)), "validos": int(obs.sum()), "positivos": int(ya.sum()), "duas_classes": bool(duas),
                                 "estado_ajuste": "ajustado_duas_classes" if aprendeu else ("copia_do_prior" if "P_" in k else "impossivel_uma_classe"),
                                 "theta": None if th is None else [float(v) for v in th]})
                    for pop in ["fcu", "baixa_renda", "comparador_urbano", "todas"]:
                        m = np.ones(len(y), bool) if pop == "todas" else dt.classe_territorial.eq(pop).to_numpy()
                        metr.append({"coorte": nome, "politica": pol, "B": int(B), "semente": int(sem_), "braco": k, "populacao": pop,
                                     "auc": auc_mann_whitney(y[m], s[m]), "positivos_teste": int(y[m].sum())})
                    f = dt.classe_territorial.eq("fcu").to_numpy()
                    for g_, a_ in v5.auc_por_grupo(y[f], s[f], dt.grupo.to_numpy()[f], 5).items():
                        comu.append({"coorte": nome, "politica": pol, "B": int(B), "semente": int(sem_), "braco": k, "grupo": g_, "auc": float(a_)})
                    acum.setdefault((pol, B, k), []).append(s)
            for (pol, B, k), ss in acum.items():
                pred[f"{k}|{pol}|B{B}"] = np.mean(ss, axis=0).astype("float32")
        g8(D / f"predicoes_{nome}.parquet", pred)
        print(nome, len(dt), int(y.sum()), round(time.time() - t0), flush=True)
    g8(D / "ajustes.parquet", pd.DataFrame(fits).assign(theta=lambda f: f.theta.map(json.dumps)))
    g8(D / "metricas_semente.parquet", pd.DataFrame(metr))
    g8(D / "auc_comunidade_semente.parquet", pd.DataFrame(comu))
    fi = pd.DataFrame(fits)
    g8(D / "execucao.json", {"quando_utc": agora(), "segundos": time.time() - t0, "ajustes": len(fi),
                             "estado_ajustes": fi.groupby(["coorte", "estado_ajuste"]).size().unstack(fill_value=0).to_dict(orient="index"),
                             "pesos_externos_sha256": sha_hex(V8 / "q_externo/modelos.pkl"), "ids_v4_sha256": sha_hex(V4 / "ids_orcamentos.csv")})
    etapa("v8_04_campanha", segundos=time.time() - t0, ajustes=len(fi))


if __name__ == "__main__":
    {"protocolo": lambda: congelar8(D / "protocolo.json", PROTOCOLO), "executar": executar}[sys.argv[1]]()
