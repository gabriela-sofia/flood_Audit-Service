"""v8: amostra probabilistica v2 -- dois estagios com PPS sistematico no 1o estagio.

Motivo (registrado antes de qualquer revisao): a v1 (`revisao_probabilistica/`, SRS de setores) e nao viesada mas
imprecisa -- simulada com o estado S2 como substituto, a prevalencia 0,063 teve DP 0,075 entre sorteios, porque os setores
tem tamanhos muito desiguais. A v1 fica preservada.
  estagio 1  setores por PPS sistematico (tamanho N_i em pontos) dentro de cada estrato social, ordem aleatoria com semente;
             setores com N_i >= N_h/m entram com certeza (pi_1 = 1) e o restante e redistribuido
  estagio 2  SRS de n_i = min(4, N_i) pontos
  inclusao   pi = pi_1(setor) * n_i / N_i, com pi_1 = m' N_i / N'_h nos nao-certeza (autoponderado no estrato)
Mesma populacao, mesma alocacao (FCU 50 / baixa renda 35 / comparador 35) e mesmas perguntas da v1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_07_amostra_probabilistica import ALOC, N_POR_SETOR, estimar, universo
from v8_comum import V8, agora, congelar8, etapa, g8, sha_hex

D = V8 / "revisao_probabilistica_v2"
SEMENTE = 20261023


def pps(tam: pd.Series, m: int, rng) -> dict:
    """PPS sistematico com unidades de certeza; devolve {setor: pi_1}."""
    tam = tam.sample(frac=1, random_state=int(rng.integers(1 << 31)))
    certos = {}
    resto = tam.copy()
    while True:
        k = m - len(certos)
        lim = resto.sum() / k if k > 0 else np.inf
        novos = resto[resto >= lim]
        if novos.empty or k <= 0:
            break
        for s in novos.index:
            certos[s] = 1.0
        resto = resto.drop(novos.index)
    k = m - len(certos)
    out = dict(certos)
    if k > 0 and len(resto):
        passo = resto.sum() / k
        acum = resto.cumsum().to_numpy()
        pontos = rng.uniform(0, passo) + passo * np.arange(k)
        idx = np.searchsorted(acum, pontos, side="right")
        for i in idx:
            s = resto.index[i]
            out[s] = float(k * resto[s] / resto.sum())
    return out


def sortear(u, rng):
    sel = []
    for estr, q in u.groupby("classe_territorial"):
        tam = q.groupby("CD_SETOR").size()
        esc = pps(tam, ALOC[estr], rng)
        for s, p1 in esc.items():
            z = q[q.CD_SETOR.eq(s)]
            N, n = len(z), min(N_POR_SETOR, len(z))
            pts = z.iloc[np.sort(rng.choice(N, n, replace=False))]
            sel.append(pts.assign(estrato=estr, pi_1=p1, N_i=N, n_i=n, pi=p1 * n / N, certeza=p1 >= 1))
    s = pd.concat(sel, ignore_index=True)
    s["peso"] = 1 / s.pi
    return s


def main():
    plano = congelar8(D / "plano.json", {"versao": "v8-revisao-2", "alocacao_setores": ALOC, "pontos_por_setor": N_POR_SETOR, "semente": SEMENTE,
                                         "estagio_1": "PPS sistematico por tamanho com unidades de certeza", "usa_escore": False,
                                         "motivo": "v1 (SRS de setores) nao viesada mas DP 0,075 na prevalencia substituta 0,063; v1 preservada",
                                         "pergunta_ao_revisor": "igual a v1"})
    u = universo()
    s = sortear(u, np.random.default_rng(SEMENTE))
    conf = {e: {"N_universo": int((u.classe_territorial == e).sum()), "soma_pesos": float(q.peso.sum()), "setores": int(q.CD_SETOR.nunique()),
                "certeza": int(q[q.certeza].CD_SETOR.nunique())} for e, q in s.groupby("estrato")}
    s = s.iloc[np.random.default_rng(SEMENTE + 1).permutation(len(s))].reset_index(drop=True)
    s["revisao_id"] = [f"REV8B_{i:04d}" for i in range(len(s))]
    g8(D / "chave_selada.parquet", s[["revisao_id", "ponto_id", "estrato", "CD_SETOR", "CD_FCU", "pi_1", "N_i", "n_i", "pi", "peso", "certeza", "estado_primaria"]])
    g8(D / "formulario_cego.csv", s[["revisao_id", "lon", "lat"]].assign(data_alvo="2024-05-06", janela="2024-05-04/2024-05-08", resposta="",
                                                                          fonte_imagem="", data_imagem="", resolucao_m="", observacao=""))
    g8(D / "trilha_adjudicacao.csv", s[["revisao_id"]].assign(revisor_1="", resposta_1="", revisor_2="", resposta_2="", adjudicador="", decisao_final="", motivo=""))
    u2 = u.assign(pos=u.estado_primaria.eq("positivo").astype(float), desc=u.estado_primaria.eq("desconhecido").astype(float))
    sim = []
    for k in range(200):
        a = sortear(u2, np.random.default_rng(5000 + k))
        sim.append({"pos": estimar(a, "pos")[0], "desc": estimar(a, "desc")[0], "ep_pos": estimar(a, "pos")[1], "ep_desc": estimar(a, "desc")[1]})
    sim = pd.DataFrame(sim)
    verdade = {"pos": float(u2.pos.mean()), "desc": float(u2.desc.mean())}
    g8(D / "suficiencia.json", {"quando_utc": agora(), "pontos": len(s), "conferencia_pesos": conf, "substituto_valor_populacional": verdade,
                                "simulacao_200": {"vies_pos": float(sim.pos.mean() - verdade["pos"]), "dp_pos": float(sim.pos.std()),
                                                  "ep_medio_pos": float(sim.ep_pos.mean()), "vies_desc": float(sim.desc.mean() - verdade["desc"]),
                                                  "dp_desc": float(sim.desc.std()), "ep_medio_desc": float(sim.ep_desc.mean())}})
    g8(D / "estado.json", {"quando_utc": agora(), "plano_quando": plano["quando_utc"], "pontos": len(s), "respostas_recebidas": 0, "revisor": None,
                           "estado": "preparado_aguardando_revisor_humano", "sha_chave": sha_hex(D / "chave_selada.parquet"),
                           "nota": "ninguem foi contatado; nenhuma resposta simulada; v1 e pacote v6 preservados"})
    etapa("v8_07b_amostra_pps", pontos=len(s))
    print(len(s), conf, verdade, sim.describe().round(4).loc[["mean", "std"]].to_dict())


if __name__ == "__main__":
    main()
