"""v8: nova amostra probabilistica de revisao (RMPA, cheia de maio/2024) em linhagem separada do pacote v6.

O pacote v6 (215 pontos, extremos do escore, um ponto por setor, pesos aproximados) fica preservado como DIAGNOSTICO dos
extremos. Aqui o desenho e independente de qualquer escore:
  populacao   pontos RMPA v1 (60 m) em FCU, baixa renda ou comparador urbano, terreno valido, WorldCover != 80,
              com QUALQUER estado S2 (positivo, negativo, desconhecido, permanente, pre_desconhecido)
  estagio 1   unidade primaria = setor censitario; estratos = classe territorial; SRS sem reposicao de m_h de K_h setores
  estagio 2   SRS sem reposicao de n_i = min(N_i, 4) pontos dentro do setor sorteado
  inclusao    pi = (m_h / K_h) * (n_i / N_i); peso = 1/pi (exato sob o desenho)
  variancia   aproximacao com reposicao no 1o estagio (Horvitz-Thompson/razao, variancia entre unidades primarias)
Alocacao fixada antes do sorteio: FCU 50 setores, baixa renda 35, comparador 35 (sobreamostra FCU, declarada).
Suficiencia: simula o estimador de proporcao com uma variavel substituta (estado S2) para medir o erro-padrao esperado.
Nenhuma resposta humana e simulada; ninguem e contatado.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_comum import TV, V8, agora, congelar8, etapa, g8, sha_hex

D = V8 / "revisao_probabilistica"
V1R = TV / "v1/confirmacao_rmpa"
ALOC = {"fcu": 50, "baixa_renda": 35, "comparador_urbano": 35}
N_POR_SETOR = 4
SEMENTE = 20261022


def universo():
    g = pd.read_parquet(V1R / "grade.parquet", columns=["ponto_id", "lon", "lat", "CD_SETOR", "CD_FCU", "classe_territorial", "terreno_valido"])
    g = g.merge(pd.read_parquet(V1R / "referencia.parquet", columns=["ponto_id", "estado_primaria"]), on="ponto_id")
    g = g.merge(pd.read_parquet(TV / "v2/covariaveis/rmpa.parquet", columns=["ponto_id", "worldcover"]), on="ponto_id")
    return g[g.classe_territorial.isin(list(ALOC)) & g.terreno_valido & (g.worldcover > 0) & (g.worldcover != 80)].reset_index(drop=True)


def sortear(u, rng):
    sel = []
    for estr, q in u.groupby("classe_territorial"):
        setores = np.array(sorted(q.CD_SETOR.unique()))
        K, m = len(setores), min(ALOC[estr], len(setores))
        esc = rng.choice(setores, m, replace=False)
        for s in sorted(esc):
            z = q[q.CD_SETOR.eq(s)]
            N, n = len(z), min(N_POR_SETOR, len(z))
            pts = z.iloc[np.sort(rng.choice(N, n, replace=False))]
            sel.append(pts.assign(estrato=estr, K_h=K, m_h=m, N_i=N, n_i=n, pi=(m / K) * (n / N)))
    s = pd.concat(sel, ignore_index=True)
    s["peso"] = 1 / s.pi
    return s


def estimar(s, var):
    """Proporcao ponderada (razao) e EP com reposicao no 1o estagio, estratificado."""
    v = s[var].astype(float)
    tot = (s.peso * v).sum() / s.peso.sum()
    var_t = 0.0
    W = s.peso.sum()
    for _, q in s.groupby("estrato"):
        u = q.assign(r=q.peso * (q[var].astype(float) - tot)).groupby("CD_SETOR").r.sum()
        m = len(u)
        if m > 1:
            var_t += m / (m - 1) * ((u - u.mean()) ** 2).sum()
    return float(tot), float(np.sqrt(var_t) / W)


def main():
    plano = congelar8(D / "plano.json", {"versao": "v8-revisao-1", "alocacao_setores": ALOC, "pontos_por_setor": N_POR_SETOR,
                                         "semente": SEMENTE, "estados_incluidos": "todos", "usa_escore": False,
                                         "inclusao": "pi=(m_h/K_h)*(n_i/N_i)", "variancia": "entre unidades primarias, com reposicao no 1o estagio",
                                         "pergunta_ao_revisor": "agua de inundacao visivel no ponto (raio 30 m) entre 2024-05-04 e 2024-05-08? agua visivel | sem agua visivel | nao observavel; fonte, data, resolucao",
                                         "pacote_v6": "preservado como diagnostico dos extremos; nao reutilizado"})
    u = universo()
    s = sortear(u, np.random.default_rng(SEMENTE))
    # conferencia: soma dos pesos estima o tamanho da populacao por estrato
    conf = {e: {"N_universo": int((u.classe_territorial == e).sum()), "soma_pesos": float(q.peso.sum())} for e, q in s.groupby("estrato")}
    s = s.iloc[np.random.default_rng(SEMENTE + 1).permutation(len(s))].reset_index(drop=True)
    s["revisao_id"] = [f"REV8_{i:04d}" for i in range(len(s))]
    chave = s[["revisao_id", "ponto_id", "estrato", "CD_SETOR", "CD_FCU", "K_h", "m_h", "N_i", "n_i", "pi", "peso", "estado_primaria"]]
    form = s[["revisao_id", "lon", "lat"]].assign(data_alvo="2024-05-06", janela="2024-05-04/2024-05-08", resposta="", fonte_imagem="",
                                                  data_imagem="", resolucao_m="", observacao="")
    trilha = s[["revisao_id"]].assign(revisor_1="", resposta_1="", revisor_2="", resposta_2="", adjudicador="", decisao_final="", motivo="")
    g8(D / "chave_selada.parquet", chave)
    g8(D / "formulario_cego.csv", form)
    g8(D / "trilha_adjudicacao.csv", trilha)
    # suficiencia: substituto = indicador de estado S2; 200 sorteios do mesmo desenho
    sim = []
    u2 = u.assign(pos=u.estado_primaria.eq("positivo").astype(float), desc=u.estado_primaria.eq("desconhecido").astype(float))
    for k in range(200):
        a = sortear(u2, np.random.default_rng(1000 + k))
        sim.append({"pos": estimar(a, "pos")[0], "desc": estimar(a, "desc")[0], "ep_pos": estimar(a, "pos")[1]})
    sim = pd.DataFrame(sim)
    verdade = {"pos": float(u2.pos.mean()), "desc": float(u2.desc.mean())}
    g8(D / "suficiencia.json", {"quando_utc": agora(), "pontos": len(s), "setores": int(s.CD_SETOR.nunique()), "conferencia_pesos": conf,
                                "substituto_valor_populacional": verdade,
                                "simulacao_200": {"vies_pos": float(sim.pos.mean() - verdade["pos"]), "dp_pos": float(sim.pos.std()),
                                                  "ep_medio_estimado_pos": float(sim.ep_pos.mean()), "vies_desc": float(sim.desc.mean() - verdade["desc"]),
                                                  "dp_desc": float(sim.desc.std())},
                                "leitura": "EP de ~dp_pos para uma proporcao com prevalencia como a do S2; concordancia/erro de rotulo terao EP da mesma ordem"})
    g8(D / "estado.json", {"quando_utc": agora(), "plano_quando": plano["quando_utc"], "pontos": len(s), "respostas_recebidas": 0, "revisor": None,
                           "estado": "preparado_aguardando_revisor_humano", "sha_chave": sha_hex(D / "chave_selada.parquet"),
                           "sha_formulario": sha_hex(D / "formulario_cego.csv"), "nota": "ninguem foi contatado; nenhuma resposta simulada"})
    etapa("v8_07_amostra", pontos=len(s))
    print(len(s), conf, verdade, sim.describe().round(4).to_dict())


if __name__ == "__main__":
    main()
