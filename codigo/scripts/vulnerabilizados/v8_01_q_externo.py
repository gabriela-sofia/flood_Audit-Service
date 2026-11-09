"""v8: o contrato fisico Q (MT05/MT11) reajustado sem o Brasil.

Achado que motiva: a populacao de ajuste MT01/MT03 usada no Q congelado contem CEMS:EMSR720 (RS, maio/2024, mesmo
episodio da coorte RMPA) e ~73 eventos GFD dentro do Brasil. O Q congelado aplicado a RMPA nao e transferencia sem
rotulo local. Aqui, com o MESMO estimador (e1.Logistica, intercepto por evento, lam herdado), a MESMA amostra MT03-A
(tetos 400 por evento-classe, semente 20261010) e os MESMOS atributos:

  reproducao   desenvolvimento completo -> deve reproduzir os pesos congelados (QA do pipeline)
  fora_BR      so eventos com todos os pontos fora do Brasil (malha IBGE, auditoria v7)
  fora_AS      fora_BR e fora da caixa da America do Sul (lon -82..-34, lat -56..13): sem bacias compartilhadas
  permutado    fora_BR com classe permutada dentro de cada evento (controle de desenvolvimento)
Bracos: Q_terreno (hand, sca, declive), Q_base (+construido, areia), Q_conexao (+dist. drenagem/canal, interacoes).
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v8_comum import MT, V7, V8, agora, congelar8, etapa, g8, ler, sha_hex

D = V8 / "q_externo"
BRACOS = ["Q_terreno", "Q_base", "Q_conexao"]
AS = {"lon": (-82.0, -34.0), "lat": (-56.0, 13.0)}


def exportar(m):
    return {"coeficientes": dict(zip(m.principais, m.b.tolist())),
            "interacoes": [{"colunas": [a, b], "peso": float(v)} for (a, b), v in zip(m.interacoes, m.g)],
            "padronizador": {"colunas": m.pad.mu.index.tolist(), "media": m.pad.mu.to_dict(), "desvio": m.pad.sd.to_dict(),
                             "mediana": m.pad.med.to_dict()},
            "convergiu": bool(m.convergiu), "iteracoes": int(m.iteracoes)}


def main():
    import mt05_transferencia_sem_amostra as mt
    congelar8(D / "protocolo.json", {"versao": "v8-q-1", "estimador": "mt01_experimento.Logistica", "amostra": "MT03-A (teto 400, semente 20261010)",
                                     "variantes": ["reproducao", "fora_BR", "fora_AS", "permutado"], "bracos": BRACOS,
                                     "regra_fora_BR": "evento com todos os pontos exterior_auditado=True (v7)",
                                     "regra_fora_AS": f"fora_BR e todos os pontos fora da caixa {AS}",
                                     "permutado": "classe permutada dentro do evento, semente 20261022",
                                     "lam": "herdado de mt03 candidatas_congeladas.json"})
    t0 = time.time()
    d = mt.e3.carregar()
    lam = ler(mt.e3.SAIDA / "candidatas_congeladas.json")["lam"]
    tr = mt.e1.amostrar(d, mt.e3.CONFIG["teto_ajuste_por_evento_classe"], mt.e3.CONFIG["semente"])
    geo = pd.read_parquet(V7 / "v7_02_contextos_tempo/geografia_pontos.parquet", columns=["ponto_id", "exterior_auditado", "no_brasil"])
    tr = tr.merge(geo, on="ponto_id", how="left")
    tr.index = range(len(tr))
    fora_as_pt = ~(tr.lon.between(*AS["lon"]) & tr.lat.between(*AS["lat"]))
    ev = tr.assign(fas=fora_as_pt).groupby("evento_id").agg(ext=("exterior_auditado", lambda s: bool(s.fillna(False).all())),
                                                               fas=("fas", "all"), fonte=("fonte", "first"))
    x = mt.caracteristicas(tr)
    rng = np.random.default_rng(20261022)
    subconj = {"reproducao": tr.index.to_numpy(),
               "fora_BR": tr.index[tr.evento_id.isin(ev.index[ev.ext])].to_numpy(),
               "fora_AS": tr.index[tr.evento_id.isin(ev.index[ev.ext & ev.fas])].to_numpy()}
    y_perm = tr.classe.to_numpy(float).copy()
    for _, g in tr.loc[subconj["fora_BR"]].groupby("evento_id"):
        y_perm[g.index] = rng.permutation(y_perm[g.index])
    exp, modelos, resumo = {}, {}, {}
    for var, idx in list(subconj.items()) + [("permutado", subconj["fora_BR"])]:
        sub = tr.loc[idx]
        y = (y_perm if var == "permutado" else tr.classe.to_numpy(float))[idx]
        resumo[var] = {"linhas": int(len(sub)), "eventos": int(sub.evento_id.nunique()), "unidades": int(sub.unidade.nunique()),
                       "positivos": int(y.sum()), "por_fonte": sub.fonte.value_counts().to_dict(),
                       "contem_EMSR720": bool((sub.evento_id == "CEMS:EMSR720").any()),
                       "pontos_no_brasil": int(sub.no_brasil.fillna(False).sum())}
        for b in BRACOS:
            m = mt.e1.Logistica(mt.BRACOS[b], mt.INTERACOES.get(b, ()), lam_p=lam["p"], lam_i=lam["i"])
            m.fit(x.loc[idx], y, sub.evento_id.to_numpy())
            if not m.convergiu:
                raise ValueError(f"nao convergiu {var} {b}")
            modelos[f"{b}|{var}"] = m
            exp[f"{b}|{var}"] = exportar(m)
            print(var, b, m.iteracoes, flush=True)
    congelado = ler(MT / "v5/sem_amostra_v1/candidatas/pesos_e_padronizacao.json")
    repro = {b: max(abs(exp[f"{b}|reproducao"]["coeficientes"][k] - v) for k, v in congelado[b]["coeficientes"].items())
             for b in ["Q_base", "Q_conexao"]}
    D.mkdir(parents=True, exist_ok=True)
    with (D / "modelos.pkl").open("wb") as f:
        pickle.dump(modelos, f)
    g8(D / "pesos.json", exp)
    g8(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "lam": lam, "subconjuntos": resumo,
                              "reproducao_max_dif_coef": repro, "pesos_sha256": sha_hex(D / "modelos.pkl"),
                              "eventos_fora_BR": sorted(ev.index[ev.ext]), "eventos_fora_AS": sorted(ev.index[ev.ext & ev.fas])})
    etapa("v8_01_q_externo", segundos=time.time() - t0, reproducao=repro)
    print(resumo, repro)


if __name__ == "__main__":
    main()
