"""v10 Frente II: ciclo integrado consulta -> evidencia revelada -> suporte -> atualizacao offset -> ganho/retencao,
rastreavel por lote, sobre o NUCLEO LIMPO (Qsep_sel_fis: lam e coeficientes so do papel fisico).

Pools = v9 F (todos os pontos dos agrupamentos de adaptacao, QUALQUER estado; desconhecido/permanente gastam
orcamento e nao viram rotulo). Rotulos escondidos ate a consulta; teste (particao v3) nunca no pool.
Estrategias (lotes de 25 ate 400; 6 sementes; desempate = ordem estavel do pool):
  congelado               nenhuma atualizacao (referencia)
  uniforme_gate           sorteio uniforme + filtro de suporte + offset
  margem_v9_gate          REPRODUCAO da v9: menor |posto - 0,8| entre candidatos livres (posto do escore atual) + filtro
  margem_v9_sem_gate      a mesma consulta, atualiza sempre que houver duas classes (isola o filtro)
  margem_p90_gate         variante: distancia ao percentil 90
  margem_explora_gate     metade uniforme, metade margem_v9, + filtro
  diversidade_gate        k-centro no espaco padronizado + filtro
Filtro: >= 5 positivos, >= 5 negativos, >= 3 agrupamentos com positivo; sem suporte a funcao do Q e conservada.
Registros: consultas.parquet (IDs, ordem, escore e posto previos, estado revelado, grupo, custo), lotes.parquet
(theta, c, atualizou, motivo, contagens, AUC teste todas/FCU em TODO o teste admissivel, AUC nas 6 unidades de retencao).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v10_comum import COORTES_B, V9, V10, agora, auc, congelar10, etapa, g10, nucleo_limpo, papeis

D = V10 / "ii_ciclo"
ESTR = ["congelado", "uniforme_gate", "margem_v9_gate", "margem_v9_sem_gate", "margem_p90_gate", "margem_explora_gate", "diversidade_gate"]
LOTE, MAXB, TAU = 25, 400, 25.0
GATE = {"positivos": 5, "negativos": 5, "grupos_com_positivo": 3}


def main():
    import mt01_experimento as e1
    from v8_04_campanha import FEAT, caracteristicas, coorte, externo, z_externa
    from v9_f_aquisicao import k_centro
    from v9_p0_continuidade import ajustar_offset
    congelar10(D / "protocolo.json", {"versao": "v10-ii-2", "estrategias": ESTR, "lote": LOTE, "max": MAXB, "tau": TAU, "gate": GATE, "nucleo": "Qsep_sel_fis",
                                      "margem_v9": "argsort estavel de |posto_pct(escore entre livres) - 0,8|", "sementes": [1, 2, 3, 4, 5, 6],
                                      "pool": "v9 F menos IDs da particao teste (defeito herdado corrigido)"})
    t0 = time.time()
    Q = nucleo_limpo()
    mu, sd, med = Q.pad.mu[FEAT], Q.pad.sd[FEAT], Q.pad.med[FEAT]
    tr, x, _ = externo()
    ret = tr.unidade.map(papeis()).eq("retencao").to_numpy()
    R, qR, zR = tr[ret], Q.escore(x[ret]), z_externa(x[ret], mu, sd, med)
    consultas, lotes, teste_removidos_do_pool = [], [], {}
    for c in COORTES_B:
        d = coorte(c)
        P_all = pd.read_parquet(V9 / f"f_aquisicao/pool_{c}.parquet")
        # defeito herdado da v9 F: o pool tomou a particao do 1o ponto de cada agrupamento, e em Curitiba 3 FCU tem
        # pontos de teste e de adaptacao; IDs de teste saem do pool aqui
        vazados = P_all.ponto_id.isin(set(d.ponto_id[d.particao.eq("teste")]))
        teste_removidos_do_pool[c] = int(vazados.sum())
        P_all = P_all[~vazados].reset_index(drop=True)
        xp = caracteristicas(P_all)
        qp_all, zp_all = Q.escore(xp), z_externa(xp, mu, sd, med)
        xt = caracteristicas(d)
        te = (d.particao.eq("teste") & d.estado_primaria.isin(["positivo", "negativo"])).to_numpy()
        qt, zt = Q.escore(xt)[te], z_externa(xt, mu, sd, med)[te]
        yt = d.estado_primaria.eq("positivo").to_numpy()[te]
        ft = d.classe_territorial.eq("fcu").to_numpy()[te]
        for pool in ["todas", "fcu"]:
            m = np.ones(len(P_all), bool) if pool == "todas" else P_all.classe_territorial.eq("fcu").to_numpy()
            P, qp, zp = P_all[m].reset_index(drop=True), qp_all[m], zp_all[m]
            est = P.estado_primaria.to_numpy()
            yv = np.where(est == "positivo", 1, np.where(est == "negativo", 0, -1))
            gr = P.grupo.to_numpy()
            for estr in ESTR:
                for sem in range(1, 7):
                    rng = np.random.default_rng(sem * 100 + ESTR.index(estr))
                    usado = np.zeros(len(P), bool)
                    th, cc, ativo = np.zeros(len(FEAT)), 0.0, False
                    for lote, B in enumerate(range(LOTE, MAXB + 1, LOTE)):
                        livres = np.where(~usado)[0]
                        s = qp + (zp @ th + cc if ativo else 0)
                        postos = pd.Series(s[livres]).rank(pct=True).to_numpy()
                        if estr in ("uniforme_gate", "congelado"):
                            esc = rng.choice(livres, LOTE, replace=False)
                        elif estr in ("margem_v9_gate", "margem_v9_sem_gate"):
                            esc = livres[np.argsort(np.abs(postos - .8), kind="stable")[:LOTE]]
                        elif estr == "margem_p90_gate":
                            esc = livres[np.argsort(np.abs(postos - .9), kind="stable")[:LOTE]]
                        elif estr == "margem_explora_gate":
                            a = rng.choice(livres, LOTE // 2, replace=False)
                            resto = np.setdiff1d(livres, a)
                            pr = pd.Series(s[resto]).rank(pct=True).to_numpy()
                            esc = np.r_[a, resto[np.argsort(np.abs(pr - .8), kind="stable")[:LOTE - len(a)]]]
                        else:
                            sub = livres if len(livres) <= 30000 else rng.choice(livres, 30000, replace=False)
                            esc = sub[k_centro(zp[sub], zp[usado], LOTE, rng)]
                        esc = np.asarray(esc, int)
                        pos_l = {i: postos[k] for k, i in enumerate(livres)}
                        for ordem, i in enumerate(esc):
                            consultas.append({"coorte": c, "pool": pool, "estrategia": estr, "semente": sem, "lote": lote, "ordem": ordem, "ponto_id": P.ponto_id.iat[i],
                                              "escore_previo": float(s[i]), "posto_previo": float(pos_l[i]), "estado_revelado": est[i], "grupo": gr[i],
                                              "classe_territorial": P.classe_territorial.iat[i], "custo": 1})
                        usado[esc] = True
                        ok = usado & (yv >= 0)
                        ya = yv[ok]
                        gp = set(gr[ok][ya == 1])
                        duas = len(np.unique(ya)) == 2
                        suporte = duas and ya.sum() >= GATE["positivos"] and (1 - ya).sum() >= GATE["negativos"] and len(gp) >= GATE["grupos_com_positivo"]
                        if estr == "congelado":
                            motivo = "congelado"
                        elif estr == "margem_v9_sem_gate":
                            motivo = "atualizado" if duas else "uma_classe"
                        else:
                            motivo = "atualizado" if suporte else ("uma_classe" if not duas else "suporte_insuficiente")
                        if motivo == "atualizado":
                            th, cc, ativo = ajustar_offset(zp[ok], ya, qp[ok], TAU)
                        stt = qt + (zt @ th + cc if ativo else 0)
                        sR = qR + (zR @ th + cc if ativo else 0)
                        lotes.append({"coorte": c, "pool": pool, "estrategia": estr, "semente": sem, "lote": lote, "consultados": int(usado.sum()),
                                      "validos": int(ok.sum()), "sem_rotulo": int((usado & (yv < 0)).sum()), "positivos": int(ya.sum()),
                                      "grupos_com_positivo": len(gp), "atualizou_neste_lote": motivo == "atualizado", "ativo": bool(ativo), "motivo": motivo,
                                      "theta": json.dumps([float(v) for v in th]), "c": cc,
                                      "auc_teste_todas": auc(yt, stt), "auc_teste_fcu": auc(yt[ft], stt[ft]),
                                      "auc_retencao": e1.auc_media_por_unidade(R, sR)})
            print(c, pool, round(time.time() - t0), flush=True)
            g10(D / "lotes.parquet", pd.DataFrame(lotes))
    g10(D / "consultas.parquet", pd.DataFrame(consultas))
    g10(D / "lotes.parquet", pd.DataFrame(lotes))
    lt = pd.DataFrame(lotes)
    r = lt.groupby(["coorte", "pool", "estrategia", "consultados"])[["positivos", "sem_rotulo", "ativo", "auc_teste_todas", "auc_teste_fcu", "auc_retencao"]].mean().reset_index()
    g10(D / "curvas.csv", r)
    g10(D / "resultado.json", {"quando_utc": agora(), "segundos": time.time() - t0, "consultas": len(consultas), "lotes": len(lotes),
                               "teste_removidos_do_pool_v9": teste_removidos_do_pool})
    etapa("ii_ciclo", segundos=time.time() - t0)


if __name__ == "__main__":
    main()
