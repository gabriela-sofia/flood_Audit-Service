"""MT10, sensibilidade pós-hoc declarada: frequência SAR só com datas ANTERIORES ao evento.

`freq_fora` exclui as janelas de evento e 60 dias de transição, mas inclui datas depois
do evento. Água ou mudança de superfície causada pela cheia e persistente além disso
entraria em F e se pareceria com previsão. Aqui F_pre usa só as datas "fora" anteriores
à primeira data de evento/transição da pilha do bloco. Mesmos pontos, braços e estatística
do MT10; não substitui a primária (foi pensada depois de ver o resultado).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import mt10_teto_transdutivo as t10

e = t10.e
pilha = t10.pilha
N_PRE_MIN = 5


def freq_pre(bloco, regra):
    pasta = "frequencia_e1" if regra == "e1" else "frequencia"
    z = np.load(pilha.SAIDA / pasta / f"{bloco}.npz", allow_pickle=True)
    j = np.asarray(z["janela"]).astype(str)
    if (j != "fora").any():
        primeira = int(np.flatnonzero(j != "fora")[0])
    else:
        primeira = len(j)
    det = z["deteccao"][:primeira].astype(float)
    det[det == 255] = np.nan  # sem observação na data
    n = np.isfinite(det).sum(0)
    with np.errstate(invalid="ignore"):
        f = np.nanmean(np.where(np.isfinite(det), det, np.nan), 0) if primeira else np.full(det.shape[1:], np.nan)
    return f, n, primeira


def tabela_pre_evento():
    """Pontos do MT10 com F só de datas anteriores ao evento; devolve (tabela, datas por bloco)."""
    t = pd.read_parquet(t10.SAIDA / "pontos.parquet")
    t = t[(t.classe_pixel == 1) & (t.n_fora >= t10.CFG["n_fora_min"]) & t.F.notna() & t.utilizavel].copy()
    blocos = {b["bloco"]: b for b in e.ler(pilha.SAIDA / "blocos.json")["blocos"]}
    from pyproj import Transformer
    vals, ns, datas = [], [], {}
    for _, r in t.iterrows():
        b = blocos[r.bloco]
        if r.bloco not in datas:
            datas[r.bloco] = freq_pre(r.bloco, r.regra)
        f, n, _ = datas[r.bloco]
        lin, col, ok = t10.pixel(np.array([r.lat]), np.array([r.lon]), b, Transformer.from_crs("EPSG:4326", b["crs"], always_xy=True))
        vals.append(float(f[lin[0], col[0]]) if ok[0] else np.nan)
        ns.append(int(n[lin[0], col[0]]) if ok[0] else 0)
    t["F"] = vals
    t["n_pre"] = ns
    return t[(t.n_pre >= N_PRE_MIN) & t.F.notna()], datas


def main():
    t10.preparar()
    sub, datas = tabela_pre_evento()
    s = t10.escores(sub)
    tab = pd.DataFrame(t10.medir(s, "pre_evento"))
    u = tab.groupby(["destino", "modelo"]).auc.mean().unstack()
    r = {"tipo": "MT10_sensibilidade_pos_hoc_pre_evento", "quando_utc": e.agora_utc(),
         "protocolo_mt10_sha256": e.sha256_arquivo(t10.SAIDA / "protocolo.json"),
         "pontos": len(sub), "componentes": int(sub.unidade.nunique()), "eventos": int(sub.evento_id.nunique()),
         "datas_pre_por_bloco": {k: v[2] for k, v in datas.items()},
         "blocos_sem_data_pre": int(sum(v[2] == 0 for v in datas.values())),
         "contrastes": {f"{a}-Q": t10.contraste(u, a, "Q") for a in t10.BRACOS[1:]},
         "auc_media": u.mean().to_dict(), "pos_hoc": True}
    e.escrever(t10.SAIDA / "sensibilidade_pre_evento.json", r)
    print(json.dumps({k: r[k] for k in ("pontos", "componentes", "eventos", "blocos_sem_data_pre")} |
                     {"contrastes": {k: (v["media"], v.get("ic"), v["veredito"], v["n"]) for k, v in r["contrastes"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
