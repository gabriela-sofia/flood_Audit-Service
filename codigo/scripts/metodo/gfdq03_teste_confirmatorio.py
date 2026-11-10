"""O UNICO teste confirmatorio: a comparacao ja congelada, em eventos GFD que nenhuma analise viu.

-------------------------------------------------------------------------
O QUE E CONFIRMATORIO AQUI, E O QUE NAO E
-------------------------------------------------------------------------

Confirmatorio: as comparacoes M1 x M0 ("aprender pesos supera a regra de
HAND?") e M3 x M1 ("nao linearidade acrescenta?") do pre-registro da
`generalizacao_por_reservas` (hash gravado abaixo), com os mesmos modelos,
hiperparametros, regra de decisao (Wilcoxon pareado sobre a AUC DENTRO do
evento, relevancia 0,01, alfa 0,05, minimo de 8 eventos) e o mesmo `veredito`.

Treino: TODAS as linhas GFD de planicie conhecidas que a comparacao congelada
usa (`carregar`), ajuste unico. Teste: os eventos de `gfd-quarentena`, que nao
dividem evento nem tile de 1 grau com o treino nem entre si.

Fora do confirmatorio, e declarado antes: M2/M4 (cobertura/solo) nao entram,
porque os eventos em quarentena nao tem essas camadas derivadas; a calibracao
na faixa (com a area por classe medida na selecao) e DESCRITIVA.

-------------------------------------------------------------------------
UMA VEZ SO
-------------------------------------------------------------------------

`--pre-registrar` grava o plano com hash antes da selecao. `--executar` roda
uma vez, grava `executado.json` e se recusa a rodar de novo: refazer o teste
depois de ver o resultado o tornaria exploratorio.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade", "geo"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

from proveniencia import agora_utc, sha256_json  # noqa: E402

Q = REPO / "local_runs" / "gfd-quarentena"
COMPARACOES_CONFIRMATORIAS = ("M1_logistica_terreno|M0_regra_hand", "M3_gbm_terreno|M1_logistica_terreno")
MODELOS_USADOS = ("M0_regra_hand", "M1_logistica_terreno", "M3_gbm_terreno")


def plano() -> dict:
    import generalizacao_por_reservas as g
    return {"comparacao_congelada": "generalizacao_por_reservas", "pre_registro_da_comparacao": sha256_json(g.pre_registro()),
            "comparacoes_confirmatorias": list(COMPARACOES_CONFIRMATORIAS), "modelos": list(MODELOS_USADOS),
            "treino": "generalizacao_por_reservas.carregar(), fonte gfd, todas as linhas, ajuste unico (semente da comparacao)",
            "teste": "pontos de gfd-quarentena com mecanismo fluvial, classe_relevo PLANO_OU_ONDULADO e as 4 variaveis de terreno",
            "decisao": "generalizacao_por_reservas.comparar + veredito, sem alteracao",
            "fora_do_confirmatorio": ["M2/M4 cobertura/solo (camadas nao derivadas na quarentena)",
                                      "calibracao na faixa com a area da selecao (descritiva)"],
            "execucoes_permitidas": 1}


def pre_registrar() -> str:
    Q.mkdir(parents=True, exist_ok=True)
    alvo = Q / "pre_registro_avaliacao.json"
    p = plano()
    h = sha256_json(p)
    if alvo.exists():
        antigo = json.loads(alvo.read_text(encoding="utf-8"))
        if antigo["hash"] != h:
            raise RuntimeError(f"plano de avaliacao mudou depois do pre-registro ({antigo['hash']} -> {h})")
        return h
    alvo.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "plano": p}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return h


def executar() -> dict:
    import generalizacao_por_reservas as g
    feito = Q / "executado.json"
    if feito.exists():
        raise SystemExit(f"o teste confirmatorio ja rodou ({feito}); rodar de novo o tornaria exploratorio")
    h = pre_registrar()
    d, origem, _ = g.carregar()
    tr = d[d.fonte == "gfd"]
    te = pd.read_parquet(Q / "pontos_quarentena.parquet")
    te = te[(te.mecanismo == "FLUVIAL_ENXURRADA") & (te.classe_relevo == "PLANO_OU_ONDULADO")]
    te = te.dropna(subset=list(g.TERRENO)).reset_index(drop=True)
    assert not set(te.evento_id) & set(tr.evento_id), "evento em quarentena aparece no treino"
    import reservas_examinadas as rx
    ja = set(te.evento_id) & rx.examinados_depois("gfd")
    if ja:
        raise SystemExit(f"eventos ja examinados apos o congelamento, nao podem confirmar: {sorted(ja)}")
    pred = {}
    for m in MODELOS_USADOS:
        fn = g.ajustar_modelo(m, tr)
        pred[m] = fn(te)
    rng = np.random.default_rng(g.SEMENTE)
    comps = {}
    for par in COMPARACOES_CONFIRMATORIAS:
        a, b = par.split("|")
        c = g.comparar(te, pred[a], pred[b], rng)
        comps[par] = {**c, "veredito": g.veredito(c)}
    r = {"tipo": "teste_confirmatorio_gfd", "quando_utc": agora_utc(), "pre_registro_avaliacao": h,
         "pre_registro_selecao": json.loads((Q / "pre_registro_selecao.json").read_text(encoding="utf-8"))["hash"],
         "dataset_de_treino": origem, "linhas_treino": int(len(tr)), "eventos_treino": int(tr.evento_id.nunique()),
         "linhas_teste": int(len(te)), "eventos_teste": int(te.evento_id.nunique()),
         "eventos_teste": sorted(te.evento_id.unique().tolist()),
         "ressalvas_antes_da_leitura": json.loads((Q / "ressalvas_antes_da_leitura.json").read_text(encoding="utf-8")),
         "medidas": g.medir(te, pred), "comparacoes": comps,
         "regime": "confirmatorio_prospectivo",
         "nao_e": ("nao e validacao no Brasil (os eventos novos sao de fora dele); nao testa cobertura/solo; "
                   "uma execucao so")}
    feito.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    # a partir daqui estes eventos nao confirmam mais nada (reservas_examinadas.regime -> exploratorio)
    rx.registrar_exame(r["eventos_teste"], "gfd", "gfdq03_teste_confirmatorio")
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-registrar", action="store_true")
    ap.add_argument("--executar", action="store_true")
    a = ap.parse_args()
    if a.executar:
        r = executar()
        print(json.dumps(r["comparacoes"], ensure_ascii=False, indent=1))
    else:
        print(pre_registrar())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
