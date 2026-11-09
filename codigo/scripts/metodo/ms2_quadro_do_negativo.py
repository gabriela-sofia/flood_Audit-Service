"""MS2: a queda CEMS -> GFD vem do QUADRO do negativo? Harmonizar o CEMS ao desenho do GFD recupera a transferencia?

-------------------------------------------------------------------------
A HIPOTESE, e de onde ela vem
-------------------------------------------------------------------------

No CEMS o negativo e toda a area observada da AOI menos a mancha, numa grade de
150 m (`cems03_amostrar_pontos`): entra terreno alto e longe do rio. No GFD o
negativo so vem da faixa de 2 km em torno da mancha, a mais de 250 m da borda
(`gfd01.classes_do_rotulo`). Em modelos de presenca/fundo, a extensao de onde
vem o fundo muda o desempenho e a importancia das variaveis (VanDerWal et al.
2009, Ecological Modelling 220:589-594). Na v2 da generalizacao a logistica do
CEMS caiu a 0,573 no GFD, abaixo do HAND (0,738).

-------------------------------------------------------------------------
DESENHO (gravado em `pre_registro.json` antes de qualquer ajuste)
-------------------------------------------------------------------------

  quadro_aoi   o CEMS como esta (populacao de terreno + posicao finita)
  quadro_2km   o desenho do GFD sobre os PONTOS do CEMS, por AOI: positivo com
               o negativo mais proximo a > 250 m; negativo com o positivo mais
               proximo entre 250 m e 2 km
  modelos      base (hand, twi, slope) e base_pos2km, logistica, ajuste unico no CEMS
  teste        GFD sem tile de 1 grau ocupado pelo CEMS -- o MESMO nos dois quadros
  comparacoes  T1 base: quadro_2km x quadro_aoi;  T2 base_pos2km: idem
  leitura      generalizacao_por_reservas.veredito sobre a AUC dentro do evento GFD
  primario     T1
  descritivo   GFD -> CEMS avaliado nos dois quadros (nivel, nao comparacao)

Pre-registrado ANTES de ler o MS1 (que roda na mesma fila).

nao_e: nao muda o rotulo do CEMS no ds06 nem decide qual quadro e o certo para
a pergunta de suscetibilidade; mede se a diferenca de desenho explica a queda.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

import generalizacao_por_reservas as gen  # noqa: E402
import ms1_posicao_topografica as ms1  # noqa: E402
from proveniencia import agora_utc, sha256_json  # noqa: E402

SAIDA = REPO / "local_runs" / "ms2-quadro-do-negativo"
MARGEM_M, FAIXA_M = 250.0, 2000.0
MODELOS = {"base": ms1.BASE, "base_pos2km": ms1.BASE + ("pos_min_2km",)}
SEMENTE = 20261007
VISTO_ANTES = ("generalizacao_v2 (07/10): CEMS -> GFD logistica 0,573 < HAND 0,738; MS1 pre-registrado e na fila, "
               "resultado NAO lido quando este pre-registro foi gravado")


def pre_registro() -> dict:
    return json.loads(json.dumps({
        "quadros": {"quadro_aoi": "CEMS como esta",
                    "quadro_2km": f"positivo com negativo mais proximo > {MARGEM_M} m; negativo com positivo mais "
                                  f"proximo em ({MARGEM_M}, {FAIXA_M}] m; por AOI"},
        "modelos": {k: list(v) for k, v in MODELOS.items()},
        "teste": "GFD (carregar(TERRENO) + posicao finita) sem tile de 1 grau do CEMS",
        "comparacoes": {"T1": "base: quadro_2km x quadro_aoi", "T2": "base_pos2km: quadro_2km x quadro_aoi"},
        "primario": "T1", "regra": "generalizacao_por_reservas.veredito", "visto_antes": VISTO_ANTES,
        "semente": SEMENTE}))


def gravar_pre_registro(destino: Path = SAIDA / "pre_registro.json") -> str:
    corpo = pre_registro()
    h = sha256_json(corpo)
    if destino.exists():
        antigo = json.loads(destino.read_text(encoding="utf-8"))
        if antigo["hash"] != h:
            raise RuntimeError(f"pre-registro mudou ({antigo['hash']} -> {h}): crie MS3")
        return h
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"quando_utc": agora_utc(), "hash": h, "criterios": corpo},
                                  ensure_ascii=False, indent=1), encoding="utf-8")
    return h


def quadro_2km(c: pd.DataFrame) -> pd.DataFrame:
    """O desenho de classes do GFD aplicado aos pontos de cada AOI do CEMS."""
    from sklearn.neighbors import BallTree
    manter = np.zeros(len(c), dtype=bool)
    c = c.reset_index(drop=True)
    for _, idx in c.groupby("grupo_validacao").indices.items():
        s = c.iloc[idx]
        pos, neg = s[s.classe == 1], s[s.classe == 0]
        if pos.empty or neg.empty:
            continue
        rad = lambda x: np.radians(x[["lat", "lon"]].to_numpy(float))  # noqa: E731
        d_pos = BallTree(rad(pos), metric="haversine").query(rad(neg), k=1)[0][:, 0] * 6_371_000
        d_neg = BallTree(rad(neg), metric="haversine").query(rad(pos), k=1)[0][:, 0] * 6_371_000
        manter[idx[(s.classe == 1).to_numpy()]] = d_neg > MARGEM_M
        manter[idx[(s.classe == 0).to_numpy()]] = (d_pos > MARGEM_M) & (d_pos <= FAIXA_M)
    return c[manter]


def executar() -> dict:
    import posicao_topografica as pt
    import verdade_observacional as vo
    h = gravar_pre_registro()
    d, versao, _ = gen.carregar(gen.TERRENO)
    d = pt.juntar(d)
    d = d[np.isfinite(d[list(ms1.POSICAO)].to_numpy(float)).all(axis=1)]
    c = d[d.fonte == "cems"].reset_index(drop=True)
    g = d[d.fonte == "gfd"]
    teste = g[~vo.chave_espacial(g).isin(set(vo.chave_espacial(c)))].reset_index(drop=True)
    quadros = {"quadro_aoi": c, "quadro_2km": quadro_2km(c)}
    rng = np.random.default_rng(SEMENTE)
    pred, coef = {}, {}
    for q, tr in quadros.items():
        for m, v in MODELOS.items():
            fn = ms1.ajustar(tr, v)
            pred[(q, m)] = fn(teste)
            coef[f"{q}|{m}"] = fn.coef
    comps = {}
    for tid, m in (("T1", "base"), ("T2", "base_pos2km")):
        r = gen.comparar(teste, pred[("quadro_2km", m)], pred[("quadro_aoi", m)], rng)
        comps[tid] = {**r, "veredito": gen.veredito(r)}
    medidas = {q: ms1.medir(teste, {m: pred[(q, m)] for m in MODELOS}) for q in quadros}
    medidas["regra_hand"] = ms1.medir(teste, {"M0_regra_hand": -teste.hand_m.to_numpy(float)})
    # descritivo: GFD -> CEMS nos dois quadros (muda a populacao de teste: nivel, nao comparacao)
    g_tr = g[~vo.chave_espacial(g).isin(set(vo.chave_espacial(c)))]
    desc = {}
    for q, te in quadros.items():
        te = te.reset_index(drop=True)
        desc[q] = ms1.medir(te, {m: ms1.ajustar(g_tr, v)(te) for m, v in MODELOS.items()}
                            | {"M0_regra_hand": -te.hand_m.to_numpy(float)})
    r = {"tipo": "ms2_quadro_do_negativo", "dataset": versao, "pre_registro": h, "regime": "exploratorio",
         "linhas_por_quadro": {q: {"linhas": int(len(x)), "positivos": int(x.classe.sum()),
                                   "eventos": int(x.evento_id.nunique()),
                                   "hand_mediano_neg": round(float(x[x.classe == 0].hand_m.median()), 2)}
                               for q, x in quadros.items()},
         "teste_gfd": {"linhas": int(len(teste)), "eventos": int(teste.evento_id.nunique())},
         "comparacoes": comps, "medidas_no_gfd": medidas, "coeficientes": coef,
         "descritivo_gfd_para_cems": desc,
         "nao_e": "nao muda o rotulo do CEMS no ds06 nem decide o quadro certo para a pergunta de suscetibilidade"}
    r["hash_resumo"] = sha256_json(json.loads(json.dumps(r, default=float)))
    r["quando_utc"] = agora_utc()
    (SAIDA / "resultado.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    return r


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pre-registrar", action="store_true")
    a = ap.parse_args()
    if a.pre_registrar:
        print(gravar_pre_registro())
        return 0
    inicio = time.time()
    r = executar()
    print(json.dumps({k: v["veredito"] for k, v in r["comparacoes"].items()}), round(time.time() - inicio), "s")
    print(f"GRAVADO={SAIDA / 'resultado.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
