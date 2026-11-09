"""MT02, passo 1: cobertura e solo v2 para a populacao vigente (ds06 v7), sem tocar no cob01 nem no ds06.

O DEFEITO (auditado em 10/10/2026)
O `cob01` rodou sobre o ds06 v3. As 15.437 linhas CEMS datadas por cena so
ganharam data no v6; no v3 nao tinham, entao foram coletadas SEM ANO e
receberam so as camadas estaticas (WorldCover, SoilGrids). Nenhum ponto ficou
sem coleta: o que faltou foram as camadas temporais (GHSL, MODIS, MapBiomas)
de pontos que hoje tem ano. 93% deles sao negativos -- dai o atalho.

O QUE ESTA VERSAO FAZ
  reaproveita   toda linha do cob01 cujo ano de evento confere com o do v7
  recoleta      so as linhas cujo ano mudou, com a mesma funcao `imagem` do cob01
  classifica    cada ausencia por atributo, sem virar sinal fisico:
                  pipeline              sem ano no dado que pediu a camada temporal
                  fora_da_cobertura     MapBiomas fora do Brasil
                  mascara_da_camada     a camada respondeu nulo com ano e no dominio
                  valor                 lido
  audita        IDs, coordenadas, anos efetivos (emprestados do extremo), resolucao
                nativa e preenchimento por fonte x classe, antes e depois

nao_e: nao reescreve `cob-01-cobertura-e-solo/`, nao cria ds06 v8, nao preenche
mascara; ausencia continua ausencia.

Uso:
    python scripts/metodo/mt02_cobertura_v2.py coletar
    python scripts/metodo/mt02_cobertura_v2.py juntar
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "geo", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import cob01_cobertura_e_solo as cob  # noqa: E402

RUNS = REPO / "local_runs"
DS06 = RUNS / "ds-06-dataset-de-treino" / "v7" / "pontos.parquet"
COB01 = RUNS / "cob-01-cobertura-e-solo" / "cobertura_e_solo.parquet"
SAIDA = RUNS / "cob-02-cobertura-v7"
LOTES = SAIDA / "lotes_recoleta"
TEMPORAIS = ("mapbiomas_classe", "modis_lc", "construido_frac")
RESOLUCAO_M = {"mapbiomas_classe": 30, "modis_lc": 500, "construido_frac": 100, "worldcover_2021": 10,
               "argila_pct_0_30": 250, "areia_pct_0_30": 250}
# caixa do Brasil (folgada): fora dela MapBiomas nulo e fora de cobertura, nao mascara
BRASIL = (-74.1, -34.0, -28.6, 5.4)


def plano() -> tuple[pd.DataFrame, pd.DataFrame]:
    v7 = pd.read_parquet(DS06, columns=["ponto_id", "fonte", "classe", "lat", "lon", "janela_inicio"])
    c = pd.read_parquet(COB01)
    if c.ponto_id.duplicated().any() or v7.ponto_id.duplicated().any():
        raise ValueError("ponto_id duplicado")
    v7["ano"] = cob.ano_de_referencia(v7.janela_inicio)
    m = v7.merge(c, on="ponto_id", how="left", indicator=True)
    m["refazer"] = (m._merge == "left_only") | (m.ano.astype("float").fillna(-1) != m.ano_evento.astype("float").fillna(-1))
    return v7, m


def coletar(trabalhadores: int = 4) -> dict:
    import ee
    ee.Initialize()
    v7, m = plano()
    alvo = v7[v7.ponto_id.isin(m[m.refazer].ponto_id)]
    LOTES.mkdir(parents=True, exist_ok=True)
    original = cob.LOTES
    cob.LOTES = LOTES  # mesma funcao de coleta, outra pasta: o cob01 fica intocado
    try:
        lotes = cob.planejar(alvo)
        falta = [l for l in lotes if not (LOTES / f"{l['nome']}.parquet").exists()]
        falhas = []
        with ThreadPoolExecutor(trabalhadores) as pool:
            fut = {pool.submit(cob.coletar_lote, ee, l): l["nome"] for l in falta}
            for i, f in enumerate(as_completed(fut), 1):
                try:
                    f.result()
                    print(i, "/", len(falta), fut[f], flush=True)
                except Exception as e:  # noqa: BLE001
                    falhas.append({"lote": fut[f], "erro": str(e)[:200]})
    finally:
        cob.LOTES = original
    return {"pontos_a_recoletar": int(len(alvo)), "lotes": len(lotes), "falhas": falhas}


def motivo(d: pd.DataFrame) -> pd.DataFrame:
    """Por que cada atributo esta ausente. Nenhum desses estados vira valor."""
    no_br = d.lon.between(BRASIL[0], BRASIL[2]) & d.lat.between(BRASIL[1], BRASIL[3])
    out = pd.DataFrame(index=d.index)
    for c in cob.COLUNAS:
        r = np.where(d[c].notna(), "valor", "mascara_da_camada").astype(object)
        if c in TEMPORAIS:
            r[(d[c].isna() & d.ano_evento.isna()).to_numpy()] = "pipeline_sem_ano"
        if c == "mapbiomas_classe":
            r[(d[c].isna() & ~no_br).to_numpy()] = "fora_da_cobertura"
        out[f"motivo_{c}"] = r
    return out


def juntar() -> dict:
    v7, m = plano()
    novos = pd.concat([pd.read_parquet(p) for p in sorted(LOTES.glob("*.parquet"))], ignore_index=True)
    if novos.ponto_id.duplicated().any():
        raise ValueError("recoleta duplicada")
    manter = m[~m.refazer][["ponto_id", *cob.COLUNAS, "ano_evento", "mapbiomas_ano", "modis_ano", "ghsl_ano"]]
    manter = manter.assign(origem="cob01_reaproveitado")
    novos = novos.assign(origem="recoleta_v2")
    d = pd.concat([manter, novos[manter.columns]], ignore_index=True)
    faltam = set(v7.ponto_id) - set(d.ponto_id)
    if faltam:
        raise RuntimeError(f"{len(faltam)} pontos sem linha; rode coletar de novo")
    d = d.merge(v7[["ponto_id", "fonte", "classe", "lat", "lon", "ano"]], on="ponto_id")
    if (d.ano.astype("float").fillna(-1) != d.ano_evento.astype("float").fillna(-1)).any():
        raise RuntimeError("ano da camada nao confere com o ano do v7")
    d = pd.concat([d, motivo(d)], axis=1)
    arq = SAIDA / "cobertura_e_solo_v2.parquet"
    d.drop(columns=["fonte", "classe", "lat", "lon", "ano"]).sort_values("ponto_id").to_parquet(arq, index=False)

    antes = m

    def preench(x):
        return {f"{f}|{k}": {c: round(float(g[c].notna().mean()), 4) for c in cob.COLUNAS}
                for (f, k), g in x.groupby(["fonte", "classe"])}

    man = {"tipo": "cobertura_e_solo_v2", "quando_utc": agora_utc(),
           "populacao": str(DS06.relative_to(REPO)), "populacao_sha256": sha256_arquivo(DS06),
           "cob01_reaproveitado_sha256": sha256_arquivo(COB01),
           "codigo": {str(Path(__file__).relative_to(REPO)): sha256_arquivo(Path(__file__)),
                      str(Path(cob.__file__).relative_to(REPO)): sha256_arquivo(Path(cob.__file__))},
           "pontos": int(len(d)), "reaproveitados": int((d.origem == "cob01_reaproveitado").sum()),
           "recoletados": int((d.origem == "recoleta_v2").sum()),
           "anos_emprestados_do_extremo_na_recoleta": {
               "modis": int((novos.modis_ano != novos.ano_evento).sum()),
               "mapbiomas": int((novos.mapbiomas_ano != novos.ano_evento).sum()),
               "ghsl_epoca": novos.ghsl_ano.value_counts().to_dict()},
           "resolucao_nativa_m": RESOLUCAO_M,
           "motivos": {c: d[f"motivo_{c}"].value_counts().to_dict() for c in cob.COLUNAS},
           "preenchimento_antes": preench(antes), "preenchimento_depois": preench(d.assign(
               fonte=d.fonte, classe=d.classe)),
           "saida_sha256": sha256_arquivo(arq),
           "nao_e": "nao reescreve cob01 nem ds06; ausencia classificada, nunca preenchida"}
    man["hash_resumo"] = sha256_json({k: v for k, v in man.items() if k != "quando_utc"})
    (SAIDA / "manifesto.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    return man


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("acao", choices=("coletar", "juntar"))
    a = ap.parse_args()
    r = coletar() if a.acao == "coletar" else juntar()
    print(json.dumps({k: v for k, v in r.items() if not k.startswith("preenchimento")}, ensure_ascii=False,
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
