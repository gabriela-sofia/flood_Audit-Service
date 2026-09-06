"""
CHUVA-03 -- Recupera a data de evento das fontes que a perderam na reducao.

POR QUE ISTO E O BLOQUEIO REAL DA CHUVA, e nao a API:

hoje 9.156 dos 64.989 pontos do pool fluvial tem chuva -- 14%. A causa nao e
falta de fonte de precipitacao: ERA5-Land e global e a API ja e usada tres
vezes neste projeto. A causa e que `rain_max_24h` se define numa JANELA ANTES
DO EVENTO, e sem data de evento nao existe janela. CEMS, Sen1Floods11 e UFO
entraram no `ds01` com `data_evento` fixado em NaT.

E o mais importante: as datas NAO estavam perdidas, estavam descartadas. As
tres existem em artefato local, cada uma num lugar diferente:

  CEMS           `cems-04-fila/ativacoes_meta.json` guarda `eventTime` para as
                 112 ativacoes. O codigo da ativacao e o prefixo da AOI
                 (EMSR847_AOI01 -> EMSR847).
  Sen1Floods11   cada chip tem um item STAC em `n1a-sen1floods11/v1.1/catalog/`
                 com `properties.datetime`.
  UFO            a data esta no PROPRIO NOME do chip -- QUE_20190429_154401_87
                 e 29/04/2019. Vale para os 215 chips, sem excecao.

O QUE CADA UMA CUSTA EM PRECISAO, declarado por linha em `precisao_data`:

  chip_stac      data do chip rotulado. E a melhor: e a cena que o analista viu.
  nome_do_chip   idem, embutida no identificador pelo proprio produtor.
  ativacao_cems  `eventTime` e da ATIVACAO, nao da AOI. Uma ativacao cobre
                 varias AOIs e pode se estender por dias. Todas as AOIs de uma
                 ativacao recebem a mesma data, o que e aproximacao e nao
                 medida. Fica marcado para que ninguem trate as duas como
                 iguais depois.

NAO faz: nao baixa nada, nao chama API, nao calcula chuva. So resolve datas e
declara a procedencia de cada uma. A aquisicao e o `chuva04`.

Uso:
    python scripts/treino/chuva03_resolver_datas_evento.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
OUT = RUNS / "chuva-03-datas"

DS01 = RUNS / "ds-01-multirregiao" / "dataset_multirregiao_v1.csv"
ATIVACOES = RUNS / "cems-04-fila" / "ativacoes_meta.json"
STAC = RUNS / "n1a-sen1floods11" / "v1.1" / "catalog" / "sen1floods11_hand_labeled_label"
N1G = RUNS / "n1g-pontos-com-features" / "nivel1_pontos_com_features_v1.csv"

DATA_NO_NOME = re.compile(r"(20[0-2]\d)(0[1-9]|1[0-2])([0-2]\d|3[01])")


def datas_cems() -> dict[str, tuple[str, str]]:
    """AOI -> (data, precisao). A data e da ativacao, nao da AOI."""
    if not ATIVACOES.exists():
        print(f"   AVISO: {ATIVACOES} ausente")
        return {}
    meta = json.loads(ATIVACOES.read_text(encoding="utf-8"))
    por_codigo = {}
    for codigo, m in meta.items():
        t = m.get("eventTime") or m.get("activationTime")
        if t:
            por_codigo[str(codigo)] = str(pd.to_datetime(t).date())
    print(f"   CEMS: {len(por_codigo)} ativacoes com data de "
          f"{len(meta)} no catalogo")
    return por_codigo


def datas_sen1floods11() -> dict[str, str]:
    """chip -> data, lida do item STAC do proprio produto."""
    if not STAC.exists():
        print(f"   AVISO: {STAC} ausente")
        return {}
    fora = {}
    for pasta in STAC.iterdir():
        if not pasta.is_dir():
            continue
        item = pasta / f"{pasta.name}.json"
        if not item.exists():
            continue
        try:
            d = json.loads(item.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        dt = (d.get("properties") or {}).get("datetime")
        if not dt:
            continue
        # o chip no dataset e <base>_LabelHand; a pasta STAC e <base>_label
        base = pasta.name[:-len("_label")] if pasta.name.endswith("_label") else pasta.name
        fora[f"{base}_LabelHand"] = str(pd.to_datetime(dt).date())
    print(f"   SEN1FLOODS11: {len(fora)} chips com datetime no STAC")
    return fora


def datas_ufo(chips) -> dict[str, str]:
    """chip -> data, extraida do identificador. Sem heuristica: ou casa, ou nao."""
    fora = {}
    for c in chips:
        m = DATA_NO_NOME.search(str(c))
        if m:
            fora[str(c)] = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    print(f"   UFO: {len(fora)} chips com data no nome")
    return fora


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("===CHUVA03_RESOLVER_DATAS===")
    if not DS01.exists():
        print(f"ABORTADO: {DS01} ausente. Rode ds01.")
        return 1

    cems = datas_cems()
    s1f = datas_sen1floods11()

    chips_por_ponto = {}
    if N1G.exists():
        n1 = pd.read_csv(N1G, low_memory=False)
        chips_por_ponto = dict(zip(n1["ponto_id"].astype(str), n1["chip"].astype(str)))
        ufo_chips = n1.loc[n1.fonte.astype(str).str.startswith("ufo"), "chip"].unique()
    else:
        print(f"   AVISO: {N1G} ausente -- UFO e Sen1Floods11 ficam sem data")
        ufo_chips = []
    ufo = datas_ufo(ufo_chips)

    base = pd.read_csv(DS01, low_memory=False)
    seletores = {
        "cems": base["regiao"].astype(str).str.startswith("analogo_EMSR"),
        "sen1floods11": base["regiao"] == "nivel1_sen1floods11_handlabeled",
        "ufo": base["regiao"] == "nivel1_ufo_urban_flood_observations",
    }
    linhas = []
    for fonte, mascara in seletores.items():
        d = base.loc[mascara].copy()
        if d.empty:
            continue
        for _, r in d.iterrows():
            pid, aoi = str(r["ponto_id"]), str(r["grupo_cv"])
            data, precisao = None, None
            if fonte == "cems":
                codigo = aoi.split("_")[0]
                data = cems.get(codigo)
                precisao = "ativacao_cems"
            else:
                chip = chips_por_ponto.get(pid, aoi)
                if fonte == "sen1floods11":
                    data, precisao = s1f.get(chip), "chip_stac"
                else:
                    data, precisao = ufo.get(chip), "nome_do_chip"
            linhas.append({"ponto_id": pid, "fonte": fonte, "aoi": aoi,
                           "data_evento": data,
                           "precisao_data": precisao if data else None,
                           "motivo_sem_data": None if data else
                           f"sem data para {aoi} em {precisao}"})

    reg = pd.DataFrame(linhas)
    if reg.empty:
        print("ABORTADO: nenhuma fonte para resolver.")
        return 1

    print("\n--- COBERTURA ---")
    r = reg.groupby("fonte").agg(
        n=("ponto_id", "size"),
        com_data=("data_evento", lambda x: int(x.notna().sum())))
    r["pct"] = (100 * r.com_data / r.n).round(1)
    print(r.to_string())
    print(f"\nTOTAL com data: {int(reg.data_evento.notna().sum()):,} de {len(reg):,}")
    print(f"precisao: {reg.loc[reg.data_evento.notna(), 'precisao_data'].value_counts().to_dict()}")

    sem = reg[reg.data_evento.isna()]
    if len(sem):
        print(f"\nSEM DATA: {len(sem):,} pontos em "
              f"{sem.aoi.nunique()} AOIs/chips -- {sorted(sem.aoi.unique())[:5]}")

    com = reg[reg.data_evento.notna()]
    if len(com):
        dt = pd.to_datetime(com.data_evento)
        print(f"\nperiodo coberto: {dt.min().date()} a {dt.max().date()}, "
              f"{com.data_evento.nunique()} datas distintas")

    reg.to_csv(OUT / "datas_evento_resolvidas.csv", index=False)
    (OUT / "resumo.json").write_text(json.dumps({
        "por_fonte": json.loads(r.to_json(orient="index")),
        "total_com_data": int(com.data_evento.notna().sum()),
        "total": int(len(reg)),
        "precisao": com.precisao_data.value_counts().to_dict(),
        "datas_distintas": int(com.data_evento.nunique()) if len(com) else 0,
        "limite_declarado": (
            "a data do CEMS e da ATIVACAO e nao da AOI: uma ativacao cobre "
            "varias AOIs e pode durar dias, entao todas as AOIs de uma "
            "ativacao recebem a mesma data. Aproximacao declarada, nao medida"),
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
