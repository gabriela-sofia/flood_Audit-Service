"""Serra: rotulo so onde dois sensores independentes concordam, com controle seco.

-------------------------------------------------------------------------
REGRA, FIXADA ANTES DE RODAR
-------------------------------------------------------------------------

Sobre os pontos do pacote de adjudicacao (`serra02`), sem HAND, sem OPERA,
sem mascara de terreno:

  agua_de_cheia   S2 limpo no ponto (SCL 4,5,6,7) com MNDWI_pos > 0,1
                  E o mesmo ponto seco no controle do ano anterior
                  (MNDWI_ctrl < -0,1)
                  E S1 depois do evento com VV < -15 dB
                  E queda de VV >= 3 dB contra a mediana pre-evento
  seco_observado  S2 limpo com MNDWI_pos < -0,2
                  E S1 com VV > -15 dB e queda < 1,5 dB
  agua_permanente JRC occurrence >= 50 (fora do binario)
  desconhecido    todo o resto, com o motivo: nuvem, sem S1, discordancia,
                  ambiguo

Optico e radar medem coisas fisicamente diferentes (reflectancia no SWIR x
retroespalhamento); concordarem e evidencia independente. Discordancia nao e
resolvida por nenhum terceiro produto: vira desconhecido e vai para revisao.

CONTROLE SECO: a mesma regra aplicada um ano antes, nos mesmos pontos. A
fracao de `agua_de_cheia` no controle estima o falso positivo do
procedimento. Evento com controle acima de FP_MAXIMO nao tem rotulo
admitido -- todos os pontos dele voltam a desconhecido.

GRUPOS: rotulo admitido conta por celula de 0,05 grau x evento, nao por
ponto. O minimo do projeto e 10 grupos por classe.

nao_e: nao e adjudicacao humana. E rotulo automatico onde criterio
independente sustenta, e lista reduzida do que sobra para revisao.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

PACOTE = REPO / "local_runs" / "serra-02-pacote-de-adjudicacao"
SAIDA = REPO / "local_runs" / "serra-03-rotulo-sensor-duplo"
FP_MAXIMO = 0.02
# v1: base S1 pre-evento de 60 a 10 dias antes (regra original).
# v2 (03/10/2026, declarada antes de ver o resultado dela): base de 365 a 10
# dias -- depois da falha do S1B o S1A passou semanas sem passar sobre o sul
# do Brasil, e o Taquari nao tem S1 nos 50 dias antes de mai/2024. Unica
# mudanca; v1 continua gravada ao lado.
VERSOES_DA_REGRA = {"v1": 60, "v2": 365}
# Seco na imagem de 4 dias depois do pico NAO e "nao inundou": em vale de serra a
# cheia rapida recua em horas. Medido em 03/10/2026: a melhor cena S2 limpa vem
# 4 (Taquari/Cai mai/2024), 6 (SP fev/2023) e ~6 dias (jun/2025) depois do dia
# do evento. Negativo de EVENTO so com cena ate DEFASAGEM_MAXIMA_DO_SECO_DIAS;
# alem disso o seco vira `seco_na_data_da_cena` -- exploracao, nunca negativo.
DEFASAGEM_MAXIMA_DO_SECO_DIAS = 1
SENTINELA = -9999
CELULA_GRUPO = 0.05
MINIMO_GRUPOS = 10
REGRA = {"mndwi_agua": 0.1, "mndwi_seco": -0.2, "mndwi_ctrl_seco": -0.1,
         "vv_agua_db": -15.0, "queda_agua_db": 3.0, "queda_seco_db": 1.5, "jrc_permanente": 50}


def amostrar(ev: dict, pts: pd.DataFrame, deslocamento_dias: int, pre_dias: int = 60) -> pd.DataFrame:
    """MNDWI/SCL do S2 e VV pos/pre do S1 nos pontos, com a data deslocada (0 = evento)."""
    import ee
    ee.Initialize()
    dia = ee.Date(ev["dia"]).advance(deslocamento_dias, "day")
    reg = ee.Geometry.Rectangle(ev["aoi"])
    fc = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"ponto_id": r.ponto_id})
                               for r in pts.itertuples()])

    def s2_melhor(a, b):
        """Mosaico de TODOS os granulos do dia mais limpo na AOI.

        Um granulo so (o `first` da colecao) cobria 5 dos 200 pontos do
        Taquari: fora do footprint dele o `unmask` nao alcanca e o ponto some.
        O mosaico do dia cobre a AOI; o que ainda faltar vira nao observado.
        """
        c = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(reg)
             .filterDate(dia.advance(a, "day"), dia.advance(b, "day")))
        c = c.map(lambda i: i.set("dia", i.date().format("YYYY-MM-dd")))
        dias = ee.List(c.aggregate_array("dia")).distinct()

        def limpeza(d):
            m = c.filter(ee.Filter.eq("dia", d)).mosaic()
            f = m.select("SCL").remap([4, 5, 6, 7], [1, 1, 1, 1], 0).reduceRegion(
                ee.Reducer.mean(), reg, 120, maxPixels=1e9).get("remapped")
            return ee.Feature(None, {"dia": d, "limpo": f})
        melhor = ee.FeatureCollection(dias.map(limpeza)).sort("limpo", False).first().get("dia")
        return c.filter(ee.Filter.eq("dia", melhor)).mosaic().set("dia", melhor)

    pos = s2_melhor(-2, 8)
    ctrl = s2_melhor(-365 - 15, -365 + 15)

    def s2_banda(img, nome):
        return ee.Image.cat([img.normalizedDifference(["B3", "B11"]).rename(f"mndwi_{nome}"),
                             img.select("SCL").rename(f"scl_{nome}")])

    s1 = ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(reg).filter(
        ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).select("VV")
    s1_pos_col = s1.filterDate(dia.advance(-1, "day"), dia.advance(7, "day"))
    s1_pre_col = s1.filterDate(dia.advance(-pre_dias, "day"), dia.advance(-10, "day"))
    # Colecao vazia devolve imagem SEM banda, e o `cat` com ela fazia o
    # `sampleRegions` perder pontos (5 de 200 no Taquari, que nao tem S1 nos
    # 50 dias antes de mai/2024): vira banda toda mascarada, nao observado.
    s1_pre = ee.Image(ee.Algorithms.If(
        s1_pre_col.size().gt(0), s1_pre_col.median(),
        ee.Image.constant(0).updateMask(ee.Image.constant(0)))).rename("vv_pre")
    # menor VV na janela: a agua no momento mais cheio
    s1_pos = ee.Image(ee.Algorithms.If(
        s1_pos_col.size().gt(0), s1_pos_col.min(),
        ee.Image.constant(0).updateMask(ee.Image.constant(0)))).rename("vv_pos")
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).rename("jrc")
    img = ee.Image.cat([s2_banda(pos, "pos"), s2_banda(ctrl, "ctrl"), s1_pos, s1_pre, jrc])
    # O `sampleRegions` DESCARTA o ponto quando qualquer banda esta mascarada
    # (nuvem, sem passagem S1) -- na primeira rodada sobraram 5 de 200 pontos
    # no Taquari. Mascara vira sentinela antes e NaN depois: nao observado e
    # informacao, nao linha que some.
    r = img.unmask(SENTINELA).sampleRegions(fc, scale=10, geometries=False).getInfo()
    linhas = [{k: (np.nan if v == SENTINELA else v) for k, v in f["properties"].items()}
              for f in r["features"]]
    meta = {"s2_pos": pos.get("dia").getInfo(), "s2_ctrl": ctrl.get("dia").getInfo(),
            "s1_cenas_pos": s1_pos_col.size().getInfo(), "s1_cenas_pre": s1_pre_col.size().getInfo()}
    return pd.DataFrame(linhas).assign(**{f"meta_{k}": v for k, v in meta.items()})


def rotular(a: pd.DataFrame) -> pd.DataFrame:
    R = REGRA
    # Banda mascarada no ponto (sem cena S1/S2 na janela) nao vem do EE: a
    # coluna ausente e "nao observado", e cai em desconhecido pelas regras.
    for c in ("scl_pos", "mndwi_pos", "mndwi_ctrl", "vv_pos", "vv_pre", "jrc"):
        if c not in a:
            a = a.assign(**{c: np.nan})
    limpo = a.get("scl_pos", pd.Series(np.nan, index=a.index)).isin([4, 5, 6, 7])
    tem_s1 = a.get("vv_pos", pd.Series(np.nan, index=a.index)).notna() & a.get(
        "vv_pre", pd.Series(np.nan, index=a.index)).notna()
    queda = a.get("vv_pre", np.nan) - a.get("vv_pos", np.nan)
    agua_s2 = limpo & (a.get("mndwi_pos") > R["mndwi_agua"]) & (a.get("mndwi_ctrl") < R["mndwi_ctrl_seco"])
    seco_s2 = limpo & (a.get("mndwi_pos") < R["mndwi_seco"])
    agua_s1 = tem_s1 & (a.get("vv_pos") < R["vv_agua_db"]) & (queda >= R["queda_agua_db"])
    seco_s1 = tem_s1 & (a.get("vv_pos") > R["vv_agua_db"]) & (queda < R["queda_seco_db"])
    classe = pd.Series("desconhecido", index=a.index, dtype=object)
    motivo = pd.Series("", index=a.index, dtype=object)
    perm = a["jrc"].fillna(0) >= R["jrc_permanente"]
    classe[perm] = "agua_permanente"
    livre = ~perm
    classe[livre & agua_s2 & agua_s1] = "agua_de_cheia"
    classe[livre & seco_s2 & seco_s1] = "seco_observado"
    d = classe == "desconhecido"
    motivo[d & ~limpo] = "nuvem_ou_sombra_no_s2"
    motivo[d & limpo & ~tem_s1] = "sem_s1_na_janela"
    motivo[d & limpo & tem_s1 & ((agua_s2 & ~agua_s1) | (agua_s1 & ~agua_s2) | (seco_s2 & ~seco_s1) | (seco_s1 & ~seco_s2))] = "sensores_discordam"
    motivo[d & (motivo == "")] = "ambiguo_entre_limiares"
    return a.assign(classe_automatica=classe, motivo_desconhecido=motivo.where(d, ""))


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regra", choices=sorted(VERSOES_DA_REGRA), default="v1")
    versao = ap.parse_args().regra
    pre_dias = VERSOES_DA_REGRA[versao]
    global SAIDA
    SAIDA = SAIDA / versao
    pac = json.loads((PACOTE / "pacote.json").read_text(encoding="utf-8"))
    pts = pd.read_csv(PACOTE / "pontos_para_revisao.csv")
    SAIDA.mkdir(parents=True, exist_ok=True)
    saida, por_evento = [], {}
    for nome, ev in pac["eventos"].items():
        if ev["estado"] != "pronto_para_revisao":
            continue
        s = pts[pts.evento == nome]
        cache_ev, cache_ct = SAIDA / f"amostra_{nome}.csv", SAIDA / f"controle_{nome}.csv"
        a = pd.read_csv(cache_ev) if cache_ev.exists() else amostrar(ev, s, 0, pre_dias)
        c = pd.read_csv(cache_ct) if cache_ct.exists() else amostrar(ev, s, -365, pre_dias)
        a.to_csv(cache_ev, index=False)
        c.to_csv(cache_ct, index=False)
        a, c = rotular(a), rotular(c)
        fp = float((c.classe_automatica == "agua_de_cheia").mean())
        admitido = fp <= FP_MAXIMO
        a = a.merge(s[["ponto_id", "lat", "lon", "bacia"]], on="ponto_id").assign(evento=nome)
        defasagem = (pd.Timestamp(str(a["meta_s2_pos"].iloc[0])) - pd.Timestamp(ev["dia"])).days
        seco_vale = defasagem <= DEFASAGEM_MAXIMA_DO_SECO_DIAS
        if not seco_vale:
            a.loc[a.classe_automatica == "seco_observado", "classe_automatica"] = "seco_na_data_da_cena"
        a["rotulo_admitido"] = admitido & a.classe_automatica.isin(["agua_de_cheia", "seco_observado"])
        a["grupo"] = (nome + "|" + (a.lat / CELULA_GRUPO).round().astype(int).astype(str) + "_"
                      + (a.lon / CELULA_GRUPO).round().astype(int).astype(str))
        saida.append(a)
        por_evento[nome] = {"pontos": int(len(a)), "fp_no_controle": round(fp, 4), "rotulos_admitidos": admitido,
                            "defasagem_cena_evento_dias": defasagem, "seco_vale_como_negativo": seco_vale,
                            "classes": a.classe_automatica.value_counts().to_dict(),
                            "desconhecido_por_motivo": a.loc[a.classe_automatica == "desconhecido",
                                                             "motivo_desconhecido"].value_counts().to_dict(),
                            "cenas": {k[5:]: a[k].iloc[0] for k in a.columns if k.startswith("meta_")}}
        print(f"  {nome:26s} fp_controle={fp:.3f} admitido={admitido} {a.classe_automatica.value_counts().to_dict()}", flush=True)
    t = pd.concat(saida, ignore_index=True)
    adm = t[t.rotulo_admitido]
    grupos = {c: int(adm.loc[adm.classe_automatica == c, "grupo"].nunique()) for c in ("agua_de_cheia", "seco_observado")}
    eventos = {c: int(adm.loc[adm.classe_automatica == c, "evento"].nunique()) for c in ("agua_de_cheia", "seco_observado")}
    revisar = t[t.classe_automatica == "desconhecido"].copy()
    revisar = revisar[revisar.motivo_desconhecido.isin(["sensores_discordam", "ambiguo_entre_limiares"])]
    t.to_csv(SAIDA / "rotulos.csv", index=False)
    revisar[["ponto_id", "evento", "bacia", "lat", "lon", "motivo_desconhecido", "mndwi_pos", "vv_pos", "vv_pre"]].to_csv(
        SAIDA / "revisao_humana_reduzida.csv", index=False)
    r = {"tipo": "rotulo_serra_sensor_duplo", "quando_utc": agora_utc(), "versao_da_regra": versao,
         "base_s1_pre_dias": pre_dias, "regra": REGRA,
         "fp_maximo_no_controle": FP_MAXIMO, "por_evento": por_evento,
         "rotulos_admitidos": int(len(adm)), "grupos_por_classe": grupos, "eventos_por_classe": eventos,
         "minimo_grupos": MINIMO_GRUPOS,
         "testavel": min(grupos.values()) >= MINIMO_GRUPOS,
         "para_revisao_humana": int(len(revisar)),
         "fora_de_revisao_por_falta_de_observacao": int((t.classe_automatica == "desconhecido").sum() - len(revisar)),
         "nao_e": "nao e adjudicacao humana: rotulo onde dois sensores independentes concordam e o controle passou"}
    r["hash_resumo"] = sha256_json({k: v for k, v in r.items() if k != "quando_utc"})
    (SAIDA / "manifesto.json").write_text(json.dumps(r, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: r[k] for k in ("rotulos_admitidos", "grupos_por_classe", "eventos_por_classe", "testavel",
                                        "para_revisao_humana")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
