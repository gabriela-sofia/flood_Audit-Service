"""Verdade observacional: o que o rotulo de cada fonte realmente ve, e quando dois rotulos sao o mesmo evento.

-------------------------------------------------------------------------
POR QUE ISTO EXISTE
-------------------------------------------------------------------------

A admissao (`admissao_cientifica`) decide PARA QUE uma linha pode ser usada.
Ela nao dizia tres coisas que a auditoria de 06/10/2026 mediu nos artefatos:

  1. O rotulo do GFD (e da replica `rfd01`) nasce de um algoritmo que MASCARA
     pixel com declive > 5 graus no GMTED2010 antes de decidir agua. A
     populacao rotulada e truncada num preditor do proprio modelo. Nao e
     circularidade (os dois lados da classe sofrem o mesmo corte), e
     dominio: o GFD nao observa terreno ingreme, e o sinal do declive e do
     HAND medido nele vale dentro desse corte.
  2. "Positivo episodico" do GFD inclui agua RECORRENTE: o projeto so tira
     occurrence JRC >= 50; entre 10 e 50 fica positivo. No treino isso e 33%
     dos positivos; no teste brasileiro (replica), 75%.
  3. O mesmo evento aparece com dois IDs em fontes diferentes (GFD:4577 e
     S1F11:Bolivia, fev/2018; GFD:4703 e S1F11:Paraguay, out/2018), e
     eventos consultados fora da tabela (pacote de serra, leitura parcial de
     GFD que toca o Brasil) nao estavam em lista nenhuma. `reservas_examinadas`
     so comparava ID dentro da mesma fonte.

Este modulo e a regra unica dessas tres coisas. `reservas_examinadas.regime`
usa `equivalencias` quando o teste traz janelas; a auditoria pre-treino le
o resto.

nao_e: nao muda nenhuma admissao nem rotulo. Declarar o dominio e o regime
hidrico nao bloqueia uso: diz o que o resultado pode afirmar.

Uso:
    python scripts/metodo/verdade_observacional.py      # grava identidade + anotacoes do ds06
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for pasta in ("metodo", "servico", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / pasta))

RUNS = REPO / "local_runs"
SAIDA = RUNS / "verdade-observacional"
PONTOS_GFD = RUNS / "gfd-brasil" / "pontos_gfd_com_terreno.csv"
PACOTE_SERRA = RUNS / "serra-02-pacote-de-adjudicacao" / "pacote.json"
LOTE_TOCAM_BRASIL = RUNS / "gfd-brasil" / "lote_que_tocam_o_brasil.json"

# --------------------------------------------------------------------------
# 1. Dominio do rotulo: o que o algoritmo do produto deixa de fora ANTES da
# classe existir. Evidencia no codigo, nao na prosa.
# --------------------------------------------------------------------------
DOMINIO_DO_ROTULO = {
    "gfd": {
        "mascara_de_terreno": ("declive <= 5 graus no GMTED2010 (~250 m): pixel mais ingreme fica sem "
                               "rotulo, nem positivo nem negativo"),
        "evidencia_no_codigo": ("scripts/geo/rfd01_replica_gfd.py `updateMask(declive.lte(5))`, copia do "
                                "codigo publico do GFD (flood_detection/modis.py)"),
        "agua_permanente": "JRC transition == 1 sai no produto; o projeto ainda reclassifica occurrence >= 50 (gfd02)",
        "agua_recorrente": "occurrence entre 0 e 50 continua positivo: varzea sazonal entra como inundacao",
        "negativo": ("clear_views > 0 em ALGUM dia da janela e nunca agua: pode ter sido visto so antes "
                     "ou depois do pico; clear_views por ponto nao e gravado"),
        "consequencia": ("populacao truncada por um descritor de terreno (declive GMTED ~250 m), correlato dos "
                         "preditores: o efeito de declive/HAND medido no GFD vale so dentro do corte, e servir fora "
                         "dele e extrapolacao (o ds03 ja tira GFD de relevo INGREME)"),
        "verificacao_empirica": ("07/10/2026 (auditoria_pre_treino.chk_mascara_de_declive): a ~240 m, 0,95% dos "
                                 "negativos do treino GFD passam de 5 graus, no nivel do controle em que a mascara e "
                                 "garantida (replica rfd: 1,4%). O corte virou NoData, nao negativo, no produto publicado"),
        "vieses_que_restam": ("(1) o negativo de vale encaixado so pode vir da parte plana do vale: atenua o contraste "
                              "de declive, nao o infla; (2) a margem de 250 m afasta positivo de NEGATIVO, nao de pixel "
                              "mascarado: positivo encostado em parede ingreme conta como interior; (3) GMTED e FABDEM "
                              "diferem: o corte nao e uma regra exata sobre o preditor slope_deg"),
    },
    "rfd": {
        "mascara_de_terreno": "a mesma do GFD (replica do algoritmo)",
        "evidencia_no_codigo": "scripts/geo/rfd01_replica_gfd.py linhas 121-123 e 168-170",
        "independencia": ("replica do MESMO algoritmo, MODIS e JRC: a concordancia com o GFD (portao do rfd01) "
                          "mede reproducao, nao acuracia independente"),
        "regiao": ("bacia HydroSHEDS nivel 4 inteira na janela do GDACS: o tile rotulado pode estar a "
                   "centenas de km do centroide do evento"),
        "consequencia": "'evento' do teste brasileiro e janela x bacia, nao a cheia catalogada no ponto",
    },
}

# --------------------------------------------------------------------------
# 2. Alvos e regime hidrico.
#
# O QUE O MODELO APRENDE e "agua nao permanente observada na janela". Os
# estratos episodico/recorrente sao LEITURAS desse alvo, nao alvos novos nem
# exclusoes: varzea que enche todo ano e suscetivel, e recorrencia nao prova
# que um modelo "memorizou" -- so que o lugar costuma ter agua.
# --------------------------------------------------------------------------
JRC = {"produto": "JRC Global Surface Water v1.4, banda occurrence", "epoca": ("1984-03-16", "2021-12-31"),
       "sensor": "Landsat 30 m, agua aberta", "denominador": "meses com observacao valida no pixel (varia com nuvem)",
       "limites": ("agua sob dossel (floresta alagada da Amazonia) e invisivel: recorrencia medida e LIMITE INFERIOR; "
                   "evento dentro da epoca contribui para a propria occurrence (um mes em ~200 validos ~0,5 ponto): "
                   "a fronteira nunca/rara e endogena ao evento ate 2021, a fronteira de 10 quase nao")}

ALVOS = {
    "agua_observada": {"definicao": "flooded == 1 em algum composto de 3 dias da janela (MODIS 250 m)",
                       "inclui": "agua permanente", "uso": "nenhum modelo aprende isto diretamente"},
    "agua_nao_permanente": {"definicao": ("agua_observada E JRC transition != permanente (produto) E occurrence < 50 "
                                          "(projeto, gfd02); negativo: visto limpo >= 1 dia na janela, nunca agua, a "
                                          "<= 2 km do positivo, occurrence < 50"),
                            "uso": "ALVO de todos os modelos de terreno do ds06 GFD e dos dois confirmatorios",
                            "denominador": ("pontos numa grade de 250 m dentro da faixa de 2 km, classes amostradas: "
                                            "fracao positiva da amostra nao e area inundada (V4)")},
    "inundacao_episodica": {"definicao": "agua_nao_permanente com occurrence < 10 (nunca ou rara)",
                            "uso": "estrato de leitura; nao e alvo separado"},
    "inundacao_recorrente": {"definicao": "agua_nao_permanente com 10 <= occurrence < 50",
                             "uso": "estrato de leitura; nunca vira exclusao nem prova de memorizacao"},
}
CORTES = {"50": "calibrado em 2026 contra o JRCWaterHand do Sen1Floods11 (IoU 0,8616); define agua permanente",
          "10": "fixado em 06/10/2026 ANTES de qualquer leitura estratificada; nao foi escolhido por desempenho",
          "sensibilidade_descritiva": "composicao do rotulo tambem nos cortes 5 e 20, sem metrica de modelo"}

REGIMES_HIDRICOS = (
    ("nunca_agua", 0.0, 0.0),            # occurrence == 0
    ("agua_rara", 0.0, 10.0),            # 0 < occ < 10
    ("agua_recorrente", 10.0, 50.0),     # 10 <= occ < 50: varzea sazonal
    ("agua_permanente", 50.0, 100.0),    # >= 50: o projeto ja tira do binario (gfd02)
)


# Populacoes-alvo: tres perguntas diferentes, nao tres jeitos de medir a mesma.
# O quadro NAO e escolhido pela AUC: e escolhido pela pergunta, antes de ler.
POPULACOES_ALVO = {
    "vizinhanca_da_inundacao": {
        "pergunta": "dado que a cheia ocorreu ali, quais pontos da vizinhanca dela foram alcancados pela agua?",
        "positivo": "interior da mancha observada, a mais de 250 m da borda",
        "negativo": "observado seco na faixa de 250 m a 2 km da mancha (GFD: clear_views > 0)",
        "delimitacao": "faixa construida a partir da propria inundacao",
        "delimitacao_usa_o_desfecho": True,
        "consultavel_antes_do_evento": False,
        "fontes": ("gfd", "rfd"),
        "permite": "ordenar pontos perto da agua observada, condicional ao evento",
        "nao_permite": ("mapa de suscetibilidade de paisagem; consulta de lugar antes do evento (o buffer depende do "
                        "desfecho); probabilidade territorial"),
    },
    "aoi_do_analista": {
        "pergunta": "na area que o analista julgou afetada pela ativacao, onde a agua chegou?",
        "positivo": "delineado como inundado pelo produtor (CEMS)",
        "negativo": "observado nao inundado na grade de 150 m da AOI (sem agua permanente delineada)",
        "delimitacao": "AOI desenhada pelo analista em torno do evento",
        "delimitacao_usa_o_desfecho": "parcial: escolhida pelo evento, antes do mapa final",
        "consultavel_antes_do_evento": False,
        "fontes": ("cems",),
        "permite": "ordenar pontos na escala da AOI, condicional a ativacao",
        "nao_permite": "consulta de lugar sem ativacao; comparar nivel de AUC com o quadro de vizinhanca",
    },
    "territorio_consultavel": {
        "pergunta": "num lugar delimitado antes do evento, onde a agua chega quando ha cheia?",
        "positivo": "inundado observado dentro da delimitacao",
        "negativo": "observado seco pelo sensor na mesma delimitacao (sem visada nao e negativo)",
        "delimitacao": "independente do desfecho: tile fixo de grade, sub-bacia ou municipio escolhidos antes",
        "delimitacao_usa_o_desfecho": False,
        "consultavel_antes_do_evento": True,
        "fontes": (),
        "permite": "ordenar lugares numa consulta de territorio novo, condicional a cheia ocorrer na delimitacao",
        "nao_permite": "probabilidade de ocorrencia da cheia (a delimitacao ainda e escolhida onde houve evento)",
    },
}


def populacao_da_fonte(fonte: str) -> str:
    for nome, p in POPULACOES_ALVO.items():
        if fonte in p["fontes"]:
            return nome
    raise KeyError(f"fonte sem populacao-alvo declarada: {fonte}")


def leitura_entre_populacoes(treino: str, teste: str) -> dict:
    """O que uma avaliacao treinada numa populacao e lida noutra responde -- sem decidir pelo resultado."""
    a, b = POPULACOES_ALVO[treino], POPULACOES_ALVO[teste]
    if treino == teste:
        return {"responde": a["pergunta"], "ressalva": None}
    return {"responde": b["pergunta"],
            "ressalva": (f"o modelo aprendeu '{a['pergunta']}' e e lido em '{b['pergunta']}': a diferenca mede "
                         "transferencia ENTRE perguntas, nao entre lugares; queda ou ganho nao sao atribuiveis "
                         "so ao instrumento ou a fisica")}


def epoca_jrc_contem(datas) -> pd.Series:
    """O evento esta dentro da epoca do JRC? Se esta, ele pesou na propria occurrence."""
    t = pd.to_datetime(pd.Series(datas).astype("string"), errors="coerce", format="mixed", utc=True).dt.tz_localize(None)
    a, b = (pd.Timestamp(x) for x in JRC["epoca"])
    return (t >= a) & (t <= b)


def composicao_por_corte(occ, classe, cortes=(5, 10, 20)) -> dict:
    """Fracao de positivos e negativos com occurrence >= corte (e < 50). Descritivo: nenhum modelo."""
    o, c = np.asarray(occ, dtype=float), np.asarray(classe)
    return {str(k): {str(cl): round(float(((o >= k) & (o < 50))[c == cl].mean()), 4) for cl in (0, 1)} for k in cortes}


def regime_hidrico(occ) -> pd.Series:
    """Rotulo do regime por ponto; NaN (sem leitura do JRC) fica `desconhecido`, nunca `nunca_agua`."""
    o = pd.Series(occ, dtype=float)
    out = pd.Series("desconhecido", index=o.index, dtype=object)
    out[o == 0] = "nunca_agua"
    out[(o > 0) & (o < 10)] = "agua_rara"
    out[(o >= 10) & (o < 50)] = "agua_recorrente"
    out[o >= 50] = "agua_permanente"
    return out


def composicao_hidrica(d: pd.DataFrame, coluna: str = "jrc_occurrence") -> dict:
    """Fracao de cada regime por classe: descreve o rotulo, nao le saida de modelo."""
    r = regime_hidrico(d[coluna].to_numpy())
    r.index = d.index
    out = {}
    for c in (0, 1):
        s = r[d.classe == c]
        out[str(c)] = {k: round(float((s == k).mean()), 4) for k in
                       ("nunca_agua", "agua_rara", "agua_recorrente", "agua_permanente", "desconhecido")}
        out[str(c)]["linhas"] = int(len(s))
    return out


# --------------------------------------------------------------------------
# 3. Identidade de evento entre fontes.
# --------------------------------------------------------------------------
FOLGA_DIAS = 30
FOLGA_GRAUS = 1.0
PRIORIDADE = ("gfd", "cems", "sen1floods11", "ufo", "uk", "serra02")


def janelas_de_eventos(d: pd.DataFrame, fonte_col: str = "fonte") -> pd.DataFrame:
    """Uma linha por (fonte, evento): inicio, fim e caixa dos pontos."""
    x = d.dropna(subset=["evento_id"])
    ini = pd.to_datetime(x["janela_inicio"].astype("string"), errors="coerce", format="mixed", utc=True)
    fim = pd.to_datetime(x["janela_fim"].astype("string"), errors="coerce", format="mixed", utc=True) \
        if "janela_fim" in x else ini
    x = x.assign(_ini=ini.dt.tz_localize(None), _fim=fim.dt.tz_localize(None))
    e = x.groupby([fonte_col, "evento_id"]).agg(
        inicio=("_ini", "min"), fim=("_fim", "max"), lat_min=("lat", "min"), lat_max=("lat", "max"),
        lon_min=("lon", "min"), lon_max=("lon", "max"), linhas=("lat", "size")).reset_index()
    e["fim"] = e.fim.fillna(e.inicio)
    return e.rename(columns={fonte_col: "fonte"})


def equivalencias(a: pd.DataFrame, b: pd.DataFrame | None = None, folga_dias: int = FOLGA_DIAS,
                  folga_graus: float = FOLGA_GRAUS, mesma_fonte: bool = False) -> list[dict]:
    """Pares (de fontes diferentes) cujas janelas se tocam com folga E cujas caixas se tocam com folga.

    Regra deliberadamente larga: erra para o lado de chamar de "mesmo evento",
    porque o custo de tratar o mesmo evento como independente (vazamento,
    pseudo-replicacao, confirmatorio falso) e maior que o de perder um evento.
    """
    b = a if b is None else b
    out = []
    folga = pd.Timedelta(days=folga_dias)
    for _, x in a.iterrows():
        if pd.isna(x.inicio):
            continue
        for _, y in b.iterrows():
            if pd.isna(y.inicio) or (x.fonte == y.fonte and not mesma_fonte):
                continue
            if x.fonte == y.fonte and x.evento_id == y.evento_id:
                continue
            if b is a and (x.fonte, x.evento_id) >= (y.fonte, y.evento_id):
                continue
            tempo = x.inicio - folga <= y.fim and y.inicio - folga <= x.fim
            espaco = (x.lat_min - folga_graus <= y.lat_max and y.lat_min - folga_graus <= x.lat_max
                      and x.lon_min - folga_graus <= y.lon_max and y.lon_min - folga_graus <= x.lon_max)
            if tempo and espaco:
                out.append({"a": f"{x.fonte}|{x.evento_id}", "b": f"{y.fonte}|{y.evento_id}",
                            "a_janela": [str(x.inicio)[:10], str(x.fim)[:10]],
                            "b_janela": [str(y.inicio)[:10], str(y.fim)[:10]]})
    return out


def canonico(pares: list[dict]) -> dict:
    """Uniao dos pares; o ID canonico e o da fonte de maior prioridade (GFD primeiro)."""
    pai = {}

    def raiz(n):
        pai.setdefault(n, n)
        while pai[n] != n:
            pai[n] = pai[pai[n]]
            n = pai[n]
        return n

    for p in pares:
        pai[raiz(p["a"])] = raiz(p["b"])
    grupos = {}
    for n in list(pai):
        grupos.setdefault(raiz(n), []).append(n)

    def ordem(n):
        f = n.split("|", 1)[0]
        return (PRIORIDADE.index(f) if f in PRIORIDADE else len(PRIORIDADE), n)

    return {n: sorted(ns, key=ordem)[0].split("|", 1)[1] for ns in grupos.values() for n in ns}


def consultados() -> pd.DataFrame:
    """Eventos cujo rotulo/imagem foi LIDO fora da tabela unica: entram na identidade, nao no treino.

    - pacote de serra (serra-02): AOI e dia de cada evento; S1/S2 lidos em 200 pontos.
    - lote GFD que toca o Brasil: eventos com pontos amostrados que nunca entraram no ds05
      (uma classe so); mesmo ID da fonte GFD.
    """
    linhas = []
    if PACOTE_SERRA.exists():
        p = json.loads(PACOTE_SERRA.read_text(encoding="utf-8"))
        for nome, ev in p.get("eventos", {}).items():
            a = ev.get("aoi")
            if not a or not ev.get("dia"):
                continue
            dia = pd.Timestamp(ev["dia"])
            # Vinculo declarado pelo proprio pacote ("cems_emsr720": "... ja na tabela"): evidencia de
            # catalogo, mais forte que coincidencia de tempo e lugar.
            vinc = sorted("CEMS:" + k.split("_", 1)[1].upper() for k in (ev.get("fontes") or {})
                          if re.fullmatch(r"cems_emsr\d+", k))
            linhas.append({"fonte": "serra02", "evento_id": f"SERRA:{nome}", "inicio": dia, "fim": dia,
                           "lon_min": a[0], "lat_min": a[1], "lon_max": a[2], "lat_max": a[3], "linhas": 0,
                           "vinculo_catalogo": vinc,
                           "o_que_foi_lido": "S1/S2 nos pontos do pacote (serra-02/serra-03)"})
    if LOTE_TOCAM_BRASIL.exists():
        lb = json.loads(LOTE_TOCAM_BRASIL.read_text(encoding="utf-8"))
        for e in lb.get("amostrados_agora", []):
            linhas.append({"fonte": "gfd", "evento_id": f"GFD:{e['evento']}", "inicio": pd.NaT, "fim": pd.NaT,
                           "lon_min": np.nan, "lat_min": np.nan, "lon_max": np.nan, "lat_max": np.nan,
                           "linhas": int(e.get("pontos", 0)),
                           "o_que_foi_lido": f"rotulo GFD em {e.get('pontos')} pontos: {e.get('por_classe')}"})
    return pd.DataFrame(linhas)


def chave_espacial(d: pd.DataFrame) -> pd.Series:
    """Tile de 1 grau pela COORDENADA, igual para toda fonte.

    `grupo_validacao` muda de natureza entre fontes (tile no GFD, AOI no CEMS, chip
    no Sen1Floods11): comparar o NOME do grupo entre fontes nunca acha lugar
    compartilhado. Em 06/10/2026 a transferencia GFD -> CEMS retirava 0 linhas por
    isso, com 19% do teste CEMS em tile que o treino GFD ocupava.
    """
    lat = np.floor(d["lat"].to_numpy(float)).astype(int).astype(str)
    lon = np.floor(d["lon"].to_numpy(float)).astype(int).astype(str)
    return pd.Series(np.char.add(np.char.add(lat, "_"), lon), index=d.index)


def qualificar(pares: list[dict], pontos: pd.DataFrame, cons: pd.DataFrame) -> list[dict]:
    """Evidencia de cada par: catalogo, dias sobrepostos, distancia entre AMOSTRAS, tiles comuns.

    Proximidade sozinha nao une eventos como identidade provada: o par e
    `provavel_mesmo_evento` (tempo E extensao se tocam) ou `mesmo_evento_por_catalogo`
    (vinculo declarado). A distancia entre amostras separa "mesmo evento, mesmo pixel"
    de "mesmo evento, lugares distintos" -- dependencia de forcamento, nao de rotulo.
    """
    from sklearn.neighbors import BallTree
    por_ev = {f"{f}|{e}": g[["lat", "lon"]].to_numpy(float) for (f, e), g in pontos.groupby(["fonte", "evento_id"])}
    vinc, caixa = {}, {}
    for r in (cons.to_dict("records") if len(cons) else []):
        chave = f"{r['fonte']}|{r['evento_id']}"
        caixa[chave] = r
        v = r.get("vinculo_catalogo")
        vinc[chave] = set(v) if isinstance(v, list) else set()
    out = []
    for p in pares:
        ea, eb = p["a"].split("|", 1)[1], p["b"].split("|", 1)[1]
        ini = max(pd.Timestamp(p["a_janela"][0]), pd.Timestamp(p["b_janela"][0]))
        fim = min(pd.Timestamp(p["a_janela"][1]), pd.Timestamp(p["b_janela"][1]))
        q = {**p, "dias_sobrepostos": int((fim - ini).days) + 1 if fim >= ini else 0}
        A, B = por_ev.get(p["a"]), por_ev.get(p["b"])
        for lado, chave in (("A", p["a"]), ("B", p["b"])):
            if (A if lado == "A" else B) is None and chave in caixa:
                c = caixa[chave]
                ponto = np.array([[(c["lat_min"] + c["lat_max"]) / 2, (c["lon_min"] + c["lon_max"]) / 2]])
                A, B = (ponto, B) if lado == "A" else (A, ponto)
        if A is not None and B is not None:
            km = BallTree(np.radians(A), metric="haversine").query(np.radians(B), k=1)[0][:, 0] * 6371
            q["km_entre_amostras_mediana"] = round(float(np.median(km)), 1)
            q["tiles_1grau_em_comum"] = len(set(map(tuple, np.floor(A).astype(int)))
                                            & set(map(tuple, np.floor(B).astype(int))))
        ligado = vinc.get(p["a"], set()) | vinc.get(p["b"], set())
        q["status"] = "mesmo_evento_por_catalogo" if ({ea, eb} & ligado) else "provavel_mesmo_evento"
        if q.get("km_entre_amostras_mediana", 0) > 50:
            q["ambiguidade"] = ("amostras a mais de 50 km: mesmo evento provavel em lugares distintos; dependencia de "
                                "forcamento e ano hidrologico, nao rotulo duplicado no mesmo pixel")
        out.append(q)
    return out


def lugar_compartilhado(pontos: pd.DataFrame) -> dict:
    """Mesmo lugar, outro evento: fracao das linhas de B em tile de 1 grau ocupado por A.

    Periodos distintos nao tornam duas fontes independentes no ESPACO: o terreno e o
    mesmo. Isto e dependencia espacial, nao identidade de evento.
    """
    t = pontos.assign(_t=chave_espacial(pontos))
    tiles = {f: set(g._t) for f, g in t.groupby("fonte")}
    return {f"{a}->{b}": round(float(t[t.fonte == b]._t.isin(tiles[a]).mean()), 4)
            for a in tiles for b in tiles if a != b}


def identidade(d: pd.DataFrame | None = None) -> dict:
    """Crosswalk da tabela de treino vigente + consultados, com evidencia e hash."""
    from proveniencia import agora_utc, sha256_json
    if d is None:
        from ds06_dataset_de_treino import versao_atual
        origem = versao_atual()
        d = pd.read_parquet(origem / "pontos.parquet",
                            columns=["fonte", "evento_id", "janela_inicio", "janela_fim", "lat", "lon"])
        dataset = origem.name
    else:
        dataset = "fornecido"
    pontos = d[d.fonte.isin(["gfd", "cems", "sen1floods11", "ufo", "uk"])].dropna(subset=["evento_id"])
    jan = janelas_de_eventos(pontos)
    cons = consultados()
    datados = cons.dropna(subset=["inicio"]) if len(cons) else cons
    pares = equivalencias(jan)
    if len(datados):
        pares += equivalencias(datados, jan)
    pares = qualificar(pares, pontos, datados)
    mapa = canonico(pares)
    faixa = jan.groupby("fonte").agg(inicio=("inicio", "min"), fim=("fim", "max"))
    r = {"tipo": "identidade_de_eventos", "dataset": dataset,
         "regra": {"folga_dias": FOLGA_DIAS, "folga_graus": FOLGA_GRAUS, "prioridade_do_canonico": PRIORIDADE,
                   "status": {"mesmo_evento_por_catalogo": "vinculo declarado por uma das fontes",
                              "provavel_mesmo_evento": "janelas e extensoes se tocam; sem vinculo de catalogo"},
                   "canonico_une": ["mesmo_evento_por_catalogo", "provavel_mesmo_evento"],
                   "por_que_une_o_provavel": "tratar o mesmo evento como independente custa mais que perder um"},
         "faixa_temporal_por_fonte": {f: [str(x.inicio)[:10], str(x.fim)[:10]] for f, x in faixa.iterrows()},
         "pares_equivalentes": pares, "canonico": mapa,
         "lugar_compartilhado_entre_fontes": lugar_compartilhado(pontos),
         "consultados_fora_da_tabela": [] if cons.empty else
         [{k: (str(v)[:10] if isinstance(v, pd.Timestamp) else v) for k, v in r_.items()
           if not (isinstance(v, float) and np.isnan(v))} for r_ in cons.to_dict("records")],
         "nao_e": ("par equivalente nao e identidade provada; e a regra para nunca tratar os dois como "
                   "independentes. Fontes de periodos distintos podem ser eventos distintos no MESMO lugar")}
    r["hash"] = sha256_json(json.loads(json.dumps(r, default=str)))
    r["quando_utc"] = agora_utc()
    return r


def evento_canonico(d: pd.DataFrame, mapa: dict) -> pd.Series:
    """ID canonico por linha; evento sem par fica com o proprio ID."""
    chave = d["fonte"].astype(str) + "|" + d["evento_id"].astype(str)
    return pd.Series([mapa.get(k, e) for k, e in zip(chave, d["evento_id"])], index=d.index, dtype=object)


# --------------------------------------------------------------------------
# Anotacao versionada do ds06: coluna nova SEM reescrever o dataset (os
# consumidores atuais e o rastreio continuam lendo a v7 intacta).
# --------------------------------------------------------------------------
def sha256_arquivo(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 22), b""):
            h.update(bloco)
    return "sha256:" + h.hexdigest()


def anotar_ds06(destino: Path = SAIDA) -> dict:
    from ds06_dataset_de_treino import versao_atual
    from proveniencia import agora_utc
    origem = versao_atual()
    pontos = origem / "pontos.parquet"
    d = pd.read_parquet(pontos, columns=["ponto_id", "fonte", "evento_id", "janela_inicio", "janela_fim",
                                         "lat", "lon", "classe"])
    ident = identidade(d)
    occ = pd.read_csv(PONTOS_GFD, usecols=["ponto_id", "jrc_occurrence"])
    a = d[["ponto_id", "fonte", "evento_id"]].merge(occ, on="ponto_id", how="left")
    a["regime_hidrico"] = regime_hidrico(a.jrc_occurrence.to_numpy()).to_numpy()
    a.loc[a.fonte != "gfd", "regime_hidrico"] = "nao_medido_nesta_fonte"
    a["evento_canonico"] = evento_canonico(a, ident["canonico"]).to_numpy()
    destino.mkdir(parents=True, exist_ok=True)
    alvo = destino / f"anotacoes_ds06_{origem.name}.parquet"
    a[["ponto_id", "evento_canonico", "jrc_occurrence", "regime_hidrico"]].to_parquet(alvo, index=False)
    man = {"tipo": "anotacao_versionada_do_ds06", "quando_utc": agora_utc(), "dataset": origem.name,
           "pontos_sha256": sha256_arquivo(pontos), "pontos_gfd_sha256": sha256_arquivo(PONTOS_GFD),
           "saida": str(alvo.relative_to(REPO)), "saida_sha256": sha256_arquivo(alvo),
           "identidade_hash": ident["hash"], "linhas": int(len(a)),
           "eventos_com_canonico_diferente": int((a.evento_canonico != a.evento_id).groupby(a.evento_id).any().sum()),
           "regime_por_classe_gfd": composicao_hidrica(a.assign(classe=d.classe.to_numpy())[a.fonte == "gfd"]),
           "migracao": ("v7 intacta; esta anotacao junta por ponto_id. Uma ds06 v8 que absorva as colunas "
                        "exige revalidar os nos a jusante no rastreio de impacto")}
    (destino / f"anotacoes_ds06_{origem.name}.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    (destino / "identidade.json").write_text(json.dumps(ident, ensure_ascii=False, indent=1, default=str),
                                             encoding="utf-8")
    return man


def anotar(d: pd.DataFrame, origem: Path | None = None) -> pd.DataFrame:
    """Junta as anotacoes ao ds06; falha se a anotacao foi feita sobre outro arquivo de pontos."""
    from ds06_dataset_de_treino import versao_atual
    origem = origem or versao_atual()
    man = SAIDA / f"anotacoes_ds06_{origem.name}.json"
    if not man.exists():
        raise FileNotFoundError(f"sem anotacao para {origem.name}: rode verdade_observacional.py")
    m = json.loads(man.read_text(encoding="utf-8"))
    if m["pontos_sha256"] != sha256_arquivo(origem / "pontos.parquet"):
        raise RuntimeError(f"anotacao de {origem.name} foi feita sobre outro pontos.parquet; refazer")
    a = pd.read_parquet(REPO / m["saida"])
    return d.merge(a, on="ponto_id", how="left", validate="many_to_one")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    import rastreio_de_impacto as ri
    _entrada, _saida = ri.ganchos("verdade_observacional", "verdade_observacional")
    _entrada()
    m = anotar_ds06()
    print(json.dumps({k: m[k] for k in ("dataset", "linhas", "eventos_com_canonico_diferente",
                                         "regime_por_classe_gfd")}, ensure_ascii=False, indent=1))
    print(f"GRAVADO={SAIDA}")
    _saida()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
