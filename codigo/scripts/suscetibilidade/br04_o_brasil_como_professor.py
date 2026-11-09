"""O Brasil como professor: o que o negativo observado brasileiro ensina.

-------------------------------------------------------------------------
POR QUE ESTE ARQUIVO REPETE O `rs03` COM OUTRO ALUNO
-------------------------------------------------------------------------

O `rs03_peso_da_temporalidade` mediu o Rio Grande do Sul como professor: cinco
AOIs de 2024-05-02 treinando, 119 desastres posteriores testando. Aquilo
respondeu "quanto um evento brasileiro sozinho ensina sobre o mundo".

O negativo observado brasileiro do Global Flood Database muda tres coisas ao
mesmo tempo, e cada uma abre uma pergunta que o RS nao podia responder:

    escala       24 tiles em vez de 5 AOIs, e eles cobrem as CINCO regioes
                 brasileiras -- Norte, Nordeste, Centro-Oeste, Sudeste e Sul
    tempo        16 janelas de evento entre 2002 e 2017, contra a data unica
                 do RS. O Brasil deixa de ter so o instante de 2024
    contexto     o RS e uma cheia fluvial subtropical; o GFD brasileiro tem
                 pulso amazonico, sertao semiarido, Pantanal e planalto

-------------------------------------------------------------------------
AS CINCO MEDIDAS, E O QUE CADA UMA RESPONDE
-------------------------------------------------------------------------

    A. o Brasil ensina o mundo
       treina SO no `gfd` e testa no `cems` e no `sen1floods11`. Esses dois
       sao delineacao de analista sobre radar de 1,5 a 20 m, e o `gfd` e
       MODIS de 250 m: se a relacao passa, ela passa atravessando instrumento
       e sete a vinte e dois anos.

    B. o mundo ensina o Brasil
       o inverso, e ele e a pergunta do produto: hoje toda regiao brasileira
       sem referencia local recebe escore por transferencia de conjunto
       estrangeiro. Aqui isso deixa de ser suposicao e vira numero.

    C. uma regiao brasileira ensina as outras quatro
       leave-one-region-out sobre N, NE, CO, SE e S. E a pergunta da tese em
       forma mensuravel: o mecanismo aprendido num contexto reaparece noutro?
       Um modelo que so funciona na regiao em que foi ajustado nao aprendeu
       fenomeno -- aprendeu lugar.

    D. cortes prospectivos dentro do Brasil
       treina no que veio antes de cada janela e testa no que veio depois.
       Com 16 janelas em 15 anos, ha serie de verdade -- e o RS nao pode
       fazer isso porque ele e um instante.

    E. o Brasil ensina as regioes que o produto serve
       Recife, Curitiba e Sao Paulo recebem escore hoje por transferencia. A
       pergunta e se o negativo observado brasileiro ensina melhor do que o
       conjunto estrangeiro que ensina hoje.

-------------------------------------------------------------------------
O QUE A MEDIDA E RESPONDEU, E POR QUE ELA FICA AQUI
-------------------------------------------------------------------------

Medido em 14/09/2026, AUC agregada, com o mundo entre parenteses:

    Sao Paulo     0,6726   (mundo 0,6836)
    Curitiba      0,5354   (mundo 0,5066)
    Recife        0,4569   (mundo 0,4690)

O `gfd` NAO ensina melhor: nas tres o resultado empata com o professor
estrangeiro dentro de alguns milesimos, e em Recife os dois ficam abaixo do
acaso. Isso nao contradiz o A, o C e o D -- explica o limite deles.

O que transfere entre A, C e D e uma relacao AREAL: cheia que ocupa planicie,
medida contra chao observado seco. Recife, Curitiba e Sao Paulo sao chamado de
central com negativo por AUSENCIA de registro, e o fenomeno e alagamento de
rua. A medicao nacional em `br02` mostrou o mesmo por outra via: de 2.771
municipios com as duas classes observadas, apenas 112 tem area construida
molhada visivel a 250 m.

E os sinais fisicos ficam 3/3 NAS TRES, inclusive em Recife com AUC 0,4569 --
o modelo continua dizendo o certo sobre o terreno, e o rotulo do aluno e que
mede outra coisa. Reportar so o A, o C e o D seria escolher o que favorece.

-------------------------------------------------------------------------
CRITERIOS FIXADOS ANTES DE ESTIMAR
-------------------------------------------------------------------------

  * sinal fisico esperado: `hand_m` negativo, `slope_deg` negativo,
    `twi_dinf` positivo -- os mesmos tres do `rs03` e do `rs02`;
  * o padronizador vem do TREINO e e aplicado ao teste, sempre;
  * cada lado de um corte precisa das duas classes e de ao menos 3 grupos;
  * a AUC sai nas duas formas -- agregada sobre linhas e mediana POR GRUPO.
    A agregada deixa um tile grande decidir; a por grupo responde "em quantos
    lugares isto funcionou", que e a pergunta do produto;
  * `elevation_m` fica fora do vetor pelo mesmo motivo do `rs02`: entre
    conjuntos sem baseline comum a elevacao absoluta mede o pais.

-------------------------------------------------------------------------
A REGIAO VEM DA MALHA DO IBGE, E NAO DE UM PREFIXO
-------------------------------------------------------------------------

Cada ponto e atribuido a uma UF por ponto-em-poligono sobre a malha oficial,
e a macrorregiao vem da UF. Nao ha lista fixa de codigo aqui, e nao ha
inferencia por nome de tile: `gfd_tile_S22W053` e uma coordenada, e a
pergunta "que regiao e essa" e respondida pela geometria -- a mesma regra que
o `regioes.py` impos ao servico.

-------------------------------------------------------------------------
O QUE ISTO NAO E
-------------------------------------------------------------------------

Nao promove o `gfd` a modelo servido, nao muda maturidade de regiao nenhuma e
nao tira ninguem de `not_testable`: quem faz isso e a matriz de sustentacao a
partir do protocolo. Aqui se mede e se declara.

E o resultado do B nao autoriza dizer que uma regiao brasileira "esta
validada": ele mede transferencia entre CONJUNTOS, com rotulo de 250 m de um
lado. A distancia de dominio continua tendo de viajar junto com o numero.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

UNICA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_v2.parquet"
MALHA = REPO / "local_runs" / "br-02-negativo-nacional" / "malha_uf.geojson"
SAIDA = REPO / "local_runs" / "br-04-professor"

FEATS = ["slope_deg", "hand_m", "twi_dinf"]
ESPERADO = {"slope_deg": -1, "hand_m": -1, "twi_dinf": +1}
MINIMO_GRUPOS_POR_LADO = 3

REGIAO_DA_UF = {
    "AC": "N", "AP": "N", "AM": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N",
    "AL": "NE", "BA": "NE", "CE": "NE", "MA": "NE", "PB": "NE", "PE": "NE",
    "PI": "NE", "RN": "NE", "SE": "NE",
    "DF": "CO", "GO": "CO", "MT": "CO", "MS": "CO",
    "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE",
    "PR": "S", "RS": "S", "SC": "S",
}
# O codigo do IBGE e a chave da malha; a sigla vem daqui, e este par nao e
# "lista fixa de regiao" no sentido que o projeto recusa -- e a divisao
# politico-administrativa oficial, que nao muda com download.
UF_DO_CODIGO = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}


def _ajustar(tr: pd.DataFrame):
    from sklearn.linear_model import LogisticRegression
    mu, sd = tr[FEATS].mean(), tr[FEATS].std()
    X = ((tr[FEATS] - mu) / sd.replace(0, 1)).to_numpy()
    m = LogisticRegression(max_iter=4000, class_weight="balanced")
    m.fit(X, tr.classe.to_numpy().astype(int))
    return m, mu, sd


def _prever(m, mu, sd, te: pd.DataFrame) -> np.ndarray:
    return m.predict_proba(
        ((te[FEATS] - mu) / sd.replace(0, 1)).to_numpy())[:, 1]


def _auc_por_grupo(te: pd.DataFrame, p: np.ndarray, coluna: str) -> dict:
    from sklearn.metrics import roc_auc_score
    t = te.assign(_p=p)
    vals = [roc_auc_score(g.classe, g._p) for _, g in t.groupby(coluna)
            if g.classe.nunique() > 1]
    if not vals:
        return {"grupos": 0}
    a = np.array(vals)
    return {
        "grupos": int(len(a)),
        "mediana": round(float(np.median(a)), 4),
        "media": round(float(a.mean()), 4),
        "ic95": [round(float(np.percentile(a, 2.5)), 4),
                 round(float(np.percentile(a, 97.5)), 4)],
        "acima_do_acaso": int((a > 0.5).sum()),
        "pct_acima_do_acaso": round(100 * float((a > 0.5).mean()), 1),
    }


def _medir(tr: pd.DataFrame, te: pd.DataFrame, coluna_grupo: str) -> dict:
    from sklearn.metrics import roc_auc_score
    if tr.classe.nunique() < 2 or te.classe.nunique() < 2:
        return {"motivo": "um dos lados tem uma classe so"}
    m, mu, sd = _ajustar(tr)
    p = _prever(m, mu, sd, te)
    coefs = {f: round(float(x), 4) for f, x in zip(FEATS, m.coef_[0])}
    return {
        "treino": {"linhas": int(len(tr)),
                   "grupos": int(tr.grupo_cv.nunique()),
                   "positivos": int((tr.classe == 1).sum())},
        "teste": {"linhas": int(len(te)),
                  "grupos": int(te.grupo_cv.nunique()),
                  "positivos": int((te.classe == 1).sum())},
        "auc_agregada": round(float(roc_auc_score(te.classe, p)), 4),
        "auc_por_grupo": _auc_por_grupo(te, p, coluna_grupo),
        "coeficientes": coefs,
        "sinal_bate_com_a_fisica": {
            f: bool((coefs[f] > 0) == (ESPERADO[f] > 0)) for f in FEATS},
        "fisica_sustentada": sum(
            (coefs[f] > 0) == (ESPERADO[f] > 0) for f in FEATS),
    }


def com_regiao(d: pd.DataFrame) -> pd.DataFrame:
    """UF por ponto-em-poligono sobre a malha oficial; macrorregiao pela UF."""
    from shapely.geometry import Point, shape
    from shapely.strtree import STRtree

    if not MALHA.exists():
        d["uf"] = None
        d["regiao_br"] = None
        return d
    g = json.loads(MALHA.read_text(encoding="utf-8"))
    geoms = [shape(f["geometry"]) for f in g["features"]]
    cods = [f["properties"]["codarea"] for f in g["features"]]
    tree = STRtree(geoms)
    pts = [Point(x, y) for x, y in zip(d.lon.values, d.lat.values)]
    idx = tree.query(pts, predicate="within")
    uf = np.array([None] * len(d), dtype=object)
    uf[idx[0]] = [UF_DO_CODIGO.get(cods[j]) for j in idx[1]]
    d = d.copy()
    d["uf"] = uf
    d["regiao_br"] = [REGIAO_DA_UF.get(u) if u else None for u in uf]
    return d


def carregar() -> pd.DataFrame:
    d = pd.read_parquet(UNICA)
    d = d[(d.admitido == True) & d.classe.isin([0, 1])]  # noqa: E712
    d = d.dropna(subset=FEATS + ["grupo_cv", "lat", "lon"]).copy()
    d["dt"] = pd.to_datetime(d.data_evento.astype("string"),
                             format="mixed", errors="coerce", utc=True)
    return d


def main() -> int:
    if not UNICA.exists():
        print(f"ABORTADO: {UNICA} ausente")
        return 1
    d = carregar()
    br = com_regiao(d[d.fonte == "gfd"])
    mundo = d[d.fonte.isin(["cems", "sen1floods11"])]
    if br.empty:
        print("ABORTADO: nenhum ponto gfd admitido na tabela unica")
        return 1

    print(f"Brasil (gfd)  : {len(br):,} linhas, {br.grupo_cv.nunique()} tiles, "
          f"{br.dt.dt.date.nunique()} janelas "
          f"({br.dt.dt.date.min()} -> {br.dt.dt.date.max()})")
    print(f"mundo areal   : {len(mundo):,} linhas, "
          f"{mundo.grupo_cv.nunique()} grupos, "
          f"{sorted(mundo.fonte.unique())}")
    por_regiao = (br.groupby("regiao_br")
                  .agg(linhas=("classe", "size"),
                       tiles=("grupo_cv", "nunique"),
                       positivos=("classe", "sum")))
    print("\npor macrorregiao:")
    print(por_regiao.to_string())

    # A. o Brasil ensina o mundo
    a = _medir(br, mundo, "grupo_cv")
    # B. o mundo ensina o Brasil
    b = _medir(mundo, br, "grupo_cv")

    # C. leave-one-region-out dentro do Brasil
    c = {}
    for reg in sorted(x for x in br.regiao_br.dropna().unique()):
        tr = br[br.regiao_br != reg]
        te = br[br.regiao_br == reg]
        if (tr.grupo_cv.nunique() < MINIMO_GRUPOS_POR_LADO
                or te.grupo_cv.nunique() < 1):
            c[reg] = {"motivo": "grupos insuficientes"}
            continue
        c[reg] = _medir(tr, te, "grupo_cv")

    # E. o Brasil ensina as regioes que o produto serve
    #
    # E aqui que "professor" deixa de ser figura de linguagem: Recife,
    # Curitiba e Sao Paulo recebem escore HOJE por transferencia de conjunto
    # estrangeiro, e nenhuma delas tem negativo observado. O `gfd` e a
    # primeira fonte brasileira de negativo OBSERVADO com cobertura nacional,
    # e a pergunta e se ele ensina melhor do que o mundo ensina.
    #
    # O tipo de negativo do aluno viaja junto com o numero, porque ele muda o
    # que a AUC significa: onde o negativo e `ausencia`, a classe 0 e "ninguem
    # registrou", e a metrica mede ordenamento contra FUNDO, e nao contra chao
    # observado seco.
    e = {}
    for regiao in ("recife", "curitiba", "sao_paulo"):
        te = d[d.fonte == regiao]
        if te.empty or te.classe.nunique() < 2:
            e[regiao] = {"motivo": "sem as duas classes na tabela"}
            continue
        r = _medir(br, te, "grupo_cv")
        if "motivo" not in r:
            r["negativo_do_aluno"] = te.nivel_negativo.value_counts().to_dict()
            r["comparacao_mundo"] = _medir(
                mundo, te, "grupo_cv").get("auc_agregada")
        e[regiao] = r

    # D. cortes prospectivos dentro do Brasil
    dcortes = []
    for corte in sorted(br.dt.dt.date.dropna().unique())[1:]:
        tr, te = br[br.dt.dt.date < corte], br[br.dt.dt.date >= corte]
        if (tr.grupo_cv.nunique() < MINIMO_GRUPOS_POR_LADO
                or te.grupo_cv.nunique() < MINIMO_GRUPOS_POR_LADO):
            continue
        r = _medir(tr, te, "grupo_cv")
        if "motivo" in r:
            continue
        r["corte"] = str(corte)
        dcortes.append(r)

    def linha(nome, r):
        if "motivo" in r:
            print(f"  {nome:<26} {r['motivo']}")
            return
        pg = r["auc_por_grupo"]
        print(f"  {nome:<26} treino {r['treino']['grupos']:>4} grupos -> "
              f"teste {r['teste']['grupos']:>4} | agregada "
              f"{r['auc_agregada']:.4f} | por grupo {pg.get('mediana')} "
              f"({pg.get('acima_do_acaso')}/{pg.get('grupos')} acima do acaso)"
              f" | fisica {r['fisica_sustentada']}/3")

    print("\nA. O BRASIL ENSINA O MUNDO")
    linha("gfd -> cems+sen1floods11", a)
    print("\nB. O MUNDO ENSINA O BRASIL")
    linha("cems+sen1floods11 -> gfd", b)
    print("\nC. UMA REGIAO ENSINA AS OUTRAS (leave-one-region-out)")
    for reg, r in c.items():
        linha(f"4 regioes -> {reg}", r)
    print("\nE. O BRASIL ENSINA AS REGIOES SERVIDAS")
    for reg, r in e.items():
        linha(f"gfd -> {reg}", r)
        if "motivo" not in r:
            print(f"  {'':26} mundo -> {reg}: {r.get('comparacao_mundo')} "
                  f"| negativo do aluno: {r.get('negativo_do_aluno')}")

    print(f"\nD. CORTES PROSPECTIVOS NO BRASIL ({len(dcortes)} avaliaveis)")
    for r in dcortes:
        linha(r["corte"], r)

    man = {
        "tipo": "o_brasil_como_professor_do_negativo_observado",
        "quando_utc": agora_utc(),
        "features": FEATS,
        "sinal_esperado": ESPERADO,
        "minimo_grupos_por_lado": MINIMO_GRUPOS_POR_LADO,
        "brasil": {"linhas": int(len(br)), "tiles": int(br.grupo_cv.nunique()),
                   "janelas": int(br.dt.dt.date.nunique()),
                   "faixa": [str(br.dt.dt.date.min()),
                             str(br.dt.dt.date.max())],
                   "por_macrorregiao": {
                       k: {"linhas": int(v.linhas), "tiles": int(v.tiles),
                           "positivos": int(v.positivos)}
                       for k, v in por_regiao.iterrows()}},
        "mundo": {"linhas": int(len(mundo)),
                  "grupos": int(mundo.grupo_cv.nunique()),
                  "fontes": sorted(mundo.fonte.unique().tolist())},
        "A_brasil_ensina_o_mundo": a,
        "B_o_mundo_ensina_o_brasil": b,
        "C_leave_one_region_out": c,
        "D_cortes_prospectivos": dcortes,
        "E_brasil_ensina_as_regioes_servidas": e,
        "o_que_isto_e": (
            "transferencia medida entre o negativo observado brasileiro e o "
            "resto do pool areal, e entre as macrorregioes brasileiras"),
        "nao_e": (
            "nao promove o gfd a modelo servido, nao muda maturidade de "
            "regiao nenhuma e nao tira ninguem de not_testable. O resultado "
            "de B nao autoriza dizer que uma regiao brasileira esta validada: "
            "mede transferencia entre CONJUNTOS, com rotulo de 250 m de um "
            "lado, e a distancia de dominio viaja junto com o numero"),
    }
    man["hash_resumo"] = sha256_json(man)
    SAIDA.mkdir(parents=True, exist_ok=True)
    (SAIDA / "professor.json").write_text(
        json.dumps(man, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGRAVADO={SAIDA}/professor.json")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
