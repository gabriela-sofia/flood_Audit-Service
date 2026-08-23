"""
SVC-02 -- O contrato de inferencia: portoes, escore com IC, model card, explicacao.

O QUE ESTE ARQUIVO E, e o que ele nao e:

e a implementacao, dentro do REV-P, do contrato que a Secao II do plano
descreve -- "a requisicao declara geometria, CRS, periodo e camadas; a resposta
devolve status, maturidade da regiao, escore com intervalo, variaveis usadas e
limitacoes, sempre junto de um model card". Ate agora esse contrato existia como
texto e como MVP no repositorio privado; aqui ele vira funcao pura, auditavel e
testavel, sem rede e sem servidor.

NAO e um servidor HTTP. A escolha e deliberada: o que precisa ser auditado e o
CONTRATO -- quais portoes existem, em que ordem, o que cada recusa significa e
como o escore e construido. Envolver isso em FastAPI e trabalho de transporte,
nao de metodo, e traria dependencia de rede para dentro de um repositorio que e
fail-closed por regra. `inferir()` e a funcao que um servidor chamaria.

-------------------------------------------------------------------------
OS PORTOES, na ordem em que sao avaliados
-------------------------------------------------------------------------

A ordem importa: cada portao so e avaliado se o anterior fechou, e a resposta
nomeia o PRIMEIRO que falhou. Recusar por "faltou HAND" quando a geometria nem
era valida esconderia o erro real de quem chamou.

  G1 geometria_valida        ha pelo menos um ponto, CRS declarado e suportado,
                             latitude e longitude finitas e dentro do globo.
                             Falha -> insufficient_data.
  G2 regiao_resolvida        a regiao vem declarada na requisicao ou e deduzida
                             pela caixa envolvente das regioes que existem na
                             tabela unica. Falha -> region_not_supported.
  G3 modelo_para_a_regiao    existe modelo servivel mapeado para a regiao.
                             Falha -> region_not_supported, nunca escore por
                             analogia.
  G4 variaveis_presentes     toda variavel do modelo esta presente e finita em
                             todo ponto. Falha -> insufficient_data.
  G5 dominio_coberto         quantas variaveis da requisicao caem dentro da
                             faixa 5-95% que o modelo viu no ajuste. Acima do
                             limite de extrapolacao -> insufficient_data; abaixo
                             dele, entra como limitacao declarada.

G5 existe porque `metodo_aplicacao_sem_rotulo_local_v1.md` mediu que 0% dos
pontos de Curitiba caiam na faixa de `elevation_m` do treino -- diferenca de
2,76 desvios. Aquilo era uma recomendacao num documento; aqui e um portao que o
servico avalia por requisicao.

-------------------------------------------------------------------------
MATURIDADE DA REGIAO -- o eixo de predicao seletiva
-------------------------------------------------------------------------

  validado                    modelo proprio, sinais fisicos corretos, IC das
                              causais sem cruzar zero e AUC agrupada dentro da
                              faixa declarada.
  mvp_local                   modelo proprio da regiao, mas algum criterio de
                              leitura nao atingido. O escore sai com a falha
                              declarada -- ver POLITICA_CRITERIO_NAO_ATINGIDO.
  transferencia_caracterizada a regiao nao entrou no treino em nenhuma linha; o
                              que se pode declarar e a distancia de dominio, nao
                              acerto contra rotulo local.
  nao_suportada               sem modelo: region_not_supported.

POLITICA_CRITERIO_NAO_ATINGIDO decide o que fazer quando o modelo da regiao
existe mas nao atinge o criterio de leitura fixado antes (por exemplo, AUC
abaixo da faixa 0,70-0,88). Duas posturas defensaveis:

  "declara"  devolve o escore com a falha escrita em `limitacoes` e maturidade
             rebaixada a mvp_local. Coerente com "resultado negativo e
             publicado" -- esconder o escore esconderia tambem o quanto ele e
             fraco.
  "recusa"   devolve insufficient_data. Coerente com fail-closed estrito.

O padrao e "declara", e a escolha esta AQUI, exposta, porque ela muda o que o
produto responde para Recife: com a fonte de chuva corrigida o LOO-AUC de
Recife e 0,6339, abaixo da faixa. Trocar esta constante e uma decisao da
Gabriela, nao um detalhe de implementacao.

-------------------------------------------------------------------------
O ESCORE, o intervalo e a unidade de resposta
-------------------------------------------------------------------------

A unidade que o contrato devolve e a AREA, nunca o pixel -- "validar no mesmo
grao que se entrega e o que torna o resultado auditavel". O escore da AOI e a
media dos escores dos seus pontos.

O IC vem das replicas de bootstrap gravadas pelo SVC-01: cada replica de
coeficiente projeta um escore para cada ponto, a media da AOI e recalculada por
replica, e o intervalo e o percentil 2,5-97,5 dessas medias. Nao ha reajuste em
tempo de requisicao.

-------------------------------------------------------------------------
A CAMADA DE EXPLICACAO
-------------------------------------------------------------------------

Gera por REGRAS, lendo o payload ja decidido. O plano declarou essa rota como
alternativa caso a explicacao divergisse do payload; adota-la desde o inicio
torna a divergencia impossivel por construcao, e nao por verificacao posterior.
A contribuicao de cada variavel e `coeficiente_padronizado x z(valor)`, medida
no mesmo espaco em que o modelo decide, e a frase so descreve o sinal dessa
contribuicao. Nao ha texto livre nem modelo de linguagem envolvido.

Uso:
    python scripts/servico/svc02_contrato_inferencia.py --demonstracao
    python scripts/servico/svc02_contrato_inferencia.py --requisicao arquivo.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
from ds03_esquema_alvo import VERSAO  # noqa: E402

RUNS = REPO / "local_runs"
MODELOS = RUNS / "svc-01-modelos"
UNICA = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.csv"
OUT = RUNS / "svc-02-contrato"

CRS_SUPORTADOS = ("EPSG:4326",)
VERSAO_CONTRATO = "v1"
POLITICA_CRITERIO_NAO_ATINGIDO = "declara"   # ou "recusa" -- ver cabecalho
MAX_PCT_EXTRAPOLACAO = 50.0                  # acima disto, G5 recusa

STATUS_OK = "ok"
STATUS_SEM_DADO = "insufficient_data"
STATUS_SEM_REGIAO = "region_not_supported"

NOME_LEGIVEL = {
    "hand_m": "altura acima da drenagem (HAND)",
    "twi_dinf": "indice topografico de umidade (TWI)",
    "elevation_m": "elevacao",
    "slope_deg": "declividade",
    "rain_max_24h": "pico de chuva em 24 h",
    "rain_decay_index": "chuva antecedente com decaimento",
}
UNIDADE = {"hand_m": "m", "twi_dinf": "", "elevation_m": "m",
           "slope_deg": "graus", "rain_max_24h": "mm", "rain_decay_index": "mm"}


# ------------------------------------------------------------ carregamento

def carregar_modelos(diretorio: Path = MODELOS) -> dict:
    if not diretorio.exists():
        return {}
    modelos = {}
    for p in sorted(diretorio.glob("*.json")):
        if p.name == "indice.json":
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        modelos[m["nome"]] = m
    return modelos


def caixas_das_regioes(caminho: Path = UNICA) -> dict:
    """Caixa envolvente de cada regiao que existe na tabela unica.

    Regiao sem linha na base -- Petropolis -- nao ganha caixa inventada: ela so
    e resolvida se a requisicao declarar `regiao` explicitamente. Deduzir
    fronteira de municipio a partir de coordenada que o projeto nao tem seria
    inventar dado dentro do servico.
    """
    import pandas as pd

    if not caminho.exists():
        return {}
    d = pd.read_csv(caminho, usecols=["fonte", "lat", "lon", "admitido"],
                    low_memory=False)
    d = d[d.admitido == True]  # noqa: E712
    caixas = {}
    for fonte in ("recife", "curitiba"):
        s = d[d.fonte == fonte]
        if s.empty:
            continue
        caixas[fonte] = {"lat_min": float(s.lat.min()), "lat_max": float(s.lat.max()),
                         "lon_min": float(s.lon.min()), "lon_max": float(s.lon.max())}
    return caixas


def regioes_com_referencia_local(caminho: Path = UNICA) -> set:
    """Regioes que tem ao menos um ponto rotulado na tabela unica.

    Nao e o mesmo que "regiao validada": Curitiba tem rotulo e o projeto decidiu
    nao usa-lo como criterio de aprovacao. E o que separa "da para verificar
    depois" de "nao ha nem o que verificar".
    """
    import pandas as pd

    if not caminho.exists():
        return set()
    d = pd.read_csv(caminho, usecols=["fonte", "classe", "admitido"],
                    low_memory=False)
    d = d[(d.admitido == True) & d.classe.isin([0, 1])]  # noqa: E712
    return set(d.fonte.unique())


def modelo_da_regiao(regiao: str, modelos: dict) -> dict | None:
    for m in modelos.values():
        if regiao in m.get("regioes_servidas", []) and m.get("servivel"):
            return m
    return None


# ------------------------------------------------------------------ portoes

def _falha(gate: str, status: str, detalhe: str, extra: dict | None = None) -> dict:
    r = {"status": status, "gate_que_nao_fechou": gate, "detalhe": detalhe,
         "escore": None, "contrato": VERSAO_CONTRATO}
    if extra:
        r.update(extra)
    return r


def g1_geometria(req: dict) -> tuple[bool, dict | None, list]:
    geo = req.get("geometria") or {}
    crs = geo.get("crs")
    if crs not in CRS_SUPORTADOS:
        return False, _falha("geometria_valida", STATUS_SEM_DADO,
                             f"CRS {crs!r} nao suportado; suportados: "
                             f"{list(CRS_SUPORTADOS)}"), []
    pontos = geo.get("pontos") or []
    if not pontos:
        return False, _falha("geometria_valida", STATUS_SEM_DADO,
                             "nenhum ponto na geometria"), []
    for i, p in enumerate(pontos):
        try:
            lat, lon = float(p["lat"]), float(p["lon"])
        except (KeyError, TypeError, ValueError):
            return False, _falha("geometria_valida", STATUS_SEM_DADO,
                                 f"ponto {i} sem lat/lon numericos"), []
        if not (np.isfinite(lat) and np.isfinite(lon)):
            return False, _falha("geometria_valida", STATUS_SEM_DADO,
                                 f"ponto {i} com coordenada nao finita"), []
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return False, _falha("geometria_valida", STATUS_SEM_DADO,
                                 f"ponto {i} fora do globo"), []
    return True, None, pontos


def g2_regiao(req: dict, pontos: list, caixas: dict) -> tuple[bool, dict | None, str]:
    declarada = (req.get("regiao") or "").strip().lower()
    if declarada:
        return True, None, declarada
    for nome, c in caixas.items():
        if all(c["lat_min"] <= float(p["lat"]) <= c["lat_max"]
               and c["lon_min"] <= float(p["lon"]) <= c["lon_max"] for p in pontos):
            return True, None, nome
    return False, _falha("regiao_resolvida", STATUS_SEM_REGIAO,
                         "geometria nao cai em nenhuma regiao com modelo, e a "
                         "requisicao nao declarou `regiao`"), ""


def g3_modelo(regiao: str, modelos: dict) -> tuple[bool, dict | None, dict]:
    m = modelo_da_regiao(regiao, modelos)
    if m is None:
        return False, _falha("modelo_para_a_regiao", STATUS_SEM_REGIAO,
                             f"sem modelo ajustado e validado para {regiao!r}; "
                             "nenhum escore por analogia",
                             {"regiao": regiao,
                              "maturidade": "nao_suportada"}), {}
    return True, None, m


def g4_variaveis(pontos: list, modelo: dict) -> tuple[bool, dict | None, np.ndarray]:
    feats = modelo["features"]
    linhas = []
    for i, p in enumerate(pontos):
        camadas = p.get("camadas", p)
        linha = []
        for f in feats:
            v = camadas.get(f)
            if v is None or not np.isfinite(float(v)):
                return False, _falha(
                    "variaveis_presentes", STATUS_SEM_DADO,
                    f"variavel {f!r} ausente ou nao finita no ponto {i}",
                    {"variaveis_exigidas": feats}), np.empty(0)
            linha.append(float(v))
        linhas.append(linha)
    return True, None, np.array(linhas, dtype=float)


def g5_dominio(X: np.ndarray, modelo: dict) -> tuple[bool, dict | None, dict]:
    feats = modelo["features"]
    faixa = modelo["faixa_dominio_5_95"]
    dentro, fora = {}, []
    for j, f in enumerate(feats):
        lo, hi = faixa[f]
        pct = 100.0 * float(np.mean((X[:, j] >= lo) & (X[:, j] <= hi)))
        dentro[f] = round(pct, 1)
        if pct < MAX_PCT_EXTRAPOLACAO:
            fora.append(f)
    pct_variaveis_fora = 100.0 * len(fora) / len(feats)
    detalhe = {"cobertura_por_variavel_pct": dentro,
               "variaveis_em_extrapolacao": fora}
    if pct_variaveis_fora > MAX_PCT_EXTRAPOLACAO:
        return False, _falha("dominio_coberto", STATUS_SEM_DADO,
                             f"{len(fora)} de {len(feats)} variaveis fora da "
                             "faixa 5-95% do ajuste: o escore seria extrapolacao",
                             detalhe), detalhe
    return True, None, detalhe


# -------------------------------------------------------------------- escore

def escore_com_ic(X: np.ndarray, modelo: dict) -> dict:
    """Escore da AREA e IC95 pelas replicas de bootstrap gravadas no SVC-01."""
    feats = modelo["features"]
    mu = np.array([modelo["padronizacao"][f]["media"] for f in feats])
    sd = np.array([modelo["padronizacao"][f]["desvio"] for f in feats])
    Z = (X - mu) / sd

    beta = np.array([modelo["coef"][f] for f in feats])
    eta = modelo["intercepto"] + Z @ beta
    p = 1.0 / (1.0 + np.exp(-eta))
    escore = float(np.mean(p))

    rep = np.array(modelo.get("replicas_bootstrap") or [])
    if rep.size:
        etas = rep[:, 0][None, :] + Z @ rep[:, 1:].T           # (n_pontos, n_rep)
        medias = np.mean(1.0 / (1.0 + np.exp(-etas)), axis=0)  # (n_rep,)
        lo, hi = np.percentile(medias, [2.5, 97.5])
        ic = [round(float(lo), 4), round(float(hi), 4)]
        n_rep = int(rep.shape[0])
    else:
        ic, n_rep = None, 0

    return {"escore": round(escore, 4), "ic95": ic, "n_replicas": n_rep,
            "escore_por_ponto": [round(float(v), 4) for v in p],
            "z": Z, "contribuicoes": (Z * beta)}


def maturidade_da_regiao(regiao: str, modelo: dict,
                         com_referencia_local: set | None = None) -> str:
    """Quatro niveis, e a diferenca entre os dois de transferencia importa.

    Curitiba e Petropolis sao ambas servidas por modelo que nao as viu, mas nao
    sao o mesmo caso: Curitiba tem inventario local (que o projeto decidiu nao
    usar como criterio de aprovacao, `metodo_aplicacao_sem_rotulo_local_v1.md`),
    e Petropolis nao tem nenhum ponto rotulado. Chamar as duas de
    `transferencia_caracterizada` esconderia que numa delas existe um caminho
    para verificar depois, e na outra nao existe nem isso.
    """
    if regiao not in modelo.get("regioes_servidas", []):
        return "nao_suportada"
    treinado_na_regiao = regiao in {f.lower() for f in modelo.get("fontes_do_ajuste", {})}
    if treinado_na_regiao:
        return "validado" if not modelo.get("falhas_de_criterio") else "mvp_local"
    if com_referencia_local is not None and regiao not in com_referencia_local:
        return "transferencia_sem_referencia_local"
    return "transferencia_caracterizada"


# --------------------------------------------------------------- explicacao

def explicar(feats: list[str], X: np.ndarray, contribuicoes: np.ndarray,
             escore: float) -> dict:
    """Le o payload ja decidido e descreve o sinal de cada contribuicao.

    Nao ha texto livre: cada frase e montada da mesma grandeza que entrou no
    escore, entao divergir do payload e impossivel por construcao.
    """
    medias_contrib = contribuicoes.mean(axis=0)
    medias_valor = X.mean(axis=0)
    ordem = np.argsort(-np.abs(medias_contrib))
    itens = []
    for j in ordem:
        f = feats[j]
        c = float(medias_contrib[j])
        v = float(medias_valor[j])
        itens.append({
            "variavel": f, "nome": NOME_LEGIVEL.get(f, f),
            "valor_medio": round(v, 3), "unidade": UNIDADE.get(f, ""),
            "contribuicao": round(c, 4),
            "sentido": "a favor" if c > 0 else ("contra" if c < 0 else "neutro"),
        })
    frases = [f"{i['nome']} = {i['valor_medio']}{(' ' + i['unidade']).rstrip()} "
              f"pesa {i['sentido']}" for i in itens if i["sentido"] != "neutro"]
    return {
        "gerador": "regras_sobre_o_payload",
        "escore_explicado": round(escore, 4),
        "itens": itens,
        "texto": ("; ".join(frases) + ".") if frases else
                 "nenhuma variavel se afasta da media do ajuste.",
    }


def model_card(modelo: dict, regiao: str, maturidade: str) -> dict:
    d = modelo.get("desempenho", {})
    return {
        "modelo": modelo["nome"],
        "regiao": regiao,
        "maturidade": maturidade,
        "mecanismo": modelo.get("mecanismo"),
        "variaveis": modelo["features"],
        "estimador": "regressao logistica penalizada de Firth",
        "ajustado_em": {
            "n": modelo.get("n"), "grupos": modelo.get("grupos"),
            "grupos_positivos": modelo.get("grupos_positivos"),
            "grupos_negativos": modelo.get("grupos_negativos"),
            "fontes": modelo.get("fontes_do_ajuste"),
            "niveis_de_negativo": modelo.get("niveis_de_negativo"),
            "prevalencia": modelo.get("prevalencia_do_ajuste"),
        },
        "desempenho": {"auc_agrupada": d.get("auc_cv"), "folds": d.get("folds")},
        "criterios_nao_atingidos": modelo.get("falhas_de_criterio", []),
        "limites_de_uso": modelo.get("limitacoes_declaradas", []),
        "por_que_este_conjunto": modelo.get("por_que_este_conjunto"),
        "ic_do_escore": ("bootstrap do preditor linear reamostrando grupos, "
                         f"{modelo.get('n_replicas', 0)} replicas"),
        "nao_e": ("nao e previsao de evento nem mapa operacional: e escore de "
                  "predisposicao do terreno, com as limitacoes acima"),
    }


# ------------------------------------------------------------------ contrato

def inferir(requisicao: dict, modelos: dict | None = None,
            caixas: dict | None = None,
            com_referencia_local: set | None = None) -> dict:
    """A funcao que um servidor chamaria. Sem rede, sem estado, sem efeito."""
    modelos = carregar_modelos() if modelos is None else modelos
    caixas = caixas_das_regioes() if caixas is None else caixas
    if com_referencia_local is None:
        com_referencia_local = regioes_com_referencia_local()

    ok, falha, pontos = g1_geometria(requisicao)
    if not ok:
        return falha
    ok, falha, regiao = g2_regiao(requisicao, pontos, caixas)
    if not ok:
        return falha
    ok, falha, modelo = g3_modelo(regiao, modelos)
    if not ok:
        return falha
    ok, falha, X = g4_variaveis(pontos, modelo)
    if not ok:
        return falha
    ok, falha, dominio = g5_dominio(X, modelo)
    if not ok:
        return falha

    maturidade = maturidade_da_regiao(regiao, modelo, com_referencia_local)
    limitacoes = list(modelo.get("limitacoes_declaradas", []))
    if dominio["variaveis_em_extrapolacao"]:
        limitacoes.append(
            "extrapolacao de dominio em "
            f"{dominio['variaveis_em_extrapolacao']}: a requisicao esta fora da "
            "faixa 5-95% que o modelo viu no ajuste")
    if modelo.get("falhas_de_criterio"):
        limitacoes.append(
            "criterio de leitura nao atingido no ajuste: "
            + "; ".join(modelo["falhas_de_criterio"]))
        if POLITICA_CRITERIO_NAO_ATINGIDO == "recusa":
            return _falha("modelo_dentro_dos_criterios", STATUS_SEM_DADO,
                          "o modelo da regiao nao atinge os criterios de leitura "
                          "fixados antes do ajuste",
                          {"regiao": regiao, "maturidade": maturidade,
                           "criterios_nao_atingidos": modelo["falhas_de_criterio"]})

    r = escore_com_ic(X, modelo)
    return {
        "status": STATUS_OK,
        "contrato": VERSAO_CONTRATO,
        "regiao": regiao,
        "maturidade": maturidade,
        "unidade_de_resposta": "area",
        "n_pontos": int(X.shape[0]),
        "escore": r["escore"],
        "ic95": r["ic95"],
        "variaveis_usadas": modelo["features"],
        "dominio": dominio,
        "limitacoes": limitacoes,
        "model_card": model_card(modelo, regiao, maturidade),
        "explicacao": explicar(modelo["features"], X, r["contribuicoes"],
                               r["escore"]),
    }


# ------------------------------------------------------------- demonstracao

def _requisicoes_de_demonstracao(modelos: dict) -> list[tuple[str, dict]]:
    """Uma requisicao por estado do contrato, com valores da propria base."""
    import pandas as pd

    demos = []
    if UNICA.exists():
        d = pd.read_csv(UNICA, low_memory=False)
        d = d[(d.admitido == True) & d.classe.isin([0, 1])]  # noqa: E712
        for regiao in ("recife", "curitiba"):
            s = d[d.fonte == regiao].dropna(
                subset=["lat", "lon", "elevation_m", "slope_deg", "hand_m",
                        "twi_dinf", "rain_max_24h", "rain_decay_index"]).head(12)
            if s.empty:
                continue
            pontos = [{"lat": float(r.lat), "lon": float(r.lon),
                       "camadas": {f: float(getattr(r, f)) for f in
                                   ("elevation_m", "slope_deg", "hand_m",
                                    "twi_dinf", "rain_max_24h",
                                    "rain_decay_index")}}
                      for r in s.itertuples()]
            demos.append((f"{regiao}: area com todas as camadas", {
                "geometria": {"tipo": "aoi", "crs": "EPSG:4326", "pontos": pontos},
                "periodo": {"inicio": "2024-01-01", "fim": "2024-12-31"}}))

    # Petropolis nao tem nenhuma linha na tabela unica, entao os valores vem da
    # grade derivada do proprio raster de terreno (SVC-03). Inventar numero aqui
    # seria pior que nao ter demonstracao -- ja aconteceu uma vez.
    grade = RUNS / "svc-03-grade" / "grade_petropolis.csv"
    if grade.exists():
        g = pd.read_csv(grade, nrows=2000).dropna(subset=["escore"]).head(10)
        if not g.empty:
            pontos = [{"lat": float(r.lat), "lon": float(r.lon),
                       "camadas": {"hand_m": float(r.hand_m)}}
                      for r in g.itertuples()]
            demos.append(("petropolis: area sem nenhum ponto rotulado", {
                "regiao": "petropolis",
                "geometria": {"tipo": "aoi", "crs": "EPSG:4326",
                              "pontos": pontos}}))
    demos.append(("petropolis: sem as camadas que o modelo exige", {
        "regiao": "petropolis",
        "geometria": {"tipo": "aoi", "crs": "EPSG:4326",
                      "pontos": [{"lat": -22.505, "lon": -43.178,
                                  "camadas": {}}]}}))
    demos.append(("variavel faltando", {
        "regiao": "recife",
        "geometria": {"tipo": "ponto", "crs": "EPSG:4326",
                      "pontos": [{"lat": -8.05, "lon": -34.9,
                                  "camadas": {"hand_m": 3.0}}]}}))
    demos.append(("CRS nao suportado", {
        "regiao": "recife",
        "geometria": {"tipo": "ponto", "crs": "EPSG:31985",
                      "pontos": [{"lat": -8.05, "lon": -34.9}]}}))
    demos.append(("geometria fora de toda regiao com modelo", {
        "geometria": {"tipo": "ponto", "crs": "EPSG:4326",
                      "pontos": [{"lat": 48.85, "lon": 2.35,
                                  "camadas": {"hand_m": 1.0, "twi_dinf": 9.0}}]}}))
    return demos


def main() -> int:
    modelos = carregar_modelos()
    if not modelos:
        print(f"ABORTADO: {MODELOS} ausente ou vazio. Gere com: "
              "python scripts/servico/svc01_construir_modelos_servidos.py")
        return 1

    if "--requisicao" in sys.argv:
        p = Path(sys.argv[sys.argv.index("--requisicao") + 1])
        r = inferir(json.loads(p.read_text(encoding="utf-8")), modelos)
        print(json.dumps(r, indent=2, ensure_ascii=False))
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    caixas = caixas_das_regioes()
    print("===SVC-02: CONTRATO DE INFERENCIA===")
    print(f"contrato={VERSAO_CONTRATO}  politica_criterio_nao_atingido="
          f"{POLITICA_CRITERIO_NAO_ATINGIDO}")
    print(f"modelos servidos: "
          f"{[n for n, m in modelos.items() if m.get('servivel')]}")
    print(f"caixas de regiao deduzidas da base: {list(caixas)}")

    respostas = []
    for titulo, req in _requisicoes_de_demonstracao(modelos):
        r = inferir(req, modelos, caixas)
        respostas.append({"caso": titulo, "requisicao": req, "resposta": r})
        print(f"\n--- {titulo} ---")
        print(f"  status={r['status']}", end="")
        if r["status"] == "ok":
            print(f"  regiao={r['regiao']}  maturidade={r['maturidade']}")
            print(f"  escore={r['escore']}  IC95={r['ic95']}  "
                  f"({r['n_pontos']} pontos, unidade={r['unidade_de_resposta']})")
            print(f"  variaveis={r['variaveis_usadas']}")
            print(f"  dominio={r['dominio']['cobertura_por_variavel_pct']}")
            print(f"  explicacao: {r['explicacao']['texto']}")
            for lim in r["limitacoes"]:
                print(f"  limitacao: {lim}")
        else:
            print(f"  gate={r['gate_que_nao_fechou']}")
            print(f"  detalhe: {r['detalhe']}")

    # a demonstracao guarda o payload sem as replicas, que sao do modelo
    (OUT / "respostas_demonstracao.json").write_text(
        json.dumps(respostas, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/servico/svc01_construir_modelos_servidos.py\n"
        "python scripts/servico/svc02_contrato_inferencia.py --demonstracao\n"
        "python -m pytest tests/test_svc_contrato_inferencia.py -q\n",
        encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
