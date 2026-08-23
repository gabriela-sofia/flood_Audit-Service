"""Testes do contrato de inferencia (SVC-01/SVC-02).

O que estes testes protegem:

um contrato que devolve escore errado e mais perigoso que um que quebra, porque
o numero chega ao cliente com aparencia de resposta. Os modos de falha que
importam aqui:

  * devolver escore quando um portao nao fechou -- o oposto de fail-closed;
  * devolver escore por analogia para regiao sem modelo, que e exatamente o que
    `region_not_supported` existe para impedir;
  * a explicacao dizer o contrario do que o escore fez. Aqui isso e impossivel
    por construcao (a frase e montada da mesma contribuicao que entrou no
    escore), e o teste guarda essa propriedade contra qualquer refatoracao;
  * IC estreito demais por reamostrar linha em vez de grupo;
  * a regiao alvo entrar no proprio treino, que anularia o proposito de prever
    onde nao ha inventario local.

A maior parte roda sobre um modelo sintetico, sem depender de `local_runs`. Os
que precisam dos artefatos reais marcam skip com o comando que os gera.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "local_runs" if (ROOT / "local_runs").is_dir() else ROOT / "modelo" / "execucoes"
MODELOS = RUNS / "svc-01-modelos"

sys.path.insert(0, str(ROOT / "scripts" / "servico"))
from svc02_contrato_inferencia import (  # noqa: E402
    CRS_SUPORTADOS, MAX_PCT_EXTRAPOLACAO, POLITICA_CRITERIO_NAO_ATINGIDO,
    STATUS_OK, STATUS_SEM_DADO, STATUS_SEM_REGIAO, VERSAO_CONTRATO,
    carregar_modelos, inferir, maturidade_da_regiao,
)

GERAR = "python scripts/servico/svc01_construir_modelos_servidos.py"


@pytest.fixture(scope="module")
def modelo_sintetico() -> dict:
    """Modelo servivel minimo, com sinais fisicos corretos e replicas."""
    rng = np.random.default_rng(11)
    feats = ["hand_m", "twi_dinf"]
    beta = np.array([-1.2, 0.5])
    rep = [[0.0] + list(beta + rng.normal(0, 0.08, size=2)) for _ in range(400)]
    return {
        "nome": "teste_planicie", "servivel": True, "features": feats,
        "intercepto": 0.0,
        "coef": {"hand_m": -1.2, "twi_dinf": 0.5},
        "ic95_coef": {"hand_m": [-1.4, -1.0], "twi_dinf": [0.3, 0.7]},
        "padronizacao": {"hand_m": {"media": 5.0, "desvio": 3.0},
                         "twi_dinf": {"media": 8.0, "desvio": 2.0}},
        "faixa_dominio_5_95": {"hand_m": [0.5, 12.0], "twi_dinf": [4.0, 13.0]},
        "replicas_bootstrap": rep, "n_replicas": len(rep),
        "desempenho": {"auc_cv": 0.75, "folds": 5},
        "prevalencia_do_ajuste": 0.5, "n": 5000, "grupos": 300,
        "grupos_positivos": 150, "grupos_negativos": 150,
        "fontes_do_ajuste": {"cems": 3000, "uk": 2000},
        "niveis_de_negativo": {"observado": 2000},
        "falhas_de_criterio": [], "veredito": "COERENTE_COM_CRITERIOS",
        "mecanismo": "FLUVIAL_ENXURRADA",
        "regioes_servidas": ["curitiba"],
        "limitacoes_declaradas": ["modelo sintetico de teste"],
    }


@pytest.fixture(scope="module")
def modelos(modelo_sintetico) -> dict:
    return {modelo_sintetico["nome"]: modelo_sintetico}


@pytest.fixture(scope="module")
def caixas() -> dict:
    return {"curitiba": {"lat_min": -25.7, "lat_max": -25.3,
                         "lon_min": -49.4, "lon_max": -49.1}}


def _req(pontos, regiao=None, crs="EPSG:4326"):
    r = {"geometria": {"tipo": "aoi", "crs": crs, "pontos": pontos}}
    if regiao:
        r["regiao"] = regiao
    return r


def _ponto(hand=4.0, twi=9.0, lat=-25.45, lon=-49.25):
    return {"lat": lat, "lon": lon, "camadas": {"hand_m": hand, "twi_dinf": twi}}


# ------------------------------------------------------------------ portoes

def test_crs_nao_suportado_recusa_sem_escore(modelos, caixas):
    r = inferir(_req([_ponto()], crs="EPSG:31985"), modelos, caixas)
    assert r["status"] == STATUS_SEM_DADO
    assert r["gate_que_nao_fechou"] == "geometria_valida"
    assert r["escore"] is None


def test_geometria_vazia_recusa(modelos, caixas):
    r = inferir(_req([]), modelos, caixas)
    assert r["status"] == STATUS_SEM_DADO
    assert r["gate_que_nao_fechou"] == "geometria_valida"


def test_coordenada_fora_do_globo_recusa(modelos, caixas):
    r = inferir(_req([_ponto(lat=120.0)], regiao="curitiba"), modelos, caixas)
    assert r["status"] == STATUS_SEM_DADO
    assert r["gate_que_nao_fechou"] == "geometria_valida"


def test_regiao_desconhecida_nao_recebe_escore_por_analogia(modelos, caixas):
    r = inferir(_req([_ponto(lat=48.85, lon=2.35)]), modelos, caixas)
    assert r["status"] == STATUS_SEM_REGIAO
    assert r["escore"] is None


def test_regiao_sem_modelo_responde_region_not_supported(modelos, caixas):
    r = inferir(_req([_ponto()], regiao="petropolis"), modelos, caixas)
    assert r["status"] == STATUS_SEM_REGIAO
    assert r["gate_que_nao_fechou"] == "modelo_para_a_regiao"
    assert r["maturidade"] == "nao_suportada"
    assert r["escore"] is None


def test_variavel_faltando_recusa_e_diz_qual(modelos, caixas):
    p = _ponto()
    del p["camadas"]["twi_dinf"]
    r = inferir(_req([p], regiao="curitiba"), modelos, caixas)
    assert r["status"] == STATUS_SEM_DADO
    assert r["gate_que_nao_fechou"] == "variaveis_presentes"
    assert "twi_dinf" in r["detalhe"]


def test_variavel_nao_finita_recusa(modelos, caixas):
    r = inferir(_req([_ponto(hand=float("nan"))], regiao="curitiba"),
                modelos, caixas)
    assert r["status"] == STATUS_SEM_DADO
    assert r["gate_que_nao_fechou"] == "variaveis_presentes"


def test_extrapolacao_de_dominio_recusa(modelos, caixas):
    """O achado de Curitiba virando portao: fora da faixa do treino, sem escore."""
    r = inferir(_req([_ponto(hand=900.0, twi=90.0)], regiao="curitiba"),
                modelos, caixas)
    assert r["status"] == STATUS_SEM_DADO
    assert r["gate_que_nao_fechou"] == "dominio_coberto"
    assert set(r["variaveis_em_extrapolacao"]) == {"hand_m", "twi_dinf"}
    assert r["escore"] is None


def test_extrapolacao_parcial_vira_limitacao_e_nao_recusa(modelos, caixas):
    """Uma variavel fora de duas nao passa do limite: entra como limitacao."""
    r = inferir(_req([_ponto(hand=900.0, twi=9.0)], regiao="curitiba"),
                modelos, caixas)
    assert r["status"] == STATUS_OK
    assert any("extrapolacao" in lim for lim in r["limitacoes"])
    assert "hand_m" in r["dominio"]["variaveis_em_extrapolacao"]


def test_nenhum_status_de_recusa_devolve_escore(modelos, caixas):
    casos = [
        _req([_ponto()], crs="EPSG:31985"),
        _req([]),
        _req([_ponto()], regiao="petropolis"),
        _req([_ponto(hand=900.0, twi=90.0)], regiao="curitiba"),
    ]
    for req in casos:
        r = inferir(req, modelos, caixas)
        assert r["status"] != STATUS_OK
        assert r.get("escore") is None
        assert "model_card" not in r


# ------------------------------------------------------------------- escore

def test_escore_e_probabilidade_e_ic_contem_o_ponto(modelos, caixas):
    r = inferir(_req([_ponto(), _ponto(hand=2.0, twi=11.0)], regiao="curitiba"),
                modelos, caixas)
    assert r["status"] == STATUS_OK
    assert 0.0 <= r["escore"] <= 1.0
    lo, hi = r["ic95"]
    assert lo <= r["escore"] <= hi
    assert lo < hi


def test_a_unidade_de_resposta_e_a_area(modelos, caixas):
    """O contrato responde por area; o escore da area e a media dos pontos."""
    from svc02_contrato_inferencia import escore_com_ic

    pontos = [_ponto(hand=2.0), _ponto(hand=8.0), _ponto(hand=5.0)]
    r = inferir(_req(pontos, regiao="curitiba"), modelos, caixas)
    assert r["unidade_de_resposta"] == "area"
    assert r["n_pontos"] == 3
    X = np.array([[p["camadas"]["hand_m"], p["camadas"]["twi_dinf"]]
                  for p in pontos])
    direto = escore_com_ic(X, modelos["teste_planicie"])
    assert r["escore"] == pytest.approx(np.mean(direto["escore_por_ponto"]),
                                        abs=1e-3)


def test_hand_maior_derruba_o_escore(modelos, caixas):
    """Sinal fisico obrigatorio, medido pela resposta e nao pelo coeficiente."""
    baixo = inferir(_req([_ponto(hand=1.0)], regiao="curitiba"), modelos, caixas)
    alto = inferir(_req([_ponto(hand=11.0)], regiao="curitiba"), modelos, caixas)
    assert baixo["escore"] > alto["escore"]


def test_resposta_e_deterministica(modelos, caixas):
    req = _req([_ponto(), _ponto(hand=3.0)], regiao="curitiba")
    a = inferir(req, modelos, caixas)
    b = inferir(req, modelos, caixas)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# --------------------------------------------------------------- explicacao

def test_explicacao_nao_contradiz_o_payload(modelos, caixas):
    r = inferir(_req([_ponto(hand=1.0, twi=12.0)], regiao="curitiba"),
                modelos, caixas)
    for item in r["explicacao"]["itens"]:
        if item["contribuicao"] > 0:
            assert item["sentido"] == "a favor"
        elif item["contribuicao"] < 0:
            assert item["sentido"] == "contra"
        else:
            assert item["sentido"] == "neutro"


def test_explicacao_cobre_exatamente_as_variaveis_usadas(modelos, caixas):
    r = inferir(_req([_ponto()], regiao="curitiba"), modelos, caixas)
    assert ([i["variavel"] for i in r["explicacao"]["itens"]]
            and set(i["variavel"] for i in r["explicacao"]["itens"])
            == set(r["variaveis_usadas"]))


def test_explicacao_ordena_por_peso(modelos, caixas):
    r = inferir(_req([_ponto(hand=1.0, twi=12.0)], regiao="curitiba"),
                modelos, caixas)
    pesos = [abs(i["contribuicao"]) for i in r["explicacao"]["itens"]]
    assert pesos == sorted(pesos, reverse=True)


def test_explicacao_e_gerada_por_regras_e_nao_por_texto_livre(modelos, caixas):
    r = inferir(_req([_ponto()], regiao="curitiba"), modelos, caixas)
    assert r["explicacao"]["gerador"] == "regras_sobre_o_payload"


# ------------------------------------------------------------- modelo card

def test_model_card_acompanha_toda_resposta_ok(modelos, caixas):
    r = inferir(_req([_ponto()], regiao="curitiba"), modelos, caixas)
    mc = r["model_card"]
    for campo in ("modelo", "regiao", "maturidade", "variaveis", "estimador",
                  "ajustado_em", "desempenho", "limites_de_uso", "nao_e"):
        assert campo in mc, campo


def test_model_card_declara_criterio_nao_atingido(modelo_sintetico, caixas):
    m = dict(modelo_sintetico)
    m["falhas_de_criterio"] = ["AUC 0.63 fora da faixa [0.7; 0.88]"]
    mods = {m["nome"]: m}
    r = inferir(_req([_ponto()], regiao="curitiba"), mods, caixas)
    if POLITICA_CRITERIO_NAO_ATINGIDO == "declara":
        assert r["status"] == STATUS_OK
        assert r["model_card"]["criterios_nao_atingidos"]
        assert any("criterio de leitura" in lim for lim in r["limitacoes"])
    else:
        assert r["status"] == STATUS_SEM_DADO


# -------------------------------------------------------------- maturidade

def test_regiao_fora_do_treino_e_transferencia_caracterizada(modelo_sintetico):
    assert maturidade_da_regiao("curitiba", modelo_sintetico) == \
        "transferencia_caracterizada"


def test_regiao_dentro_do_treino_sem_falha_e_validado(modelo_sintetico):
    m = dict(modelo_sintetico)
    m["fontes_do_ajuste"] = {"curitiba": 100}
    assert maturidade_da_regiao("curitiba", m) == "validado"


def test_regiao_dentro_do_treino_com_falha_e_mvp_local(modelo_sintetico):
    m = dict(modelo_sintetico)
    m["fontes_do_ajuste"] = {"curitiba": 100}
    m["falhas_de_criterio"] = ["AUC fora da faixa"]
    assert maturidade_da_regiao("curitiba", m) == "mvp_local"


# --------------------------------------------------- artefatos reais servidos

@pytest.fixture(scope="module")
def servidos() -> dict:
    if not MODELOS.exists():
        pytest.skip(f"{MODELOS.relative_to(ROOT)} ausente -- gere com: {GERAR}")
    m = carregar_modelos(MODELOS)
    if not m:
        pytest.skip(f"nenhum modelo servido -- gere com: {GERAR}")
    return m


def test_modelo_servido_nunca_treina_na_regiao_que_serve_por_transferencia(servidos):
    """A regra de `metodo_aplicacao_sem_rotulo_local`, virada invariante.

    O modelo proprio de uma regiao pode -- e deve -- treinar nela. O que nao
    pode e um modelo declarado como rota de transferencia carregar a regiao
    alvo dentro do proprio ajuste.
    """
    for nome, m in servidos.items():
        if not m.get("servivel") or "planicie" not in nome and "serra" not in nome:
            continue
        fontes = {f.lower() for f in m.get("fontes_do_ajuste", {})}
        for regiao in m.get("regioes_servidas", []):
            assert regiao not in fontes, (
                f"{nome} serve {regiao} por transferencia mas treinou nela")


def test_todo_modelo_servido_respeita_o_orcamento_de_epv(servidos):
    for nome, m in servidos.items():
        if not m.get("servivel"):
            continue
        assert len(m["features"]) <= m["orcamento_estrito"], nome


def test_todo_modelo_servido_guarda_replicas_para_o_ic(servidos):
    for nome, m in servidos.items():
        if not m.get("servivel"):
            continue
        assert m["n_replicas"] >= 100, nome
        assert len(m["replicas_bootstrap"][0]) == len(m["features"]) + 1, nome


def test_todo_modelo_servido_declara_faixa_de_dominio(servidos):
    for nome, m in servidos.items():
        if not m.get("servivel"):
            continue
        for f in m["features"]:
            lo, hi = m["faixa_dominio_5_95"][f]
            assert lo < hi, f"{nome}/{f}"


def test_petropolis_so_responde_sem_referencia_local(servidos):
    """A politica mudou em 20/08/2026, ao executar o E5 -- e o invariante mudou junto.

    Ate entao Petropolis devolvia `region_not_supported`. O E5 desempatou a
    divergencia entre os documentos do projeto: ele manda levar o modelo as tres
    regioes, e a evidencia que exige nao e acerto, e sim que nao se afirme
    acerto onde falta inventario. Entao Petropolis passa a receber escore -- e o
    que este teste guarda e que ele NUNCA saia sem a maturidade que diz que ali
    nao existe nada com que verificar.
    """
    r = inferir(_req([{"lat": -22.505, "lon": -43.178,
                       "camadas": {"hand_m": 12.0, "twi_dinf": 6.0,
                                   "elevation_m": 840.0, "slope_deg": 21.0}}],
                     regiao="petropolis"), servidos, {}, com_referencia_local=set())
    if r["status"] != STATUS_OK:
        assert r["escore"] is None
        return
    assert r["maturidade"] == "transferencia_sem_referencia_local"
    assert any("nunca afirmacao de acerto" in lim or "inventario local" in lim
               for lim in r["limitacoes"]), r["limitacoes"]


def test_regiao_inexistente_continua_sem_escore_por_analogia(servidos):
    r = inferir(_req([{"lat": -3.1, "lon": -60.0, "camadas": {"hand_m": 2.0}}],
                     regiao="manaus"), servidos, {})
    assert r["status"] == STATUS_SEM_REGIAO
    assert r["escore"] is None


def test_contrato_declara_versao(modelos, caixas):
    r = inferir(_req([_ponto()], regiao="curitiba"), modelos, caixas)
    assert r["contrato"] == VERSAO_CONTRATO
    assert "EPSG:4326" in CRS_SUPORTADOS
    assert 0 < MAX_PCT_EXTRAPOLACAO <= 100
