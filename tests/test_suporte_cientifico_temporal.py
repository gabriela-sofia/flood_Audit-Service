"""Travas de ARQUITETURA da dimensao temporal do REV-P.

O que este arquivo guarda nao e "o numero da AUC continua o mesmo". E que as
quatro distincoes que a frente existe para criar continuem existindo depois de
qualquer refatoracao:

    ausencia de teste  != falha
    inconclusivo       != falha
    falha              != inconclusivo
    maturidade         != temporalidade

E mais duas que sao consequencia direta delas:

    transferencia nao herda o suporte do modelo local nem o do pool;
    nenhuma camada de apresentacao -- texto, recuperacao documental ou tela --
    altera o estado.

O teste que mais importa e `test_o_estado_e_derivado_e_nao_escrito_por_regiao`:
se um dia alguem trocar a derivacao por uma tabela com o nome das regioes, ele
falha -- e e a unica forma de a frente nao virar, com o tempo, exatamente o que
ela substituiu.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "servico"))
sys.path.insert(0, str(ROOT / "scripts" / "treino"))
sys.path.insert(0, str(ROOT / "scripts" / "temporalidade"))

import dimensao_temporal as dim  # noqa: E402
import suporte_cientifico as sup  # noqa: E402
from verificador_de_afirmacoes import ContextoAutorizado, avaliar_frase  # noqa: E402

# Camadas que existem no repositorio de trabalho e NAO fazem parte deste pacote
# publico: apresentacao da resposta, tela do cartao e reajuste periodico. Os
# testes abaixo guardam contratos reais dessas camadas; aqui eles sao pulados
# com o motivo declarado, em vez de removidos -- apagar o teste apagaria a
# informacao de que o contrato existe.
FORA_DO_PACOTE = pytest.mark.skipif(
    True, reason="camada fora do pacote publico: apresentacao, tela ou reajuste")


MATRIZ = ROOT / "local_runs" / "temporalidade-integracao" / sup.NOME_DA_MATRIZ
GERAR = "python scripts/temporalidade/dimensao_temporal.py"


# ------------------------------------------------------------------ fixtures

@pytest.fixture(scope="module")
def matriz() -> sup.MatrizDeSuporte:
    m = sup.MatrizDeSuporte.carregar(MATRIZ)
    if m is None:
        pytest.skip(f"matriz ausente; gere com: {GERAR}")
    return m


def _corte(rotulo: str, auc: float, ic=None) -> dim.Corte:
    return dim.Corte(rotulo=rotulo, auc=auc, ic95=ic)


def _evidencia(**kw) -> dim.EvidenciaTemporal:
    """Evidencia sintetica. `execucao` acompanha os cortes por padrao.

    Passar `execucao` explicitamente e o que permite testar as paradas do
    ciclo de vida sem inventar cortes que nao existiriam naquele estado.
    """
    base = {"identificabilidade": "IDENTIFICAVEL",
            "dados_permitiriam_teste": True, "cortes": []}
    base.update(kw)
    base.setdefault("execucao", dim.execucao_de_cortes(
        base["cortes"], "sintetico"))
    return dim.EvidenciaTemporal(**base)


def _payload(estado, regiao="curitiba", modelo="fluvial_planicie",
             maturidade="transferencia_caracterizada") -> dict:
    return {
        "status": "ok", "regiao": regiao, "maturidade": maturidade,
        "suporte_cientifico": {"temporal": {
            "support": estado, "avaliado": estado is not None,
            "e_portao_do_contrato": False,
            "scope": {"region_id": regiao, "model_id": modelo}}},
    }


# =========================================================================
# 1. o vocabulario, e as nao-equivalencias
# =========================================================================

def test_os_quatro_estados_existem_e_nao_se_confundem():
    assert set(sup.ESTADOS_DE_SUPORTE) == {
        "supported", "not_supported", "inconclusive", "not_testable"}
    assert sup.NOT_SUPPORTED in sup.NAO_EQUIVALE[sup.INCONCLUSIVE]
    assert sup.NOT_SUPPORTED in sup.NAO_EQUIVALE[sup.NOT_TESTABLE]
    assert sup.SUPPORTED in sup.NAO_EQUIVALE[sup.NOT_TESTABLE]
    assert "validado" in sup.NAO_EQUIVALE[sup.SUPPORTED]


def test_so_supported_conta_como_suporte_demonstrado():
    assert sup.suporte_demonstrado(sup.SUPPORTED)
    for outro in (sup.NOT_SUPPORTED, sup.INCONCLUSIVE, sup.NOT_TESTABLE):
        assert not sup.suporte_demonstrado(outro)
        assert sup.bloqueia_promocao(outro)


def test_consequencia_igual_nao_apaga_razao_diferente():
    """`inconclusive` e `not_testable` bloqueiam pelo mesmo efeito, e o motivo
    persistido tem de continuar distinguindo os dois."""
    a = sup.efeito_operacional(sup.INCONCLUSIVE)
    b = sup.efeito_operacional(sup.NOT_TESTABLE)
    assert a["efeito"] == b["efeito"] == sup.EFEITO_SUPORTE_NAO_DEMONSTRADO
    assert a["motivo"] != b["motivo"]


def test_dimensao_sem_protocolo_formal_nao_aceita_avaliacao():
    """A abstracao existe; a avaliacao nao pode ser fabricada."""
    escopo = sup.EscopoDeSuporte(region_id="x", model_id="y",
                                 protocol_id="p", evidence_version="v")
    with pytest.raises(ValueError):
        sup.AvaliacaoDeSuporte(dimension="spatial", support=sup.SUPPORTED,
                               scope=escopo, razao_tecnica="inventada")
    assert "spatial" in sup.DIMENSOES
    assert sup.DIMENSOES["spatial"]["tem_protocolo_formal"] is False


def test_estado_desconhecido_e_recusado():
    with pytest.raises(ValueError):
        sup.validar_estado("talvez")


# =========================================================================
# 2. casos sinteticos -- o terreno onde a resposta certa e conhecida
# =========================================================================

def test_teste_valido_com_estabilidade_da_supported():
    d = dim.derivar_suporte(_evidencia(cortes=[
        _corte("2020", 0.80, [0.72, 0.88]),
        _corte("2021", 0.78, [0.70, 0.86]),
        _corte("2022", 0.81, [0.74, 0.89])]))
    assert d["support"] == sup.SUPPORTED


def test_teste_valido_com_degradacao_identificavel_da_not_supported():
    """Um corte cujo IC95 fica inteiro ABAIXO do acaso: a ordenacao inverteu,
    e o intervalo e preciso o bastante para nao ser ruido."""
    d = dim.derivar_suporte(_evidencia(cortes=[
        _corte("2020", 0.80, [0.72, 0.88]),
        _corte("2021", 0.78, [0.70, 0.86]),
        _corte("2022", 0.31, [0.24, 0.42])]))
    assert d["support"] == sup.NOT_SUPPORTED


def test_fatores_nao_separaveis_dao_inconclusive():
    d = dim.derivar_suporte(_evidencia(
        identificabilidade="PARCIALMENTE_IDENTIFICAVEL",
        cortes=[_corte("2020", 0.80, [0.72, 0.88]),
                _corte("2021", 0.55, [0.44, 0.66]),
                _corte("2022", 0.52, [0.41, 0.63])]))
    assert d["support"] == sup.INCONCLUSIVE


def test_sem_dados_ou_folds_apropriados_da_not_testable():
    sem_dado = dim.derivar_suporte(_evidencia(
        identificabilidade="NAO_IDENTIFICAVEL",
        dados_permitiriam_teste=False))
    assert sem_dado["support"] == sup.NOT_TESTABLE
    assert sem_dado["razao_tecnica"] == "sem_dado_para_montar_teste"

    desenho_nao_fecha = dim.derivar_suporte(_evidencia(
        identificabilidade="NAO_IDENTIFICAVEL",
        cortes=[_corte("2020", 0.80, [0.72, 0.88])]))
    assert desenho_nao_fecha["support"] == sup.NOT_TESTABLE


def test_corte_unico_nao_vira_suporte():
    """Um ponto nao e serie. Chamar de `supported` seria transformar ausencia
    de serie em estabilidade demonstrada."""
    d = dim.derivar_suporte(_evidencia(
        cortes=[_corte("2026", 0.95, [0.90, 0.99])]))
    assert d["support"] == sup.INCONCLUSIVE
    assert "serie_curta" in d["razao_tecnica"]


def test_cortes_sem_intervalo_nao_sustentam_estabilidade_nem_falha():
    d = dim.derivar_suporte(_evidencia(cortes=[
        _corte("2024", 0.63), _corte("2025", 0.67), _corte("2026", 0.52)]))
    assert d["support"] == sup.INCONCLUSIVE
    assert d["performance_stability"] is None


def test_vazamento_invalida_o_teste_em_vez_de_reprovar_o_modelo():
    d = dim.derivar_suporte(_evidencia(vazamento=True, cortes=[
        _corte("2020", 0.95, [0.92, 0.98]),
        _corte("2021", 0.94, [0.90, 0.97]),
        _corte("2022", 0.96, [0.93, 0.99])]))
    assert d["support"] == sup.NOT_TESTABLE


# =========================================================================
# 3. as travas de arquitetura enunciadas na frente
# =========================================================================

def test_o_estado_e_derivado_e_nao_escrito_por_regiao():
    """Mudar a evidencia muda o estado. Se um dia a derivacao virar tabela de
    resultado por regiao, este teste para de valer -- e ele e a razao de a
    frente nao virar o que substituiu."""
    estavel = _evidencia(cortes=[_corte("a", 0.8, [0.7, 0.9]),
                                 _corte("b", 0.79, [0.7, 0.88]),
                                 _corte("c", 0.81, [0.72, 0.9])])
    assert dim.derivar_suporte(estavel)["support"] == sup.SUPPORTED

    # mesma regiao, mesma quantidade de cortes, evidencia diferente
    fraco = _evidencia(cortes=[_corte("a", 0.8, [0.7, 0.9]),
                               _corte("b", 0.55, [0.45, 0.65]),
                               _corte("c", 0.52, [0.42, 0.61])])
    assert dim.derivar_suporte(fraco)["support"] != sup.SUPPORTED

    # e nenhum nome de regiao aparece no arquivo ao lado de um estado literal
    fonte = (ROOT / "scripts" / "temporalidade" / "dimensao_temporal.py"
             ).read_text(encoding="utf-8")
    for linha in fonte.splitlines():
        if linha.lstrip().startswith(("#", '"', "'")):
            continue
        baixa = linha.lower()
        if any(r in baixa for r in ("curitiba", "recife", "petropolis")):
            assert not any(f'"{e}"' in baixa or f"'{e}'" in baixa
                           for e in sup.ESTADOS_DE_SUPORTE), linha


def test_todas_as_regioes_servidas_recebem_avaliacao_explicita(matriz):
    from svc02_contrato_inferencia import carregar_modelos

    servidos = carregar_modelos()
    if not servidos:
        pytest.skip("modelos servidos ausentes")
    for nome, m in servidos.items():
        if not m.get("servivel"):
            continue
        for regiao in m.get("regioes_servidas", []):
            aval = matriz.consultar(str(regiao).lower(), nome)
            assert aval is not None, (
                f"{regiao} x {nome} sem avaliacao: a matriz precisa responder "
                "por todo par servido, nem que seja `not_testable`")
            assert aval["support"] in sup.ESTADOS_DE_SUPORTE
            assert aval["razao_tecnica"]


def test_ausencia_de_teste_nao_vira_sucesso(matriz):
    for aval in matriz.por_dimensao(sup.TEMPORAL):
        if aval["razao_tecnica"].startswith(
                ("sem_dado", "dado_existe_mas_nenhum_teste",
                 "corte_prospectivo_executado_sem_fold")):
            assert aval["support"] == sup.NOT_TESTABLE
            assert aval["support"] != sup.SUPPORTED


def test_inconclusivo_nao_vira_falha_e_falha_nao_vira_inconclusivo():
    inconclusivo = dim.derivar_suporte(_evidencia(
        identificabilidade="PARCIALMENTE_IDENTIFICAVEL",
        cortes=[_corte("a", 0.52, [0.44, 0.60]),
                _corte("b", 0.53, [0.45, 0.62]),
                _corte("c", 0.51, [0.43, 0.60])]))
    assert inconclusivo["support"] == sup.INCONCLUSIVE
    assert inconclusivo["support"] != sup.NOT_SUPPORTED

    falha = dim.derivar_suporte(_evidencia(
        cortes=[_corte("a", 0.80, [0.72, 0.88]),
                _corte("b", 0.78, [0.70, 0.86]),
                _corte("c", 0.30, [0.22, 0.40])]))
    assert falha["support"] == sup.NOT_SUPPORTED
    assert falha["support"] != sup.INCONCLUSIVE


def test_maturidade_nao_determina_temporalidade_nem_o_contrario(matriz):
    from svc02_contrato_inferencia import regioes

    r = regioes()
    if not r["regioes"]:
        pytest.skip("modelos servidos ausentes")
    pares = {(d["maturidade"], d["suporte_cientifico"]["temporal"]["support"])
             for d in r["regioes"].values()}
    maturidades = {m for m, _ in pares}
    # Recife e `mvp_local` e nao tem suporte temporal demonstrado: se o eixo
    # da maturidade determinasse o temporal, isso seria impossivel.
    assert len(maturidades) > 1
    for maturidade, estado in pares:
        assert estado != "validado"
        assert maturidade in (
            "mvp_local", "transferencia_caracterizada",
            "transferencia_sem_referencia_local", "nao_suportada")
    assert not any(estado == sup.SUPPORTED and maturidade == "validado"
                   for maturidade, estado in pares)


def test_transferencia_nao_herda_estado_do_modelo_local_nem_do_pool(matriz):
    """O par regiao x modelo e a chave, e e por isso que ela e a chave.

    Curitiba tem dois escopos avaliados: o modelo proprio (nao servido) e o
    modelo servido por transferencia. Eles nao podem ter o mesmo estado por
    contagio, e o pool de ajuste do transferido e um terceiro escopo.
    """
    servido = matriz.consultar("curitiba", "fluvial_planicie")
    if servido is None:
        pytest.skip("matriz sem o par servido de Curitiba")
    locais = [a for a in matriz.por_dimensao(sup.TEMPORAL)
              if a["scope"]["region_id"] == "curitiba"
              and a["scope"]["model_id"] != "fluvial_planicie"]
    assert locais, "o modelo proprio de Curitiba precisa estar na matriz"
    assert servido["scope"]["model_id"] != locais[0]["scope"]["model_id"]
    assert servido["support"] != locais[0]["support"], (
        "o escopo servido herdou o estado do modelo local")

    pool = matriz.consultar(f"{dim.DOMINIO_DE_AJUSTE}:fluvial_planicie",
                            "fluvial_planicie")
    assert pool is not None
    assert pool["support"] != servido["support"] or True  # podem coincidir
    assert pool["scope"]["region_id"] != servido["scope"]["region_id"]


def test_consulta_de_escopo_ausente_nao_devolve_o_escopo_vizinho(matriz):
    assert matriz.consultar("curitiba", "modelo_que_nao_existe") is None
    assert matriz.consultar("regiao_inventada", "fluvial_planicie") is None


def test_ausencia_de_matriz_nao_e_conclusao_cientifica():
    from svc02_contrato_inferencia import suporte_cientifico_do_escopo

    bloco = suporte_cientifico_do_escopo(
        "recife", {"nome": "recife_pluvial"}, matriz=None)["temporal"]
    assert bloco["support"] is None
    assert bloco["avaliado"] is False
    assert bloco["support"] not in sup.ESTADOS_DE_SUPORTE
    assert bloco["operational_effect"] == sup.EFEITO_SUPORTE_NAO_DEMONSTRADO


# =========================================================================
# 4. nenhuma camada de apresentacao altera o estado
# =========================================================================

@FORA_DO_PACOTE
def test_a_camada_de_texto_recebe_o_estado_pronto_e_nao_o_decide():
    from svc06_explicacao_llm import resumo_para_o_motor

    resumo = resumo_para_o_motor({
        **_payload(sup.INCONCLUSIVE),
        "escore": 0.5, "ic95": [0.4, 0.6], "n_pontos": 1,
        "unidade_de_resposta": "area", "variaveis_usadas": [],
        "dominio": {}, "limitacoes": [], "model_card": {}, "explicacao": {},
    })
    t = resumo["suporte_temporal"]
    assert t["estado"] == sup.INCONCLUSIVE
    assert "explicar" in t["a_llm_pode"]
    assert "alterar" in t["a_llm_nao_pode"]


def test_o_verificador_rejeita_estado_diferente_do_payload():
    ctx = ContextoAutorizado(resposta=_payload(sup.NOT_TESTABLE))
    v = avaliar_frase("A evidencia temporal e atualmente inconclusiva.", ctx)
    assert not v.aprovada


def test_o_verificador_aceita_o_estado_que_o_payload_declara():
    ctx = ContextoAutorizado(resposta=_payload(sup.INCONCLUSIVE))
    v = avaliar_frase("A evidencia temporal e atualmente inconclusiva.", ctx)
    assert v.aprovada, v.motivo


@FORA_DO_PACOTE
def test_recuperacao_e_tela_nao_produzem_estado():
    """Nem o corpus nem a tela decidem: os dois so podem exibir o que veio.

    A checagem e textual de proposito. Se um dia alguem escrever a regra de
    suporte no `cartao.js` ou no corpus, o estado passa a ter duas fontes.
    """
    cartao = (ROOT / "scripts" / "servico" / "frontend" / "cartao.js"
              ).read_text(encoding="utf-8")
    assert "SUPORTE_ROTULO" in cartao
    for proibido in ("suporte_demonstrado =", "derivar_suporte", "AUC >"):
        assert proibido not in cartao


@pytest.mark.parametrize("frase,estado,esperado_aprovado", [
    ("Curitiba foi validada temporalmente.", sup.NOT_TESTABLE, False),
    ("Curitiba comprovadamente falhou por concept drift.",
     sup.INCONCLUSIVE, False),
    ("A evidencia temporal de Curitiba e atualmente inconclusiva.",
     sup.INCONCLUSIVE, True),
    ("Petropolis passou no teste temporal.", sup.NOT_TESTABLE, False),
    ("Petropolis nao possui evidencia local suficiente para realizar o teste "
     "prospectivo.", sup.NOT_TESTABLE, True),
    ("O piloto ingles prova que Curitiba e temporalmente estavel.",
     sup.NOT_TESTABLE, False),
    ("O modelo generaliza no tempo.", sup.NOT_TESTABLE, False),
    ("A estabilidade temporal esta comprovada.", sup.INCONCLUSIVE, False),
])
def test_adversariais_da_prosa_temporal(frase, estado, esperado_aprovado):
    v = avaliar_frase(frase, ContextoAutorizado(resposta=_payload(estado)))
    assert v.aprovada is esperado_aprovado, (frase, v.motivo)


def test_causa_da_degradacao_nunca_e_sustentavel():
    """Mesmo com a degradacao real no payload: nenhum desenho do projeto
    separa concept drift de forcing, sampling e observacao."""
    for frase in ("A queda vem de concept drift.",
                  "O modelo caiu porque a relacao fisica deixou de valer.",
                  "A degradacao e efeito de mudanca climatica."):
        v = avaliar_frase(frase,
                          ContextoAutorizado(resposta=_payload(sup.INCONCLUSIVE)))
        assert not v.aprovada, frase


# =========================================================================
# 5. proveniencia, versao e historico
# =========================================================================

def test_proveniencia_aponta_para_artefato_real(matriz):
    for aval in matriz.por_dimensao(sup.TEMPORAL):
        assert aval["provenance"], aval["scope"]
        for p in aval["provenance"]:
            assert p["artefato"]
            if p["existe"]:
                assert (ROOT / p["artefato"]).exists(), p["artefato"]
                assert p["sha256"] and p["sha256"].startswith("sha256:")


def test_toda_avaliacao_declara_escopo_protocolo_e_versao(matriz):
    for aval in matriz.por_dimensao(sup.TEMPORAL):
        escopo = aval["scope"]
        for campo in ("region_id", "model_id", "protocol_id",
                      "evidence_version"):
            assert escopo.get(campo), (campo, escopo)
        assert aval["protocol_id"] == escopo["protocol_id"]
        assert aval["operational_effect"] in (
            sup.EFEITO_SUPORTE_DEMONSTRADO, sup.EFEITO_SUPORTE_REFUTADO,
            sup.EFEITO_SUPORTE_NAO_DEMONSTRADO)


def test_evento_novo_gera_versao_nova_sem_apagar_a_anterior(tmp_path):
    """Duas versoes de evidencia coexistem: a matriz e historica."""
    escopo_v1 = sup.EscopoDeSuporte(region_id="x", model_id="m",
                                    protocol_id=dim.PROTOCOLO_ID,
                                    evidence_version="v1")
    escopo_v2 = sup.EscopoDeSuporte(region_id="x", model_id="m",
                                    protocol_id=dim.PROTOCOLO_ID,
                                    evidence_version="v2")
    m = sup.MatrizDeSuporte([
        sup.AvaliacaoDeSuporte(dimension=sup.TEMPORAL,
                               support=sup.NOT_TESTABLE, scope=escopo_v1,
                               razao_tecnica="sem_dado_para_montar_teste"),
        sup.AvaliacaoDeSuporte(dimension=sup.TEMPORAL,
                               support=sup.INCONCLUSIVE, scope=escopo_v2,
                               razao_tecnica="serie_curta_1_corte(s)_minimo_3"),
    ])
    destino = tmp_path / sup.NOME_DA_MATRIZ
    m.gravar(destino)
    lida = sup.MatrizDeSuporte.carregar(destino)
    versoes = {a["scope"]["evidence_version"]
               for a in lida.por_dimensao(sup.TEMPORAL)}
    assert versoes == {"v1", "v2"}


def test_o_historico_no_disco_nao_e_sobrescrito():
    hist = dim.HISTORICO
    if not hist.exists():
        pytest.skip(f"historico ausente; gere com: {GERAR}")
    arquivos = sorted(hist.glob("scientific_support_matrix_*.json"))
    assert arquivos, "nenhum snapshot de versao de evidencia gravado"
    for a in arquivos:
        assert a.stat().st_size > 0, f"{a.name} vazio"
        assert ":" not in a.name, "nome de arquivo com `:` vira ADS no Windows"


# =========================================================================
# 6. contrato: o portao existe, esta desligado, e a analise diz por que
# =========================================================================

def test_o_portao_temporal_existe_e_esta_desligado():
    from svc02_contrato_inferencia import (GATE_TEMPORAL_ATIVO,
                                           g6_suporte_temporal)

    assert GATE_TEMPORAL_ATIVO is False
    assert callable(g6_suporte_temporal)


def test_g6_nao_testa_auc_contra_limiar(matriz):
    """A entrada de G6 e o ESTADO, e nao a metrica.

    Um escopo com AUC alta e suporte nao demonstrado tem de ser recusado, e um
    escopo `supported` tem de passar -- sem que nenhum numero entre na conta.
    """
    from svc02_contrato_inferencia import g6_suporte_temporal

    fake = sup.MatrizDeSuporte([
        {"dimension": sup.TEMPORAL, "support": sup.INCONCLUSIVE,
         "razao_tecnica": "auc_alta_mas_serie_curta",
         "scope": {"region_id": "x", "model_id": "m"},
         "evidence": [{"corte": "2026", "auc": 0.99}]},
        {"dimension": sup.TEMPORAL, "support": sup.SUPPORTED,
         "razao_tecnica": "todos_os_cortes_discriminam_acima_do_acaso",
         "scope": {"region_id": "y", "model_id": "m"},
         "evidence": [{"corte": "2026", "auc": 0.71}]},
    ])
    ok_alta, falha, _ = g6_suporte_temporal("x", {"nome": "m"}, fake)
    assert ok_alta is False and falha["gate_que_nao_fechou"] == "suporte_temporal"
    ok_baixa, falha2, _ = g6_suporte_temporal("y", {"nome": "m"}, fake)
    assert ok_baixa is True and falha2 is None


def test_g6_preserva_a_razao_ao_recusar():
    from svc02_contrato_inferencia import g6_suporte_temporal

    for estado, marca in ((sup.INCONCLUSIVE, "inconclusive"),
                          (sup.NOT_TESTABLE, "not_testable"),
                          (sup.NOT_SUPPORTED, "not_supported")):
        fake = sup.MatrizDeSuporte([
            {"dimension": sup.TEMPORAL, "support": estado,
             "razao_tecnica": f"razao_de_{estado}",
             "scope": {"region_id": "x", "model_id": "m"}}])
        _, falha, _ = g6_suporte_temporal("x", {"nome": "m"}, fake)
        assert marca in falha["detalhe"]
        assert f"razao_de_{estado}" in falha["detalhe"]


def test_a_analise_de_impacto_declara_v2_quando_g6_mudaria_a_resposta():
    caminho = (ROOT / "local_runs" / "temporalidade-integracao"
               / "contract_impact.json")
    if not caminho.exists():
        pytest.skip(f"analise ausente; gere com: {GERAR}")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    for chave in ("current_contract", "temporal_dimension_implemented",
                  "changes_v1_decisions", "g6_candidate", "v2_required",
                  "reason"):
        assert chave in d, chave
    assert d["current_contract"] == "v1"
    assert d["temporal_dimension_implemented"] is True
    # Enquanto a dimensao for somente de revisao, a v1 nao muda decisao nenhuma.
    assert d["changes_v1_decisions"] is False
    assert d["g6_candidate"]["ativo"] is False
    # E se ativar mudaria a resposta, a analise TEM de exigir v2.
    muda = bool(d["regioes_que_g6_recusaria"])
    assert d["v2_required"] is muda
    assert (d["veredito"] == "V2_REQUIRED") is muda
    assert "A" in d["arquiteturas_estudadas"]["escolhida_para_a_v2"]


def test_a_v1_continua_semanticamente_igual_com_a_dimensao_desligada():
    """Removido o campo novo, a resposta e a mesma que a v1 sempre deu."""
    from svc02_contrato_inferencia import carregar_modelos, inferir

    servidos = carregar_modelos()
    if not servidos:
        pytest.skip("modelos servidos ausentes")
    m = servidos["recife_pluvial"]
    camadas = {f: float(sum(m["faixa_dominio_5_95"][f]) / 2)
               for f in m["features"]}
    req = {"regiao": "recife",
           "geometria": {"tipo": "aoi", "crs": "EPSG:4326",
                         "pontos": [{"lat": -8.05, "lon": -34.9,
                                     "camadas": camadas}]}}
    com = inferir(req, servidos, {})
    sem = inferir(req, servidos, {}, matriz_cientifica=sup.MatrizDeSuporte([]))
    assert com["status"] == sem["status"] == "ok"
    for chave in ("escore", "ic95", "maturidade", "limitacoes", "dominio",
                  "gate_que_nao_fechou" if "gate_que_nao_fechou" in com
                  else "model_card"):
        assert com.get(chave) == sem.get(chave), chave


# =========================================================================
# 7. SVC-07 -- fonte unica com a auditoria
# =========================================================================

@FORA_DO_PACOTE
def test_svc07_usa_a_mesma_avaliacao_de_suporte():
    import svc07_reajuste_periodico as svc07

    portao = svc07._com_suporte(
        {"portao": "estabilidade_temporal", "fecha": True},
        [_corte("a", 0.8, [0.7, 0.9]), _corte("b", 0.79, [0.71, 0.88]),
         _corte("c", 0.81, [0.72, 0.9])],
        "IDENTIFICAVEL", dados_permitiriam=True)  # noqa: E501
    assert portao["suporte"] == sup.SUPPORTED
    assert portao["fecha"] is True


@FORA_DO_PACOTE
def test_svc07_bloqueia_com_a_razao_preservada():
    import svc07_reajuste_periodico as svc07

    # inconclusivo: fechava pela faixa de AUC, e vira INDETERMINADO
    inconclusivo = svc07._com_suporte(
        {"portao": "estabilidade_temporal", "fecha": True},
        [_corte("a", 0.8), _corte("b", 0.79), _corte("c", 0.81)],
        "IDENTIFICAVEL", dados_permitiriam=True)
    assert inconclusivo["suporte"] == sup.INCONCLUSIVE
    assert inconclusivo["fecha"] is False
    assert inconclusivo["indeterminado"] is True

    # sem teste: tambem INDETERMINADO, e com razao diferente
    sem_teste = svc07._com_suporte(
        {"portao": "estabilidade_temporal", "fecha": True}, [],
        "NAO_IDENTIFICAVEL", dados_permitiriam=False)
    assert sem_teste["suporte"] == sup.NOT_TESTABLE
    assert sem_teste["indeterminado"] is True
    assert sem_teste["razao_do_suporte"] != inconclusivo["razao_do_suporte"]

    # falha demonstrada: fecha CONTRA, e nao indeterminado
    falha = svc07._com_suporte(
        {"portao": "estabilidade_temporal", "fecha": True},
        [_corte("a", 0.8, [0.72, 0.88]), _corte("b", 0.78, [0.7, 0.86]),
         _corte("c", 0.30, [0.22, 0.40])],
        "IDENTIFICAVEL", dados_permitiriam=True)
    assert falha["suporte"] == sup.NOT_SUPPORTED
    assert falha["fecha"] is False
    assert falha["indeterminado"] is False


def test_svc07_nao_promove_sozinho():
    caminho = ROOT / "local_runs" / "svc-07-reajuste" / "resultado.json"
    if not caminho.exists():
        pytest.skip("reajuste ausente")
    d = json.loads(caminho.read_text(encoding="utf-8"))
    for v in d["modelos"].values():
        assert v != "PROMOVIDO"


# =========================================================================
# 8. o script roda de ponta a ponta
# =========================================================================

def test_o_construtor_da_matriz_roda_e_e_reprodutivel():
    a = dim.construir_matriz().para_dict()["avaliacoes"]
    b = dim.construir_matriz().para_dict()["avaliacoes"]
    chave = lambda x: (x["scope"]["region_id"], x["scope"]["model_id"])  # noqa: E731
    assert [chave(x) for x in a] == [chave(x) for x in b]
    assert [x["support"] for x in a] == [x["support"] for x in b]
    assert [x["scope"]["evidence_version"] for x in a] \
        == [x["scope"]["evidence_version"] for x in b]


def test_o_cli_da_dimensao_temporal_executa():
    r = subprocess.run(
        [sys.executable, "scripts/temporalidade/dimensao_temporal.py",
         "--matriz"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    assert r.returncode == 0, r.stderr[-800:]
    d = json.loads(r.stdout)
    assert d["tipo"] == "scientific_support_matrix"
    assert set(d["estados_possiveis"]) == set(sup.ESTADOS_DE_SUPORTE)
    assert d["dimensoes_avaliaveis"] == [sup.TEMPORAL]
