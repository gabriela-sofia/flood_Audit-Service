"""Todo numero publicado tem de sair do artefato que o documento cita.

POR QUE ESTE TESTE EXISTE:

Uma revisao encontrou duas celulas da tabela de recuperadores em `docs/EVIDENCIA.md`
com valores que nao existiam em artefato nenhum -- MRR do hibrido e recall morfologico
do hibrido. O erro nao vinha do codigo nem do dado: vinha de alguem transcrever a mao.

Numero errado numa tabela publica nao e erro de digitacao, e afirmacao sem evidencia.
E revisao humana nao pega esse tipo de erro de forma confiavel: sao dezenas de celulas,
todas plausiveis.

COMO ELE FUNCIONA:

Cada caso declara (documento, rotulo da linha, artefato, caminho dentro do artefato).
O teste le a linha da tabela markdown pelo rotulo, extrai os numeros dela e compara com
o que o artefato tem. Nao ha valor esperado escrito no teste: o esperado E o artefato.
Se o artefato mudar, o documento tem de mudar junto ou a suite falha.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]


def carregar(caminho: str) -> dict:
    return json.loads((RAIZ / caminho).read_text(encoding="utf-8"))


def dentro(obj, *chaves):
    for c in chaves:
        obj = obj[c]
    return obj


def celulas_da_linha(documento: str, rotulo: str,
                    secao: str | None = None) -> list[str]:
    """As celulas da primeira linha de tabela que comeca pelo rotulo dado.

    `secao` recorta a busca a partir de um titulo markdown. Sem ela, um rotulo
    que aparece em duas tabelas do mesmo documento -- `hand_m` aparece na secao
    de Nova York e na de Chicago -- casaria com a tabela errada e o teste
    passaria a comparar a coisa certa com o lugar errado.
    """
    texto = (RAIZ / documento).read_text(encoding="utf-8")
    if secao is not None:
        corte = texto.find(secao)
        assert corte >= 0, f"secao '{secao}' nao encontrada em {documento}"
        texto = texto[corte:]
        fim = texto.find(chr(10) + "## ", 1)
        if fim > 0:
            texto = texto[:fim]
    for linha in texto.splitlines():
        if not linha.startswith("|"):
            continue
        partes = [p.strip() for p in linha.strip().strip("|").split("|")]
        if partes and partes[0].strip("`* ") == rotulo:
            return partes
    onde = f"{documento} ({secao})" if secao else documento
    raise AssertionError(f"linha '{rotulo}' nao encontrada em {onde}")


NUMERO = re.compile(r"[−-]?\d[\d.]*(?:,\d+)?")


def numeros(celula: str) -> list[float]:
    """Numeros de uma celula em convencao pt-BR, ja com o sinal de menos unicode."""
    fora = []
    for bruto in NUMERO.findall(celula.replace("−", "-")):
        s = bruto.replace(".", "").replace(",", ".")
        try:
            fora.append(float(s))
        except ValueError:
            continue
    return fora


def perto(a: float, b: float) -> bool:
    """Compara com a tolerancia da propria casa decimal publicada."""
    return abs(a - b) <= 5e-5 + abs(b) * 1e-9


# =========================================================================
# modelos servidos
# =========================================================================

INDICE = "modelo/execucoes/svc-01-modelos/indice.json"

MODELOS = [
    ("recife_pluvial", "Recife pluvial", "`recife_pluvial`"),
    ("pluvial_urbano_nyc", "Pluvial urbano NYC", "`pluvial_urbano_nyc`"),
    ("pluvial_urbano_sao_paulo", "Pluvial urbano Sao Paulo",
     "`pluvial_urbano_sao_paulo`"),
    ("fluvial_planicie", "Fluvial planicie", "`fluvial_planicie`"),
    ("fluvial_serra", "Fluvial serra", "`fluvial_serra`"),
]


@pytest.mark.parametrize("chave,rotulo_readme,_ev", MODELOS)
def test_auc_n_e_grupos_do_readme_saem_do_indice(chave, rotulo_readme, _ev):
    m = carregar(INDICE)["modelos"][chave]
    celulas = celulas_da_linha("README.md", rotulo_readme)
    # README: | modelo | uso | n | grupos | AUC | veredito |
    assert numeros(celulas[2]) == [float(m["n"])], rotulo_readme
    assert numeros(celulas[3]) == [float(m["grupos"])], rotulo_readme
    assert perto(numeros(celulas[4])[0], m["auc_cv"]), rotulo_readme


@pytest.mark.parametrize("chave,_readme,rotulo_evidencia", MODELOS)
def test_a_evidencia_repete_o_mesmo_indice(chave, _readme, rotulo_evidencia):
    m = carregar(INDICE)["modelos"][chave]
    celulas = celulas_da_linha("docs/EVIDENCIA.md",
                               rotulo_evidencia.strip("`"))
    # EVIDENCIA: | modelo | n | grupos | AUC | servivel | veredito |
    assert numeros(celulas[1]) == [float(m["n"])], rotulo_evidencia
    assert numeros(celulas[2]) == [float(m["grupos"])], rotulo_evidencia
    assert perto(numeros(celulas[3])[0], m["auc_cv"]), rotulo_evidencia
    servivel = celulas[4].strip().lower().startswith("sim")
    assert servivel is bool(m["servivel"]), rotulo_evidencia


@pytest.mark.parametrize("chave,_readme,rotulo_evidencia", MODELOS)
def test_o_cartao_do_modelo_repete_o_mesmo_indice(chave, _readme, rotulo_evidencia):
    m = carregar(INDICE)["modelos"][chave]
    celulas = celulas_da_linha("modelo/MODELO.md", rotulo_evidencia.strip("`"))
    # MODELO: | modelo | servido | regiao | n | grupos | AUC | leitura |
    assert numeros(celulas[3]) == [float(m["n"])], rotulo_evidencia
    assert numeros(celulas[4]) == [float(m["grupos"])], rotulo_evidencia
    assert perto(numeros(celulas[5])[0], m["auc_cv"]), rotulo_evidencia


# =========================================================================
# grade
# =========================================================================

GRADE = "modelo/execucoes/svc-03-grade/resultado.json"


def regiao_da_grade(nome: str) -> dict:
    for r in carregar(GRADE)["regioes"]:
        if r["regiao"] == nome:
            return r
    raise AssertionError(nome)


@pytest.mark.parametrize("nome,rotulo,cenario", [
    ("recife", "Recife", "mediana_observada"),
    ("curitiba", "Curitiba", "unico"),
    ("petropolis", "Petropolis", "unico"),
])
def test_grade_do_readme_sai_do_resultado(nome, rotulo, cenario):
    r = regiao_da_grade(nome)
    c = r["cenarios"][cenario]
    celulas = celulas_da_linha("README.md", rotulo)
    # README: | regiao | validas | serviveis | p50 | p90 | maturidade |
    assert numeros(celulas[1]) == [float(r["celulas_validas"])], rotulo
    assert numeros(celulas[2]) == [float(r["celulas_serviveis"])], rotulo
    assert perto(numeros(celulas[3])[0], c["escore_mediano"]), rotulo
    assert perto(numeros(celulas[4])[0], c["escore_p90"]), rotulo
    assert celulas[5].strip("` ") == r["maturidade"], rotulo


# =========================================================================
# chuva como forcamento
# =========================================================================

FORCAMENTO = "modelo/execucoes/chuva-autopareamento/matriz_suporte_forcamento.json"


def avaliacao_de(regiao: str) -> dict:
    for a in carregar(FORCAMENTO)["avaliacoes"]:
        if a["scope"]["region_id"] == regiao:
            return a
    raise AssertionError(regiao)


@pytest.mark.parametrize("regiao,rotulo", [("nyc", "NYC"),
                                           ("sao_paulo", "Sao Paulo")])
def test_razao_de_risco_da_evidencia_sai_da_matriz(regiao, rotulo):
    a = avaliacao_de(regiao)
    ev = a["evidence"][0]
    celulas = celulas_da_linha("docs/EVIDENCIA.md", rotulo,
                               secao="## 5. Chuva como forcamento")
    # EVIDENCIA: | recorte | RR | IC | estratos | suporte |
    assert perto(numeros(celulas[1])[0], ev["valor"]), rotulo
    baixo, alto = numeros(celulas[2])
    assert perto(baixo, ev["ic95_bootstrap_de_estrato"][0]), rotulo
    assert perto(alto, ev["ic95_bootstrap_de_estrato"][1]), rotulo
    assert numeros(celulas[3]) == [float(ev["estratos_informativos"])], rotulo
    assert celulas[4].strip("` ") == a["support"], rotulo


# =========================================================================
# transferencia entre cidades
# =========================================================================

CHICAGO = "modelo/execucoes/chicago-teste-da-lei/teste_da_lei_chicago.json"

TERMOS = ["hand_m", "twi_dinf", "slope_deg", "elevation_rel_m",
          "mrms_qpe_24h_mm", "twi_dinf_x_chuva", "hand_m_x_chuva"]

ROTULOS = {
    "twi_dinf": "`twi_dinf`",
    "slope_deg": "`slope_deg`",
    "elevation_rel_m": "`elevation_rel_m`",
    "mrms_qpe_24h_mm": "`mrms_qpe_24h_mm`",
    "hand_m": "`hand_m`",
    "twi_dinf_x_chuva": "`twi_dinf` x chuva",
    "hand_m_x_chuva": "`hand_m` x chuva",
}


@pytest.mark.parametrize("termo", TERMOS)
def test_comparacao_nyc_chicago_da_evidencia_sai_do_artefato(termo):
    cmp_ = carregar(CHICAGO)["comparacao_com_nyc"][termo]
    celulas = celulas_da_linha("docs/EVIDENCIA.md", ROTULOS[termo].strip("`"),
                               secao="## 7. O teste da segunda cidade")
    # EVIDENCIA secao 7: | termo | nyc | chicago | mesmo sinal | ICs | aprendido |
    assert perto(numeros(celulas[1])[0], cmp_["nyc"]["coef"]), termo
    assert perto(numeros(celulas[2])[0], cmp_["chicago"]["coef"]), termo
    disse_sim = celulas[3].strip("* ").lower().startswith("sim")
    assert disse_sim is bool(cmp_["mesmo_sinal"]), termo


@pytest.mark.parametrize("termo", ["twi_dinf", "slope_deg", "elevation_rel_m",
                                   "mrms_qpe_24h_mm", "twi_dinf_x_chuva"])
def test_tabela_de_transferencia_do_readme_sai_do_artefato(termo):
    cmp_ = carregar(CHICAGO)["comparacao_com_nyc"][termo]
    celulas = celulas_da_linha("README.md", ROTULOS[termo].strip("`"),
                               secao="| Termo | Nova York |")
    assert perto(numeros(celulas[1])[0], cmp_["nyc"]["coef"]), termo
    assert perto(numeros(celulas[2])[0], cmp_["chicago"]["coef"]), termo


# =========================================================================
# recuperacao de fonte -- a tabela que estava errada
# =========================================================================

RECUPERACAO = "modelo/execucoes/recuperacao/comparacao_recuperadores_k5.json"


@pytest.mark.parametrize("chave,rotulo", [
    ("lexical_fts5", "Lexical FTS5"),
    ("denso_lsa", "Denso LSA"),
    ("hibrido_rrf", "Hibrido RRF"),
])
def test_tabela_de_recuperadores_sai_do_artefato(chave, rotulo):
    m = carregar(RECUPERACAO)["medidas"][chave]
    celulas = celulas_da_linha("docs/EVIDENCIA.md", rotulo,
                               secao="## 9. Recuperacao de fonte")
    # EVIDENCIA: | recuperador | recall | mrr | recall morfologia |
    assert perto(numeros(celulas[1])[0], m["agregado"]["recall_em_k"]), rotulo
    assert perto(numeros(celulas[2])[0], m["agregado"]["mrr"]), rotulo
    assert perto(numeros(celulas[3])[0],
                 m["por_nivel"]["morfologica"]["recall_em_k"]), rotulo


# =========================================================================
# tabela unica
# =========================================================================

MANIFESTO = "modelo/execucoes/ds-05-tabela-unica/manifesto_v2.json"

CONTAGENS = [
    ("Observacoes de entrada", "entrada"),
    ("Observacoes admitidas", "admitidos"),
    ("Pool fluvial", "pool_fluvial"),
    ("Grupos admitidos", "grupos_admitidos"),
    ("Grupos no pool fluvial", "grupos_pool"),
]


@pytest.mark.parametrize("documento", ["docs/EVIDENCIA.md", "docs/METODO.md",
                                       "modelo/DADOS.md"])
@pytest.mark.parametrize("rotulo,chave", CONTAGENS)
def test_contagem_da_tabela_unica_sai_do_manifesto(documento, rotulo, chave):
    m = carregar(MANIFESTO)
    celulas = celulas_da_linha(documento, rotulo)
    assert numeros(celulas[1]) == [float(m[chave])], (documento, rotulo)


@pytest.mark.parametrize("fonte,rotulo", [
    ("nyc", "NYC"), ("sen1floods11", "Sen1Floods11"), ("cems", "CEMS"),
    ("ufo", "Urban Flood Observations"), ("uk", "UK"),
    ("sao_paulo", "Sao Paulo"), ("curitiba", "Curitiba"), ("recife", "Recife"),
])
def test_composicao_por_fonte_da_ficha_de_dados_sai_do_manifesto(fonte, rotulo):
    linha = next(c for c in carregar(MANIFESTO)["contagem_por_fonte"]
                 if c["fonte"] == fonte)
    celulas = celulas_da_linha("modelo/DADOS.md", rotulo)
    # DADOS: | fonte | entrada | admitidos | pool |
    assert numeros(celulas[1]) == [float(linha["entrada"])], rotulo
    assert numeros(celulas[2]) == [float(linha["admitido"])], rotulo
    assert numeros(celulas[3]) == [float(linha["elegivel_pool_fluvial"])], rotulo


# =========================================================================
# camada social
# =========================================================================

def test_contagem_social_da_evidencia_sai_do_manifesto():
    m = carregar("modelo/execucoes/camada-social-territorial/ibge-curitiba/"
                 "manifesto_ibge_curitiba_v1.json")["resumo"]
    texto = (RAIZ / "docs/EVIDENCIA.md").read_text(encoding="utf-8")
    for valor in (m["setores_malha"], m["setores_com_unidade_social"],
                  m["setores_em_fcu"]):
        formatado = f"{valor:,}".replace(",", ".")
        assert formatado in texto, valor


@pytest.mark.parametrize("termo,coluna,tipo", [
    ("hand_m", 1, "Water On Street Complaint"),
    ("hand_m", 2, "Water in Basement Complaint"),
    ("elevation_rel_m", 1, "Water On Street Complaint"),
    ("elevation_rel_m", 2, "Water in Basement Complaint"),
    ("mrms_qpe_24h_mm", 1, "Water On Street Complaint"),
    ("mrms_qpe_24h_mm", 2, "Water in Basement Complaint"),
    ("twi_dinf", 1, "Water On Street Complaint"),
    ("twi_dinf", 2, "Water in Basement Complaint"),
])
def test_corte_por_tipo_de_chamado_sai_do_artefato(termo, coluna, tipo):
    """Agua na via e agua no porao sao fenomenos diferentes, e o pacote publica
    os dois separados. Este teste guarda os numeros dessa separacao."""
    efeito = carregar(CHICAGO)["por_tipo_de_chamado"][tipo][termo]
    celulas = celulas_da_linha("docs/EVIDENCIA.md", termo,
                               secao="| Termo | Agua na via | Agua no porao |")
    assert perto(numeros(celulas[coluna])[0], efeito["coef"]), (termo, tipo)


# =========================================================================
# pesos dos modelos servidos, no cartao do modelo
# =========================================================================

PESOS = [
    ("recife_pluvial", "### `recife_pluvial`",
     ["elevation_rel_m", "slope_deg", "hand_m", "twi_dinf", "rain_decay_index"]),
    ("pluvial_urbano_nyc", "### `pluvial_urbano_nyc`",
     ["elevation_rel_m", "slope_deg", "hand_m", "twi_dinf"]),
    ("fluvial_planicie", "### `fluvial_planicie`",
     ["elevation_rel_m", "slope_deg", "hand_m", "twi_dinf"]),
    ("fluvial_serra", "### `fluvial_serra`", ["hand_m"]),
]


@pytest.mark.parametrize("chave,secao,variaveis", PESOS,
                         ids=[p[0] for p in PESOS])
def test_coeficiente_publicado_sai_do_artefato_do_modelo(chave, secao, variaveis):
    m = carregar(f"modelo/execucoes/svc-01-modelos/{chave}.json")
    for v in variaveis:
        celulas = celulas_da_linha("modelo/MODELO.md", v, secao=secao)
        assert perto(numeros(celulas[1])[0], m["coef"][v]), (chave, v)


@pytest.mark.parametrize("chave,secao,variaveis", PESOS,
                         ids=[p[0] for p in PESOS])
def test_intervalo_publicado_sai_do_artefato_do_modelo(chave, secao, variaveis):
    m = carregar(f"modelo/execucoes/svc-01-modelos/{chave}.json")
    for v in variaveis:
        celulas = celulas_da_linha("modelo/MODELO.md", v, secao=secao)
        baixo, alto = numeros(celulas[2])
        assert perto(baixo, m["ic95_coef"][v][0]), (chave, v, "baixo")
        assert perto(alto, m["ic95_coef"][v][1]), (chave, v, "alto")


@pytest.mark.parametrize("chave,secao,variaveis", PESOS,
                         ids=[p[0] for p in PESOS])
def test_faixa_de_dominio_publicada_sai_do_artefato(chave, secao, variaveis):
    m = carregar(f"modelo/execucoes/svc-01-modelos/{chave}.json")
    rotulo = {"recife_pluvial": "`recife_pluvial`",
              "pluvial_urbano_nyc": "`pluvial_urbano_nyc`",
              "fluvial_planicie": "`fluvial_planicie`",
              "fluvial_serra": "`fluvial_serra`"}[chave]
    celulas = celulas_da_linha("modelo/MODELO.md", rotulo.strip("`"),
                               secao="## Dominio de covariaveis")
    ordem = ["elevation_rel_m", "slope_deg", "hand_m", "twi_dinf",
             "rain_decay_index"]
    for i, v in enumerate(ordem, start=1):
        if v not in m["faixa_dominio_5_95"]:
            continue
        baixo, alto = numeros(celulas[i])
        assert perto(baixo, m["faixa_dominio_5_95"][v][0]), (chave, v)
        assert perto(alto, m["faixa_dominio_5_95"][v][1]), (chave, v)


# =========================================================================
# aderencia temporal, na pagina de validacao
# =========================================================================

@pytest.mark.parametrize("chave", ["recife_pluvial", "pluvial_urbano_nyc",
                                   "pluvial_urbano_sao_paulo",
                                   "fluvial_planicie", "fluvial_serra"])
def test_aderencia_temporal_publicada_sai_do_artefato(chave):
    a = carregar(f"modelo/execucoes/svc-01-modelos/{chave}.json")["aderencia_temporal"]
    celulas = celulas_da_linha("docs/VALIDACAO.md", chave,
                               secao="## 5. Aderencia temporal")
    assert numeros(celulas[1]) == [float(a["datas_distintas"])], chave
    n_ambas, pct = numeros(celulas[2])
    assert n_ambas == float(a["datas_com_ambas_as_classes"]), chave
    assert perto(pct / 100, a["fracao_de_datas_com_ambas_as_classes"]), chave
    assert perto(numeros(celulas[3])[0] / 100,
                 a["fracao_de_pontos_em_data_compartilhada"]), chave


# =========================================================================
# predicao seletiva das frentes
# =========================================================================

SELETIVA = "modelo/execucoes/sintese-neural/predicao_seletiva_das_frentes.json"


@pytest.mark.parametrize("caminho,rotulo", [
    (("A",), "A"), (("B",), "B"), (("C",), "C"),
    (("D", "lexico"), "D lexico"), (("D", "conjuncao"), "D conjuncao"),
], ids=["A", "B", "C", "D_lexico", "D_conjuncao"])
def test_aurc_publicada_sai_do_artefato(caminho, rotulo):
    f = carregar(SELETIVA)["frentes"]
    for c in caminho:
        f = f[c]
    celulas = celulas_da_linha("docs/EVIDENCIA.md", rotulo,
                               secao="## 11. Predicao seletiva")
    assert numeros(celulas[2]) == [float(f["n"])], rotulo
    assert perto(numeros(celulas[3])[0], f["aurc"]), rotulo
    assert perto(numeros(celulas[4])[0], f["aurc_do_acaso"]["p05"]), rotulo
    disse_sim = celulas[5].strip("* ").lower().startswith("sim")
    assert disse_sim is bool(f["ordena_melhor_que_o_acaso"]), rotulo


# =========================================================================
# cadeia seletiva
# =========================================================================

@pytest.mark.parametrize("rotulo,chave", [
    ("Casos", "casos"),
    ("Publicados", "publicados"),
    ("Cobertura", "cobertura"),
    ("Acuracia da decisao", "acuracia_da_decisao"),
])
def test_cadeia_seletiva_publicada_sai_do_artefato(rotulo, chave):
    c = carregar("modelo/execucoes/sintese-neural/cadeia_seletiva.json")
    celulas = celulas_da_linha("docs/EVIDENCIA.md", rotulo,
                               secao="## 12. Cadeia seletiva")
    assert perto(numeros(celulas[1])[0], float(c[chave])), rotulo


def test_abstencoes_da_cadeia_saem_do_artefato():
    c = carregar("modelo/execucoes/sintese-neural/cadeia_seletiva.json")
    celulas = celulas_da_linha("docs/EVIDENCIA.md",
                               "Abstencoes no elo de verificacao",
                               secao="## 12. Cadeia seletiva")
    assert numeros(celulas[1]) == [float(c["abstencoes_por_elo"]["verificacao"])]


# =========================================================================
# regiao por regiao
# =========================================================================

@pytest.mark.parametrize("nome,secao,cenario", [
    ("recife", "## Recife", "mediana_observada"),
    ("curitiba", "## Curitiba", "unico"),
    ("petropolis", "## Petropolis", "unico"),
])
def test_grade_da_pagina_de_regioes_sai_do_resultado(nome, secao, cenario):
    r = regiao_da_grade(nome)
    c = r["cenarios"][cenario]
    linha = celulas_da_linha("docs/REGIOES.md", "Grade", secao=secao)
    valores = numeros(linha[1])
    assert valores[0] == float(r["celulas_da_grade"]), nome
    assert valores[1] == float(r["celulas_validas"]), nome
    assert valores[2] == float(r["celulas_serviveis"]), nome
    assert valores[3] == float(c["celulas_recusadas_por_dominio"]), nome

    escore = celulas_da_linha("docs/REGIOES.md",
                              "Escore (cenario de chuva mediano)"
                              if nome == "recife" else "Escore", secao=secao)
    # "p50" e "p90" sao rotulos, nao valores: saem antes de extrair numero
    p50, p90 = numeros(escore[1].replace("p50", "").replace("p90", ""))
    assert perto(p50, c["escore_mediano"]), nome
    assert perto(p90, c["escore_p90"]), nome


@pytest.mark.parametrize("fonte,secao", [
    ("nyc", "## NYC"),
    ("sao_paulo", "## Sao Paulo"),
    ("curitiba", "## Curitiba"),
])
def test_presenca_na_tabela_unica_por_regiao_sai_do_manifesto(fonte, secao):
    linha = next(c for c in carregar(MANIFESTO)["contagem_por_fonte"]
                 if c["fonte"] == fonte)
    celulas = celulas_da_linha("docs/REGIOES.md", "Presenca na tabela unica",
                               secao=secao)
    admitido, grupos = numeros(celulas[1])
    assert admitido == float(linha["admitido"]), fonte
    assert grupos == float(linha["grupos_admitidos"]), fonte


def test_a_amostra_de_chicago_na_pagina_de_regioes_sai_do_artefato():
    a = carregar(CHICAGO)["amostra"]
    celulas = celulas_da_linha("docs/REGIOES.md", "Amostra", secao="## Chicago")
    pontos, positivos, dias = numeros(celulas[1])
    assert pontos == float(a["pontos"])
    assert positivos == float(a["positivos"])
    assert dias == float(a["dias_informativos"])
