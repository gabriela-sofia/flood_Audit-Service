"""Consolida a prontidao do framework sem treinar, adquirir ou abrir reservas.

Um resultado recente pode contrariar um resumo antigo. Este fechamento le os
artefatos, verifica seus hashes e separa integridade de arquivo, evidencia
experimental e lacuna observacional. Nenhuma dessas propriedades substitui as
outras. O snapshot e imutavel e os modelos servidos ficam intactos.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

ARTEFATOS = {
    "auditoria": "local_runs/auditoria-pre-treino/resultado.json",
    "elevacao_local": "local_runs/auditoria-elevacao-local/resultado.json",
    "posicao_v2": "local_runs/posicao-topografica/v2/manifesto.json",
    "decomposicao": "local_runs/ms3-decomposicao-do-quadro/resultado.json",
    "aprendizado": "local_runs/ms4-bracos-de-desenvolvimento/resultado.json",
    "incerteza": "local_runs/ms5-recusa-decomposta/resultado.json",
    "consulta": "local_runs/ms6-quadro-de-treino-para-consulta/resultado.json",
    "territorio": "local_runs/tc01-territorio-consultavel/desenvolvimento/manifesto.json",
    "preparo_brasil": "local_runs/br-preparo-sem-rotulo/resumo.json",
    "replica_territorio": "local_runs/rfd-replica-no-territorio-v5/validacao.json",
    "politica": "local_runs/politica-sob-orcamento/v2_populacao_da_tarefa/resultado.json",
    "social": "local_runs/camada-social-territorial/experimentos_sociais_v0.json",
}
CODIGOS = (
    "scripts/metodo/fechamento_do_framework.py",
    "scripts/metodo/auditoria_tc01.py",
    "scripts/metodo/relatorio_de_fechamento.py",
    "scripts/metodo/verdade_observacional.py",
    "scripts/metodo/auditoria_pre_treino.py",
    "scripts/metodo/resposta_lugar_novo.py",
    "scripts/servico/svc02_contrato_inferencia.py",
    "scripts/suscetibilidade/posicao_no_ponto.py",
    "scripts/metodo/ms3_decomposicao_do_quadro.py",
    "scripts/metodo/ms4_bracos_de_desenvolvimento.py",
    "scripts/metodo/ms5_recusa_decomposta.py",
    "scripts/metodo/ms6_quadro_de_treino_para_consulta.py",
    "scripts/social/camada_social.py",
    "scripts/social/social01_censo_municipal.py",
    "scripts/geo/geoid01_catalogo_publico.py",
    "scripts/metodo/ms7_territorio_curado.py",
)
FONTES = (
    ("Datasheets for Datasets", "https://arxiv.org/abs/1803.09010",
     "Documentar composicao, coleta, usos e limites; ficha nao comprova qualidade observacional."),
    ("Model Cards for Model Reporting", "https://arxiv.org/abs/1810.03993",
     "Avaliar e declarar usos, contextos e limites por modelo, nao por produto inteiro."),
    ("IPCC AR6: Glossario", "https://www.ipcc.ch/report/ar6/wg2/chapter/annex-ii/",
     "Separar ameaca, exposicao, vulnerabilidade e risco; nao somar social ao escore fisico."),
    ("Roberts et al., 2017", "https://doi.org/10.1111/ecog.02881",
     "Definir avaliacao de acordo com dependencia espacial, temporal e hierarquica."),
)


def ler_estavel(caminho: Path) -> tuple[dict | None, dict]:
    """Uma gravacao concorrente nao pode produzir um snapshot aparentemente atual."""
    if not caminho.is_file():
        return None, {"estado": "ausente", "sha256": None}
    antes = sha256_arquivo(caminho)
    try:
        dados = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError) as erro:
        return None, {"estado": "ilegivel", "sha256": antes, "motivo": str(erro)}
    depois = sha256_arquivo(caminho)
    if antes != depois:
        raise RuntimeError(f"arquivo mudou durante leitura: {caminho}")
    if not isinstance(dados, dict):
        return None, {"estado": "ilegivel", "sha256": depois, "motivo": "esperado objeto JSON"}
    return dados, {"estado": "lido", "sha256": depois}


def verificar_resultado(dados: dict, pre: dict | None) -> dict:
    """Confere a convencao dos MS; nao infere independencia da existencia de hash."""
    problemas = []
    if "hash_resumo" in dados:
        corpo = {k: v for k, v in dados.items() if k not in ("hash_resumo", "quando_utc")}
        if dados["hash_resumo"] != sha256_json(corpo):
            problemas.append("hash do resultado divergente")
    else:
        problemas.append("resultado sem hash de conteudo verificavel")
    if "pre_registro" in dados:
        if not pre or "criterios" not in pre:
            problemas.append("pre-registro ausente ou sem criterios")
        elif pre.get("hash") != sha256_json(pre["criterios"]):
            problemas.append("hash do pre-registro divergente")
        elif dados["pre_registro"] != pre["hash"]:
            problemas.append("resultado cita outro pre-registro")
    return {"estado": "verificado" if not problemas else "nao_verificado", "problemas": problemas,
            "nao_e": "integridade nao demonstra cronologia, independencia ou validade do rotulo"}


def verificar_posicao(dados: dict, repo: Path) -> dict:
    """Verifica a linhagem da derivacao, nao apenas a existencia do manifesto."""
    problemas = []
    versao = str(dados.get("dataset", ""))
    if not versao.startswith("v") or not versao[1:].isdigit():
        return {"estado": "nao_verificado", "problemas": ["versao ds06 invalida"]}
    alvos = {
        "pontos_ds06_sha256": repo / "local_runs/ds-06-dataset-de-treino" / versao / "pontos.parquet",
        "codigo_sha256": repo / "scripts/suscetibilidade/posicao_no_ponto.py",
        "saida_sha256": repo / "local_runs/posicao-topografica/v2/pontos.parquet",
    }
    for chave, alvo in alvos.items():
        observado = sha256_arquivo(alvo)
        if observado is None or dados.get(chave) != observado:
            problemas.append(f"{chave}: arquivo ausente ou diferente da derivacao")
    if any(n != 0 for n in (dados.get("pos_min_negativo") or {}).values()):
        problemas.append("posicao acima do minimo negativa")
    return {"estado": "verificado" if not problemas else "nao_verificado", "problemas": problemas,
            "nao_e": "nao valida DEM historico, grade metrica ou observacao de inundacao"}


def conclusoes(artefatos: dict) -> list[dict]:
    """Nao perde resultados negativos nem amplia o escopo de um ganho."""
    out = []
    for nome in ("decomposicao", "aprendizado", "consulta", "territorio_curado"):
        dados = artefatos.get(nome) or {}
        for chave, c in (dados.get("comparacoes") or {}).items():
            out.append({"frente": nome, "comparacao": chave, "regime": dados.get("regime", "nao_declarado"),
                        "veredito": c.get("veredito", "untested"), "a": c.get("a"), "b": c.get("b"),
                        "eventos": c.get("eventos_pareados"), "delta_mediano": c.get("dentro_do_evento_mediana"),
                        "ic95_media": c.get("ic95_media_dentro"),
                        "grupos_espaciais": c.get("grupos_espaciais_conectados"),
                        "limite": "criterio do experimento; nao e promocao nem validade em outro quadro"})
    for sinal, c in ((artefatos.get("incerteza") or {}).get("sinais") or {}).items():
        out.append({"frente": "incerteza", "comparacao": sinal, "regime": "exploratorio",
                    "veredito": c.get("estado", "untested"), "eventos": c.get("eventos"),
                    "delta_mediano": c.get("delta_mediano"), "ic95_media": None,
                    "limite": "ordenacao de erro na amostra; nao confianca calibrada nem portao servido"})
    return out


def pendencias(artefatos: dict) -> list[dict]:
    """Concluir uma tarefa nao exige que todas as hipoteses tenham resultado positivo."""
    consulta = bool(artefatos.get("consulta"))
    estatica = (artefatos.get("elevacao_local") or {}).get("estatica") or {}
    tc01 = (artefatos.get("auditoria_territorio") or {}).get("auditoria") or {}
    curado = artefatos.get("curadoria_territorio") or {}
    campos = [
        ("F1", "Alvo e populacao", "decisao_cientifica",
         "Consulta de territorio e vizinhanca da agua sao perguntas diferentes; nao escolher quadro pela AUC.",
         "Declarar o alvo e a populacao de cada modelo e pergunta principal do artigo.",
         "Contrato versionado e consumidores coerentes, com exclusoes e observabilidade declaradas."),
        ("F2", "Treino e inferencia", "parcialmente_resolvido",
         f"A16: le classe diretamente = {estatica.get('le_a_classe', 'nao verificado')}; depende da amostra.",
         "Manter piso amostral so como linhagem historica; avaliar variante calculavel sem rotulo.",
         "Mesma funcao e suporte no lote/consulta, procedencia e teste independente da variante."),
        ("F3", "Territorio consultavel", "exploratorio" if consulta else "pendente_execucao",
         "MS6 registrado em desenvolvimento." if consulta else "MS6 tem codigo, mas nenhum resultado disponivel neste snapshot.",
         "Concluir e auditar MS6; nao chamar tile escolhido entre eventos de amostra territorial representativa.",
         "Modelo avaliado no quadro correspondente a consulta, com selecao espacial explicitada."),
        ("F4", "Confirmacao da variante corrigida", "bloqueado_por_observacao",
         "Testes antigos nao confirmam nova populacao, novo vetor ou nova pergunta.",
         "Preparar catalogo, mecanismo, espaco e observabilidade; congelar desenho antes de ler reserva.",
         "Avaliacao independente da variante completa, com suficiente precisao e conclusao delimitada."),
        ("F5", "Inferencia contextual e explicacao", "parcialmente_resolvido",
         "Contrato de uso contextual distingue demonstracao historica e consulta nova.",
         "Exercitar casos reais de elegibilidade, recusa e explicacao fiel sem promover modelos legados.",
         "Casos completos rastreaveis; contribuicoes reproduzem o estimador, sem afirmacao causal."),
        ("F6", "Dimensao socioambiental", "pendente_validacao_empirica",
         "Camada social v0 demonstra separacao de eixos; nao valida vulnerabilidade territorial.",
         "Integrar indicadores oficiais com data, escala e cobertura; avaliar exposicao e lacunas de evidencia.",
         "Estudo territorial reproduzivel sem somar vulnerabilidade ao escore fisico."),
        ("F7", "Aprendizado autonomo", "indecidivel",
         "Politica v2 foi medida; vantagem consistente ainda nao demonstrada.",
         "Comparar ciclos sob mesmo orcamento e base, medir suporte, desempenho e regressao em desenvolvimento.",
         "Ciclo auditavel avaliado; reivindicar eficacia so se sustentada, preservar resultado negativo."),
        ("F8", "Observacao fora do nucleo", "bloqueado_por_observacao",
         "Serra/pluvial brasileiro e forcamento no instante requerem observacoes elegiveis.",
         "Adquirir observacao que destrave tarefa definida, nao ampliar volume indiscriminadamente.",
         "Instrumento resolve mecanismo, tempo, positivos e negativos da tarefa; ou extensao explicitamente limitada."),
    ]
    out = [dict(zip(("id", "frente", "estado", "evidencia", "acao", "criterio_de_conclusao"), c)) for c in campos]
    if tc01:
        f3 = out[2]
        f3["evidencia"] += (f" TC01: {tc01.get('classes_divergentes')} classes divergentes; "
                            f"{tc01.get('coordenadas_repetidas_excedentes')} repeticoes de coordenada/evento.")
        if curado:
            f3["evidencia"] += f" Derivacao separada removeu {curado.get('repeticoes_removidas')}; consumidores antigos preservados."
            f3["acao"] = "Avaliar a derivacao curada em experimento novo; MS6 original nao passa a representar dados curados."
        if tc01.get("problemas"):
            f3["estado"] = "bloqueado_por_divergencia"
    replica = artefatos.get("replica_territorio") or {}
    ms7 = artefatos.get("territorio_curado") or {}
    if ms7:
        grupos = sorted({c.get("grupos_espaciais_conectados", 0) for c in ms7.get("comparacoes", {}).values()})
        out[2].update(estado="exploratorio_executado",
                      evidencia=out[2]["evidencia"] + f" MS7 executado com curadoria e exclusao de tiles; grupos conectados: {grupos}.",
                      acao="Preservar MS7 como desenvolvimento; aumentar suporte independente por instrumento elegivel, nao por reparticao favoravel.")
    social = artefatos.get("social_oficial") or {}
    if social:
        out[5]["evidencia"] += f" Censo 2022: {len(social.get('municipios', []))} municipios adquiridos e {social.get('linhas')} categorias; contexto de saneamento, nao indice validado."
    catalogo = artefatos.get("catalogo_geoid") or {}
    if catalogo:
        novidades = sorted({e for c in catalogo.get("catalogos", {}).values() for e in c.get("brasil_sem_ativacao_local", [])})
        out[3]["evidencia"] += f" GEOID catalogado sem rotulos: {len(novidades)} ativacoes brasileiras ausentes do inventario local; identidade ainda exige confronto entre fontes."
    if replica.get("passou") is False:
        out[3]["evidencia"] += (f" RFD05 reprovou: recall mediano {replica.get('recall_mediano')} "
                                f"em {replica.get('eventos_avaliaveis')} eventos avaliaveis.")
        out[3]["acao"] = ("Nao admitir replica territorial como verdade equivalente ao GFD nem otimizar limiar no mesmo teste. "
                          "Definir instrumento observacional elegivel e validar a cadeia antes de abrir eventos novos.")
    return out


def resumo_qa(arq: Path | None) -> dict:
    if arq is None:
        return {"estado": "nao_executado_neste_fechamento"}
    raiz = ElementTree.parse(arq).getroot()
    casos = list(raiz.iter("testcase"))
    falhas = sum(c.find("failure") is not None or c.find("error") is not None for c in casos)
    pulados = sum(c.find("skipped") is not None for c in casos)
    motivos = Counter(c.find("skipped").get("message", "sem motivo informado")
                      for c in casos if c.find("skipped") is not None)
    return {"estado": "passou" if casos and not falhas else "falhou_ou_vazio", "total": len(casos),
            "falhas": falhas, "pulados": pulados, "passaram": len(casos) - falhas - pulados,
            "motivos_pulados": dict(motivos),
            "sha256": sha256_arquivo(arq), "arquivo": arq.name,
            "limite": "resultado de software fornecido; nao demonstra validade cientifica nem cobre alteracoes posteriores"}


def construir(repo: Path = REPO, qa: Path | None = None, *, auditoria_territorio: Path | None = None,
              curadoria_territorio: Path | None = None, experimento_territorio: Path | None = None,
              social_oficial: Path | None = None, catalogo_geoid: Path | None = None) -> dict:
    repo = Path(repo)
    dados, registro = {}, {}
    for nome, relativo in ARTEFATOS.items():
        arq = repo / relativo
        d, r = ler_estavel(arq)
        r["caminho"] = relativo
        if d is not None:
            dados[nome] = d
            r["quando_utc"] = d.get("quando_utc")
            if nome in ("decomposicao", "aprendizado", "incerteza", "consulta", "politica"):
                pre, pr = ler_estavel(arq.parent / "pre_registro.json")
                pr["caminho"] = str((arq.parent / "pre_registro.json").relative_to(repo))
                r["pre_registro"] = pr
                r["integridade"] = verificar_resultado(d, pre)
            if nome == "posicao_v2":
                r["integridade"] = verificar_posicao(d, repo)
        registro[nome] = r
    for nome, arq in (("territorio_curado", experimento_territorio), ("social_oficial", social_oficial),
                      ("catalogo_geoid", catalogo_geoid)):
        if arq is None:
            continue
        arq = Path(arq).resolve()
        d, r = ler_estavel(arq)
        r["caminho"] = str(arq.relative_to(repo.resolve()))
        if d is not None:
            pre = None
            if nome == "territorio_curado":
                pre, pr = ler_estavel(arq.parent / "pre_registro.json")
                pr["caminho"] = str((arq.parent / "pre_registro.json").relative_to(repo.resolve()))
                r["pre_registro"] = pr
            r["integridade"] = verificar_resultado(d, pre)
            problemas = r["integridade"]["problemas"]
            if nome == "territorio_curado":
                origens = d.get("origens", {})
                if not curadoria_territorio or origens.get("curadoria") != sha256_arquivo(Path(curadoria_territorio).resolve()):
                    problemas.append("experimento sem vinculo a curadoria fornecida")
                if origens != ((pre or {}).get("criterios") or {}).get("origens"):
                    problemas.append("resultado nao usa as origens pre-registradas")
                if curadoria_territorio and origens.get("pontos_curados") != sha256_arquivo(Path(curadoria_territorio).resolve().parent / "pontos.parquet"):
                    problemas.append("pontos curados do experimento divergentes")
                for chave, caminho in (("codigo", "scripts/metodo/ms7_territorio_curado.py"),
                                        ("estimador", "scripts/metodo/generalizacao_por_reservas.py"),
                                        ("posicao", "local_runs/posicao-topografica/v2/pontos.parquet")):
                    if origens.get(chave) != sha256_arquivo(repo / caminho):
                        problemas.append(f"origem MS7 divergente: {chave}")
                versao = (dados.get("posicao_v2") or {}).get("dataset", "")
                if not re_versao(versao) or origens.get("ds06") != sha256_arquivo(repo / "local_runs/ds-06-dataset-de-treino" / versao / "pontos.parquet"):
                    problemas.append("DS06 do experimento divergente")
                for dobra in d.get("suporte_por_dobra", []):
                    if any(o.get("tiles_compartilhados", 1) or o.get("eventos_compartilhados", 1) for o in dobra.get("por_origem", {}).values()):
                        problemas.append("suporte declara sobreposicao treino/teste")
            elif nome == "social_oficial":
                for arquivo, chave in (("indicadores.csv", "indicadores_sha256"), ("origem.json", "origem_arquivo_sha256")):
                    if d.get(chave) != sha256_arquivo(arq.parent / arquivo):
                        problemas.append(f"social divergente: {arquivo}")
            elif nome == "catalogo_geoid":
                raiz = arq.parents[2]
                for arquivo, origem in d.get("downloads", {}).items():
                    alvo = (raiz / arquivo).resolve()
                    if raiz.resolve() not in alvo.parents or origem.get("sha256") != sha256_arquivo(alvo):
                        problemas.append(f"catalogo divergente: {arquivo}")
            r["integridade"]["estado"] = "nao_verificado" if problemas else "verificado"
            dados[nome] = d
        registro[nome] = r
    for nome, arq in (("auditoria_territorio", auditoria_territorio), ("curadoria_territorio", curadoria_territorio)):
        if arq is None:
            continue
        arq = Path(arq).resolve()
        relativo = arq.relative_to(repo.resolve())
        d, r = ler_estavel(arq)
        r["caminho"] = str(relativo)
        if d is not None:
            r["integridade"] = verificar_resultado(d, None)
            fonte = repo / "local_runs/tc01-territorio-consultavel/desenvolvimento/pontos.parquet"
            declarado = (d.get("origens") or {}).get("pontos") if nome == "auditoria_territorio" else d.get("origem_sha256")
            if declarado != sha256_arquivo(fonte):
                r["integridade"]["estado"] = "nao_verificado"
                r["integridade"]["problemas"].append("TC01 de origem mudou")
            if nome == "curadoria_territorio":
                for arquivo, chave in (("pontos.parquet", "pontos_sha256"), ("aliases.parquet", "aliases_sha256")):
                    if d.get(chave) != sha256_arquivo(arq.parent / arquivo):
                        r["integridade"]["estado"] = "nao_verificado"
                        r["integridade"]["problemas"].append(f"curadoria divergente: {arquivo}")
                if d.get("auditoria_sha256") != (dados.get("auditoria_territorio") or {}).get("hash_resumo"):
                    r["integridade"]["estado"] = "nao_verificado"
                    r["integridade"]["problemas"].append("curadoria nao vinculada a auditoria fornecida")
            dados[nome] = d
        registro[nome] = r
    if "territorio_curado" in registro and registro.get("curadoria_territorio", {}).get("integridade", {}).get("estado") != "verificado":
        integ = registro["territorio_curado"].get("integridade")
        if integ:
            integ["estado"] = "nao_verificado"
            integ["problemas"].append("curadoria vinculada nao verificada neste snapshot")
    confiaveis = {n: d for n, d in dados.items()
                 if registro[n].get("integridade", {}).get("estado") != "nao_verificado"}
    corpo = {"tipo": "fechamento_do_framework", "versao": "v1",
             "objetivo": "framework de geoinformatica socioambiental para inferencia contextual de suscetibilidade a inundacao e aprendizado autonomo auditavel",
             "estado": "em_consolidacao", "ciencia_fechada": False, "promovido": False,
             "rotulos_novos_lidos": False, "avaliacoes_rerodadas": False,
             "codigo": {c: sha256_arquivo(repo / c) for c in CODIGOS},
             "artefatos": registro, "conclusoes": conclusoes(confiaveis), "pendencias": pendencias(confiaveis),
             "qa": resumo_qa(qa), "fontes_metodologicas": FONTES,
             "nao_e": "nao e promocao, reproducao dos experimentos ou garantia de independencia das observacoes"}
    # Falha se os produtores escreveram enquanto a sintese era calculada.
    for r in registro.values():
        if r.get("sha256") != sha256_arquivo(repo / r["caminho"]):
            raise RuntimeError(f"arquivo mudou durante fechamento: {r['caminho']}")
        pr = r.get("pre_registro")
        if pr and pr.get("sha256") != sha256_arquivo(repo / pr["caminho"]):
            raise RuntimeError(f"pre-registro mudou durante fechamento: {pr['caminho']}")
    if any(h != sha256_arquivo(repo / c) for c, h in corpo["codigo"].items()):
        raise RuntimeError("codigo mudou durante fechamento")
    corpo["hash_resumo"] = sha256_json(corpo)
    corpo["quando_utc"] = agora_utc()
    return corpo


def re_versao(versao: str) -> bool:
    return isinstance(versao, str) and versao.startswith("v") and versao[1:].isdigit()


def markdown(r: dict) -> str:
    linhas = ["# Fechamento do framework REV-P", "",
              "Gerado por `scripts/metodo/fechamento_do_framework.py`. Nao editar a mao.", "",
              r["objetivo"], "", f"Snapshot: {r['quando_utc']}; `{r['hash_resumo']}`.", "",
              "**Estado: em consolidacao.** Ciencia fechada: nao. Nenhum modelo promovido ou teste final rerodado.",
              "Integridade de arquivo nao e validade observacional. Este documento nao substitui a auditoria pre-treino.",
              "", "## Pergunta cientifica e fronteiras", "",
              "O nucleo atual aprende associacoes entre terreno e agua nao permanente observada em eventos de planicie. "
              "Nao aprende a probabilidade de chover ou de ocorrer um evento em uma data. HAND, TWI, declive e posicao "
              "topografica descrevem condicionantes; nao substituem chuva, vazao, drenagem urbana ou observacao no pico.", "",
              "Vizinhanca da inundacao, AOI do analista e territorio delimitado previamente sao populacoes distintas. "
              "O TC01 amplia a extensao de fundo, mas seleciona tiles tocados por eventos conhecidos; nao representa "
              "todo o Brasil nem recupera reservas independentes. Uma visada seca nao prova ausencia durante todo o pico.", "",
              "O MDT preenchido e a mesma base necessaria ao lote e a consulta. Um MDT contemporaneo nao prova "
              "a configuracao historica de todos os eventos; mudancas de ocupacao, canal e obras continuam limites.", "",
              "## Como cada aprendizado contribui", "",
              "1. **Pre-treino mascarado:** atualiza pesos para reconstruir dados ocultados. Reconstrucao melhor "
              "nao demonstra segmentacao melhor nem compreensao causal da cheia. A H1 SAR indecidivel nao e equivalencia.",
              "2. **Ajuste supervisionado:** aprende a separar classes observacionais na populacao declarada. "
              "MS4 testa representacoes sobre a mesma base de desenvolvimento; a melhora incremental K5 nao "
              "confirma a transferencia do pre-treino SAR nem autoriza territorio novo.",
              "3. **Ciclo autonomo auditavel:** seleciona lacunas, registra admissao, avalia candidatos e protege "
              "regressao e reservas. Atualizar catalogo, memoria ou RAG nao e atualizar pesos; eficacia da politica "
              "continua dependente da comparacao sob orcamento igual.", "",
              "Uma explicacao fiel mostra como os descritores contribuiram para o estimador, nao prova por si "
              "a causa do evento. Sigmoide, AUC e bootstrap nao transformam um escore condicionado a amostra "
              "em probabilidade territorial calibrada. A camada social permanece separada do escore fisico.", "",
              "## Resultados atuais lidos dos artefatos", "",
              "Todos os MS abaixo sao desenvolvimento em eventos conhecidos; nao confirmam uma consulta nova.", "",
              "| frente / metrica | comparacao (A menos B) | estado | eventos | delta mediano | IC da media |",
              "|---|---|---|---:|---:|---|"]
    for c in r["conclusoes"]:
        metrica = "AURC" if c["frente"] == "incerteza" else "AUC"
        bracos = f"{c.get('a')} menos {c.get('b')}" if c.get("a") else "sinal menos acaso"
        linhas.append(f"| {c['frente']} / {metrica} | {c['comparacao']}: {bracos} | {c['veredito']} | {c.get('eventos')} | "
                      f"{c.get('delta_mediano')} | {c.get('ic95_media')} |")
    linhas += ["", "Os deltas de AUC e AURC nao tem a mesma semantica. IC da media nao e IC da mediana.",
               "MS4 K5 compara adicionar embedding a vizinhanca, nao substituir a vizinhanca pelo encoder.",
              "MS5 ordena erro na amostra; nao calibra probabilidade territorial. Os varios contrastes "
               "exploratorios nao constituem uma confirmacao independente nem uma familia confirmatoria corrigida.", "",
               "MS7 elimina coordenadas repetidas por evento e exclui tiles do teste do ajuste. As comparacoes "
               "continuam `not_testable` quando faltam oito componentes espaciais conectados. O IC por evento "
               "da tabela e descritivo nesse caso; nao supera o bloqueio espacial. Ver detalhes em "
               "[MS7 e fichas dos modelos](MS7_TERRITORIO_CURADO.md).", "",
               "## O que falta para cada contribuicao", "",
               "| frente | estado | evidencia atual | acao | criterio de conclusao |", "|---|---|---|---|---|"]
    for p in r["pendencias"]:
        linhas.append(f"| {p['id']} {p['frente']} | {p['estado']} | {p['evidencia']} | {p['acao']} | {p['criterio_de_conclusao']} |")
    linhas += ["", "## Procedencia e integridade", "",
               "| artefato | leitura | integridade | hash |", "|---|---|---|---|"]
    for nome, a in r["artefatos"].items():
        integ = a.get("integridade", {})
        linhas.append(f"| {nome}: `{a['caminho']}` | {a['estado']} | {integ.get('estado', 'nao auditada aqui')} "
                      f"{'; '.join(integ.get('problemas', []))} | `{a.get('sha256')}` |")
    linhas += ["", "## Verificacao de software", "", f"`{json.dumps(r['qa'], ensure_ascii=False)}`", "",
               "A QA nao fecha nenhuma lacuna observacional. Hashes e pendencias completos no snapshot JSON.", "",
               "## Base metodologica", ""]
    linhas += [f"- [{titulo}]({url}): {uso}" for titulo, url, uso in FONTES]
    comando = "python scripts/metodo/fechamento_do_framework.py --saida local_runs/fechamento-framework --documento docs/FECHAMENTO_DO_FRAMEWORK.md"
    for nome, opcao in (("auditoria_territorio", "--auditoria-territorio"), ("curadoria_territorio", "--curadoria-territorio"),
                        ("territorio_curado", "--experimento-territorio"), ("social_oficial", "--social-oficial"),
                        ("catalogo_geoid", "--catalogo-geoid")):
        if nome in r["artefatos"]:
            caminho = r["artefatos"][nome]["caminho"].replace("\\", "/")
            comando += f' {opcao} "{caminho}"'
    linhas += ["", "## Reproduzir", "", "```powershell", comando, "```", "",
               "Adicionar `--qa` com o XML da verificacao desta revisao; omitir nao reproduz o registro de QA. "
               "O comando so le artefatos conhecidos. Nao adquire dados nem consome reserva."]
    return "\n".join(linhas) + "\n"


def gravar(r: dict, saida: Path, documento: Path | None = None) -> Path:
    r = json.loads(json.dumps(r, ensure_ascii=False))
    destino = Path(saida) / r["hash_resumo"].split(":")[1]
    destino.mkdir(parents=True, exist_ok=True)
    arq = destino / "manifesto.json"
    if arq.exists():
        antigo = json.loads(arq.read_text(encoding="utf-8"))
        if {k: v for k, v in antigo.items() if k != "quando_utc"} != {k: v for k, v in r.items() if k != "quando_utc"}:
            raise RuntimeError("snapshot existente divergente; nao sobrescrever")
        r = antigo
    else:
        arq.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    texto = markdown(r)
    (destino / "relatorio.md").write_text(texto, encoding="utf-8")
    if documento:
        Path(documento).write_text(texto, encoding="utf-8")
    return arq


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, required=True)
    ap.add_argument("--documento", type=Path)
    ap.add_argument("--qa", type=Path)
    ap.add_argument("--auditoria-territorio", type=Path)
    ap.add_argument("--curadoria-territorio", type=Path)
    ap.add_argument("--experimento-territorio", type=Path)
    ap.add_argument("--social-oficial", type=Path)
    ap.add_argument("--catalogo-geoid", type=Path)
    args = ap.parse_args()
    r = construir(qa=args.qa, auditoria_territorio=args.auditoria_territorio,
                  curadoria_territorio=args.curadoria_territorio, experimento_territorio=args.experimento_territorio,
                  social_oficial=args.social_oficial, catalogo_geoid=args.catalogo_geoid)
    print(f"GRAVADO={gravar(r, args.saida, args.documento)}")
    print("ciencia_fechada=False; promovido=False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
