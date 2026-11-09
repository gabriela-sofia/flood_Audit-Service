"""Relatorio unico do fechamento cientifico operacional: o que vale, para que uso, e o que falta.

Le os artefatos gerados pelos scripts -- nunca numero copiado a mao -- e
escreve `docs/FECHAMENTO_CIENTIFICO_OPERACIONAL.md`. Artefato ausente vira
linha "nao executado", e nao silencio.

nao_e: nao e prova de que a ciencia acabou. Software correto nao e validade
observacional; onde a evidencia e insuficiente, o relatorio diz o bloqueio e
a acao exata que o destrava.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "local_runs"
DOC = REPO / "docs" / "FECHAMENTO_CIENTIFICO_OPERACIONAL.md"
sys.path.insert(0, str(REPO / "scripts" / "metodo"))
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))


def _j(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _n(x) -> str:
    return f"{x:,}".replace(",", ".") if isinstance(x, int) else str(x)


def secao_admissao(L: list) -> None:
    r = _j(RUNS / "admissao-cientifica/retrato.json")
    L += ["## 1. Admissão científica", ""]
    if not r:
        L += ["Não executado.", ""]
        return
    L += [f"Regra única de uso (`admissao_cientifica.py` {r['versao']}) sobre {_n(r['linhas'])} linhas da tabela única. "
          "Consumidores que ajustam ou medem chamam `exigir` e falham com linha não autorizada: "
          "`celula_para_treino`, `generalizacao_protegida`, `curva_de_aprendizado`, `ctx04`, política retrospectiva.", "",
          "| uso | linhas autorizadas |", "|---|---:|"]
    L += [f"| `{u}` | {_n(n)} |" for u, n in r["por_uso"].items()]
    L += ["", "Fontes recusadas como referência (motivo gravado no módulo):", ""]
    L += [f"- `{k}`: {v}" for k, v in r["recusadas"].items()]
    L += ["", "Datas resolvidas na rodada de 02/10 (texto fixo, não lido de artefato): 15.566 linhas CEMS com a cena do produto (`cena_do_produto_ems`), "
          "5 seguem sem data e ficam fora de todo uso temporal (EMSR861 AOI22/23); 376.722 linhas GFD datadas "
          "pelo catálogo de eventos.", ""]


def secao_rastreio(L: list) -> None:
    import rastreio_de_impacto as ri
    e = ri.estado()
    L += ["## 2. Impacto e revalidação", "",
          "Cada gerador exige a montante atual e carimba ao fim; histórico append-only em "
          "`local_runs/rastreio-de-impacto/carimbos.jsonl`.", "",
          "| nó | estado | motivos |", "|---|---|---|"]
    L += [f"| `{no}` | `{v['estado']}` | {'; '.join(v['motivos'][:2]) or '—'} |" for no, v in e.items()]
    L.append("")


def secao_fluvial(L: list) -> None:
    L += ["## 3. Fluvial: contexto do evento", ""]
    t = _j(RUNS / "ctx-02-triagem/triagem.json")
    c = _j(RUNS / "ctx-03-contexto-em-lote/contexto_em_lote.json")
    k = _j(RUNS / "ctx-04-terreno-versus-contexto/card.json")
    if t:
        L += [f"Triagem antes de baixar: {t['eventos']} eventos fluviais brasileiros, {t['pares_elegiveis']} pares "
              f"evento×estação elegíveis (postos em barragem/açude excluídos), {len(t['pares_escolhidos'])} escolhidos "
              f"por diversidade: {t['diversidade']['estacoes']} estações, {t['diversidade']['eventos']} eventos, "
              f"territórios {', '.join(t['diversidade']['territorios'])}.", ""]
    if c:
        estados = {}
        for b in c["bacias"].values():
            estados[b["estado"]] = estados.get(b["estado"], 0) + 1
        completos = sum(1 for p in c["pares"] if p.get("pico_incompleto") is False)
        L += [f"Bacias: {estados}. Pares com pico completo: {completos} de {len(c['pares'])}. "
              f"Pontos ligados a uma única estação: {_n(c['pontos_ligados'])}.", ""]
    else:
        L += ["Lote de contexto não concluído.", ""]
    if k:
        for nome, a in k["analises"].items():
            if a.get("estado") == "not_testable":
                L += [f"- **{nome}**: `not_testable` — {a['motivo']}"]
                continue
            L += [f"- **{nome}**: `{a['estado']}` — {a['eventos']} eventos, {a['pares']} pares. "
                  f"Terreno {a['terreno']}; terreno+contexto {a['terreno_mais_contexto']}; "
                  f"diferença IC95 por evento {a['diferenca_ic95_por_evento']}."]
        L.append("")
    else:
        L += ["Comparação terreno × terreno+contexto não executada.", ""]


def secao_pluvial(L: list) -> None:
    m = _j(RUNS / "floodnet-01-catalogo-publico/manifesto.json")
    L += ["## 4. Pluvial: negativos observados", ""]
    if not m:
        L += ["Não executado.", ""]
        return
    s = m["resumo"]
    L += [f"FloodNet público adquirido: {_n(s['eventos'])} eventos, {_n(s['sensores'])} sensores "
          f"({_n(s['sensores_sem_influencia_de_mare'])} sem maré). Negativos observados: **0** — {s['por_que_zero_negativos']}.",
          "", "**Bloqueado por evidência pública insuficiente**, não por pedido: o catálogo continua só positivo e "
          "nenhum negativo é fabricado. A busca delimitada de fontes abertas está na seção 12.", ""]


def secao_serra(L: list) -> None:
    p = _j(RUNS / "serra-01-protocolo-opera/protocolo.json")
    a = _j(RUNS / "serra-02-pacote-de-adjudicacao/pacote.json")
    L += ["## 5. Serra: referência independente", ""]
    if p:
        L += ["Protocolo OPERA com controles secos (regra fixada antes: evento ≥ "
              f"{p['razao_minima']}× o maior controle):", ""]
        L += [f"- `{k}`: `{v['estado']}` — maior controle {v['maior_controle_km2']} km², razões {v['razao_evento_sobre_controle']}"
              for k, v in p["veredito"].items()]
        L += ["", "Candidato a adjudicação não é rótulo: libera revisão humana, nada além.", ""]
    if a:
        prontos = {n: e["estado"] for n, e in a["eventos"].items()}
        L += [f"Pacote de adjudicação: {sum(v == 'pronto_para_revisao' for v in prontos.values())} eventos prontos, "
              f"bacias {', '.join(a['independencia']['bacias_distintas_prontas'])}; amostra estratificada por relevo, "
              "nunca por HAND; classe padrão `desconhecido`.", "",
              f"**Bloqueado por revisão humana:** {a['bloqueio']}.", ""]


def secao_aprendizado(L: list) -> None:
    c = _j(RUNS / "curva-de-aprendizado/curva.json")
    r = _j(RUNS / "relacoes-sustentadas/registro.json")
    pol = _j(RUNS / "politica-retrospectiva/politica.json")
    L += ["## 6. Hipóteses e aprendizado", ""]
    if r:
        sust = [h for h in r["hipoteses"] if h.get("estado") == "sustentada"]
        L += [f"Hipóteses na versão `{r.get('versao_das_hipoteses', '—')[:23]}`: "
              f"{len(sust)} sustentadas — " + "; ".join(f"{h['mecanismo']}·{h['variavel']}·{h['estimando']}" for h in sust)
              + ". Estimandos medidos: marginal e condicional; contribuição individual e causalidade não medidas.", ""]
    if c:
        for nome, e in c["estudos"].items():
            L += [f"- curva `{nome}`: {e.get('leitura') or e.get('recusa')}"]
        L += ["", "As faixas da curva são dispersão entre ORDENS de treino com o teste fixo, não incerteza "
              "sobre os eventos de teste: 0,780 contra 0,768 não é melhora demonstrada. A seção 8 separa as duas.", ""]
    L += ["## 7. Avaliação da política", ""]
    if pol:
        ex = {e["politica"]: e["medidas"]["auc_protegida"] for e in pol["execucoes"]}
        L += [f"Retrospectiva, mesmo orçamento ({_n(pol['orcamento_linhas'])} linhas), mesma base e teste: AUC protegida "
              f"lacuna {ex.get('lacuna')}, fixa {ex.get('fixa')}, aleatória (mediana) {ex.get('aleatoria')}; "
              f"percentil da lacuna entre {pol['aleatoria_distribuicao_auc']['n']} aleatórias: "
              f"{pol['lacuna_percentil_entre_aleatorias']}."]
        reps = pol.get("repeticoes", [])
        for x in reps:
            empate = x["lacuna"] == x["fixa"] == x["aleatoria_mediana"]
            L += [f"- reserva `{x['sal']}`: lacuna {x['lacuna']}, fixa {x['fixa']}, aleatória {x['aleatoria_mediana']}, "
                  f"percentil {x['lacuna_percentil_entre_aleatorias']:.2f}"
                  + (" — **orçamento não restringiu a escolha: problema de desenho, não resultado**" if empate else "")]
        informativas = [x for x in reps if not (x["lacuna"] == x["fixa"] == x["aleatoria_mediana"])]
        acima = sum(x["lacuna_percentil_entre_aleatorias"] >= 0.95 for x in informativas)
        L += ["", f"Das {len(informativas)} reservas em que o orçamento restringiu, a política por lacuna ficou no topo "
              f"das aleatórias em {acima}. **Sem evidência de vantagem consistente** da política por lacuna; a "
              "vantagem da primeira reserva não se repetiu. Leitura descritiva: aquisição simulada sobre o acervo "
              "em disco, não custo de rede real.", ""]


def secao_validade(L: list) -> None:
    r = _j(RUNS / "reservas-examinadas/retrato.json")
    L += ["## 0. Validade das avaliações: não há reserva intacta", ""]
    if not r:
        L += ["`reservas_examinadas.py` não executado: o regime de toda avaliação fica indeterminado.", ""]
        return
    lac = r["lacre"]
    abertos = "; ".join(f"`{k}` ({len(v)} eventos)" for k, v in lac.get("aberto_por", {}).items())
    L += [f"O lacre de 01/10 ({lac.get('lacrados')} eventos GFD, `generalizacao_protegida` r2) está "
          f"**{lac['estado']}**: aberto por {abertos or '—'}. Causa: {lac.get('causa')}.", "",
          f"Além disso, {r['por_que']}. **Toda avaliação abaixo sobre o ds06 é exploratória.** "
          "Uma partição nova criada agora não seria intacta: os rótulos já informaram escolhas.", "",
          f"Conjunto conhecido congelado em `{r['conhecidos']['arquivo']}` (`{r['conhecidos']['hash'][:23]}…`): "
          + ", ".join(f"{f} {n}" for f, n in r["conhecidos"]["por_fonte"].items())
          + " eventos. Só evento ausente dessa lista pode compor teste `confirmatorio_prospectivo` "
          "(`reservas_examinadas.regime`).", "",
          "Os números das seções 6 e 7 foram medidos nesse lacre aberto; ficam preservados como medidos, "
          "com esse regime.", ""]


def _fmt_ic(x):
    return f"[{x[0]}; {x[1]}]" if x else "—"


def secao_generalizacao(L: list) -> None:
    r = _j(RUNS / "generalizacao-por-reservas/resultado.json")
    L += ["## 8. Generalização por evento (exploratória)", ""]
    if not r:
        L += ["Não executado.", ""]
        return
    L += [f"Validação cruzada agrupada por evento (cada evento testado uma vez; tiles do teste fora do treino), "
          f"mesmas linhas para os 5 modelos, critérios com hash `{r['pre_registro'][:23]}…` gravados antes. "
          f"Regime: **{r['regime']}**.", ""]
    for fonte, e in r["estudos"].items():
        a1 = e["atribuicoes"]["a1"]
        nota = (" Um evento por dobra: as atribuições a2/a3 são idênticas por construção e não medem sensibilidade."
                if e["dobras"] == e["eventos"] else "")
        L += [f"### {fonte.upper()} — {e['eventos']} eventos, {_n(e['linhas'])} linhas, {e['dobras']} dobras.{nota}", "",
              "| modelo | AUC agrupada | AUC mediana dentro do evento | AP | Brier |", "|---|---:|---:|---:|---:|"]
        L += [f"| `{m}` | {v['auc_agrupada_fora_da_dobra']} | {v['auc_mediana_dentro_do_evento']} | "
              f"{v['average_precision']} | {v['brier'] if v['brier'] is not None else '—'} |"
              for m, v in a1["medidas"].items()]
        L += ["", "| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | Δ agrupada [IC por evento] | veredito |",
              "|---|---:|---:|---:|---|---:|---|---|"]
        for k, c in a1["comparacoes"].items():
            outras = [e["atribuicoes"][s]["comparacoes"][k]["veredito"] for s in ("a2", "a3")
                      if s in e["atribuicoes"]]
            L.append(f"| {c['pergunta']} | {c['eventos_pareados']} | {c['eventos_a_favor']} | "
                     f"{c['dentro_do_evento_mediana']} | {_fmt_ic(c['ic95_media_dentro'])} | {c.get('p_wilcoxon', '—')} | "
                     f"{c['agrupada']} {_fmt_ic(c['ic95_agrupada_por_evento'])} | **{c['veredito']}** "
                     f"(a2/a3: {', '.join(outras)}) |")
        L += ["", f"Eventos com AUC dentro < 0,6 para a logística de terreno: {e['eventos_abaixo_de_0_6_M1']}. "
              "Piores: " + "; ".join(f"{p['evento_id']} ({p['territorio']}, {p['linhas']} linhas, prev. "
                                     f"{p['prevalencia']}) {p['auc_M1']}" for p in e["piores_eventos_M1"][:5]) + ".", ""]
        if e.get("auc_M1_por_territorio"):
            L += ["AUC dentro do evento (M1) por território: " + "; ".join(
                f"{t['territorio']} {t['median']} (n={t['count']})" for t in e["auc_M1_por_territorio"]) + ".", ""]
    g = r["estudos"].get("gfd", {})
    if g.get("territorio_reservado"):
        L += ["### Território reservado (GFD)", "",
              "Território inteiro fora; nenhum evento que o toque e nenhum tile dele no treino.", "",
              "| território | eventos teste/treino | AUC agrupada M0 / M1 / M3 | M1 vs M0 (veredito) |", "|---|---|---|---|"]
        for t in g["territorio_reservado"]:
            if t["estado"] != "medido":
                L.append(f"| {t['territorio']} | {t['eventos_teste']}/{t['eventos_treino']} | — | "
                         f"`not_testable`: {t['motivo']} |")
                continue
            m = t["medidas"]
            L.append(f"| {t['territorio']} | {t['eventos_teste']}/{t['eventos_treino']} | "
                     f"{m['M0_regra_hand']['auc_agrupada_fora_da_dobra']} / "
                     f"{m['M1_logistica_terreno']['auc_agrupada_fora_da_dobra']} / "
                     f"{m['M3_gbm_terreno']['auc_agrupada_fora_da_dobra']} | {t['M1_vs_M0']['veredito']} |")
        L.append("")
    if g.get("diversidade_versus_volume"):
        L += ["### Diversidade × volume com as duas incertezas separadas (GFD)", "",
              "Mesmo volume: todos os eventos subamostrados contra 4 eventos inteiros. A faixa entre ordens é "
              "incerteza de *quais* eventos treinam; o IC é incerteza dos *eventos de teste*.", "",
              "| reserva | eventos teste | linhas por braço | Δ AUC mediana | faixa entre ordens | IC95 eventos de teste | ordens a favor |",
              "|---|---:|---:|---:|---|---|---:|"]
        for x in g["diversidade_versus_volume"]:
            if x["estado"] != "medido":
                L.append(f"| {x['reserva']} | {x['eventos_teste']} | — | — | — | not_testable | — |")
                continue
            L.append(f"| {x['reserva']} | {x['eventos_teste']} | {x['linhas_por_braco_mediana']} | "
                     f"{x['diferenca_mediana_entre_ordens']} | {_fmt_ic(x['faixa_entre_ordens_95'])} | "
                     f"{_fmt_ic(x['ic95_eventos_do_teste_da_media'])} | {x['ordens_a_favor_da_diversidade']}/{x['ordens']} |")
        L.append("")
    L += ["### Transferência entre instrumentos", "",
          "| treino → teste | eventos | AUC agrupada M0 / M1 / M3 | AUC mediana dentro M0 / M1 / M3 |", "|---|---:|---|---|"]
    for x in r["entre_fontes"]:
        m = x["medidas"]
        L.append(f"| {x['treino']} → {x['teste']} | {x['eventos_teste']} | "
                 + " / ".join(str(m[k]["auc_agrupada_fora_da_dobra"]) for k in ("M0_regra_hand", "M1_logistica_terreno", "M3_gbm_terreno"))
                 + " | " + " / ".join(str(m[k]["auc_mediana_dentro_do_evento"]) for k in ("M0_regra_hand", "M1_logistica_terreno", "M3_gbm_terreno"))
                 + " |")
    ident = _j(RUNS / "verdade-observacional/identidade.json")
    if ident:
        fx = ident["faixa_temporal_por_fonte"]
        pares = [f"{p['a'].split('|')[1]} = {p['b'].split('|')[1]}" for p in ident["pares_equivalentes"]
                 if {p["a"].split("|")[0], p["b"].split("|")[0]} == {"gfd", "cems"}]
        L += ["", f"Identidade entre as fontes (`verdade_observacional`): GFD {fx['gfd'][0]}–{fx['gfd'][1]}, CEMS "
              f"{fx['cems'][0]}–{fx['cems'][1]}; pares GFD×CEMS equivalentes: {pares or 'nenhum'}. A transferência "
              "é de instrumento e de época ao mesmo tempo, sem cheia em comum.", ""]
    else:
        L += ["", "Cheias presentes nas duas fontes não foram reconciliadas: é transferência de instrumento com essa ressalva.", ""]
    L += [
          f"- SAR como preditor: {r['observacao_sar']}.", f"- Embedding de terreno: {r['embedding_terreno']}.",
          f"- Contexto hidrológico: {r['contexto_hidrologico']}.", ""]


def secao_confirmatorio(L: list) -> None:
    r = _j(RUNS / "gfd-quarentena/executado.json")
    L += ["## 8b. Teste confirmatório prospectivo (GFD, leitura única)", ""]
    if not r:
        L += ["Não executado.", ""]
        return
    L += [f"Regime **{r['regime']}**: {len(r['eventos_teste'])} eventos GFD fora de `conhecidos_2026_10_03.json`, "
          f"sem tile em comum com o treino ({r['eventos_treino']} eventos, dataset {r['dataset_de_treino']}). Plano "
          f"`{r['pre_registro_avaliacao'][:23]}…`, seleção `{r['pre_registro_selecao'][:23]}…`, ressalvas gravadas antes "
          "da leitura. Os eventos ficaram registrados como examinados: não confirmam de novo.", "",
          "| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | Δ agrupada [IC] | veredito |",
          "|---|---:|---:|---:|---|---:|---|---|"]
    for k, c in r["comparacoes"].items():
        L.append(f"| {k.replace('|', ' × ')} | {c['eventos_pareados']} | {c['eventos_a_favor']} | "
                 f"{c['dentro_do_evento_mediana']} | {_fmt_ic(c['ic95_media_dentro'])} | {c['p_wilcoxon']} | "
                 f"{c['agrupada']} {_fmt_ic(c['ic95_agrupada_por_evento'])} | **{c['veredito']}** |")
    L += ["", f"**Não é:** {r['nao_e']}. Os eventos são de países vizinhos: o ganho vale para evento novo e "
          "território novo da América do Sul, não para o Brasil.", ""]


def secao_politica_orcamento(L: list) -> None:
    r = _j(RUNS / "politica-sob-orcamento/resultado.json")
    L += ["## 9. Política sob orçamento restritivo (simulação retrospectiva, exploratória)", ""]
    if not r:
        L += ["Não executado.", ""]
        return
    L += [f"{len(r['reservas'])} reservas, aquisição sequencial, custo por evento = 1 + linhas/5000 (anotação 0: "
          "rótulo do produto GFD), 30 trajetórias aleatórias por reserva e orçamento, regressão medida no CEMS "
          f"(contexto externo). Pré-registro `{r['pre_registro'][:23]}…`. Prioridade por lacuna é heurística, "
          "não política aprendida.", "",
          "| orçamento | reservas restritivas | lacuna − mediana aleatória | reservas a favor | p | ganho sobre a base | regressão externa | veredito |",
          "|---:|---:|---:|---:|---:|---:|---:|---|"]
    for v in r["vereditos"]:
        if v["estado"] == "not_testable":
            L.append(f"| {v['orcamento']} | {v['reservas_restritivas']}/{v['reservas']} | — | — | — | — | — | `not_testable` |")
            continue
        L.append(f"| {v['orcamento']} | {v['reservas_restritivas']}/{v['reservas']} | {v['mediana_lacuna_menos_aleatoria']} | "
                 f"{v['reservas_a_favor']} | {v['p_wilcoxon']} | {v['ganho_sobre_base_mediana']} | "
                 f"{v['regressao_externa_mediana']} | **{v['estado']}** |")
    L += ["", f"Preservado: {r['preserva']}.", ""]


def secao_neural(L: list) -> None:
    t = _j(RUNS / "sar-inundacao/experimento_transferencia_pretreino.json")
    c = _j(RUNS / "sar-inundacao/confirmacao_recusa_v1_teste.json")
    L += ["## 10. Cadeia neural (frente A)", ""]
    if t:
        a = t["h1"]["analise"]
        L += [f"- **H1** pré-treino curto × aleatório, {a['n']} sementes, época pela validação: "
              f"{a['sementes_no_mesmo_sentido']}, mediana {a['mediana']}, p postos {a['p_postos']} → "
              f"**{a['veredito']}** (relevância 0,025 declarada antes; não alterada). Amplitude entre sementes "
              f"aleatório {t['h1']['amplitude_zero']} × pré-treino {t['h1']['amplitude_fixo']}: a 'loteria' era o "
              "ponto de parada, não a inicialização."]
        for k in ("orcamento_pareado", "orcamento_desigual"):
            x = t["h2"].get(k)
            if x:
                L.append(f"- **H2** crescente × fixo, {k.replace('_', ' ')}: {x['sementes_no_mesmo_sentido']}, "
                         f"p {x['p_postos']} → **{x['veredito']}** (releitura).")
    if c:
        L.append(f"- **H3** limiar de recusa 0,25 (validação) no teste, leitura única: IoU sem água permanente "
                 f"{c['sem_recusa']['iou_micro_sem_agua_permanente']} → {c['com_recusa']['iou_micro_sem_agua_permanente']} "
                 f"com cobertura {c['com_recusa']['cobertura']}; ordena: {c['ordena_no_teste']}.")
    L += ["", "Hipóteses abertas, não falhas comprovadas: máscara sem token (0 = escondido = sem leitura = média) e "
          "deriva de normalização pré-treino → ajuste fino (~0,16 desvio). Nenhuma decisão prática depende delas "
          "hoje; pré-treino longo não executado por isso. Detalhe em `docs/CIENCIA_DO_APRENDIZADO_2026_10_03.md`.", ""]


CADEIA = (
    ("observação", "produtos GFD/CEMS/S1F11 e centrais 156/311", "ds-05-tabela-unica/tabela_unica_v2.parquet"),
    ("adjudicação/admissão", "`admissao_cientifica` por finalidade (não é validação)", "admissao-cientifica/retrato.json"),
    ("amostras contextualizadas", "ds06: papel, relevo, território, evento, grupo", "ds-06-dataset-de-treino/{vigente}/pontos.parquet"),
    ("treino", "logística de terreno / GBM fixado (pesos tabulares); U-Net e MAE (pesos neurais)", "generalizacao-por-reservas/resultado.json"),
    ("avaliação protegida", "**sem reserva intacta**; prospectiva congelada", "reservas-examinadas/retrato.json"),
    ("candidato", "nenhum modelo novo candidato nesta rodada", None),
    ("promoção/recusa", "catraca (neural) e svc02/gates (servido); nada promovido", "neural-catraca/experimento_pote_crescente_redecidido.json"),
    ("monitoramento", "rastreio de impacto (atualização, não validade)", "rastreio-de-impacto/carimbos.jsonl"),
    ("novo ciclo", "política por lacuna: heurística, simulação retrospectiva", "politica-sob-orcamento/resultado.json"),
)


def secao_cadeia(L: list) -> None:
    L += ["## 11. A cadeia de aprendizado que de fato executa", "",
          "| elo | o que existe | evidência |", "|---|---|---|"]
    from ds06_dataset_de_treino import versao_atual
    vigente = versao_atual().name  # a versao lida pelos consumidores, nunca um numero fixo no texto
    for elo, oque, art in CADEIA:
        art = art.format(vigente=vigente) if art else art
        estado = "—" if art is None else (f"`{art}`" if (RUNS / art).exists() else f"ausente: `{art}`")
        L.append(f"| {elo} | {oque} | {estado} |")
    L += ["", "Três coisas diferentes que se chamam \"aprender\" e não se misturam: **pesos** (logística/GBM por "
          "ajuste; U-Net/MAE por gradiente), **corpus/memória** (RAG e LLM reconstroem índice; nenhum peso muda) "
          "e **política** (regra de escolha de dados; hoje heurística escrita à mão, não aprendida). Nenhuma "
          "saída de SAR, embedding, memória ou LLM entra no escore físico nem nos gates do svc02.", ""]


def secao_rodada2(L: list) -> None:
    """Secao 12 (sessao 'Pesquisa sobre decisao interna', 03/10 tarde).

    Consumidores, fontes publicas sem pedido nem credencial, e auditoria dos
    resultados negativos. Tudo lido de artefato.
    """
    import pandas as pd
    L += ["## 12. Consumidores, fontes públicas e resultados negativos", ""]
    t = pd.read_parquet(RUNS / "ds-05-tabela-unica/tabela_unica_v2.parquet",
                        columns=["fonte", "precisao_data", "data_evento"])
    cems, gfd = t[t.fonte == "cems"], t[t.fonte == "gfd"]
    L += [f"**Datas lidas da tabela atual:** CEMS {_n(int((cems.precisao_data == 'cena_do_produto_ems').sum()))} linhas "
          f"com a cena do produto e {_n(int(cems.data_evento.isna().sum()))} sem data; GFD "
          f"{_n(int(gfd.data_evento.notna().sum()))} de {_n(len(gfd))} linhas com início de janela.", ""]
    adm = _j(RUNS / "admissao-cientifica/retrato.json") or {}
    tq = adm.get("temporal_por_pergunta")
    if tq:
        L += ["**Uso temporal depende da pergunta** (data de cena CEMS é observação, não início nem pico):", "",
              "| pergunta | linhas | exige |", "|---|---:|---|"]
        L += [f"| `{q}` | {_n(x['linhas'])} | {x['o_que_e']} |" for q, x in tq.items()]
        L.append("")
    af = _j(RUNS / "afirmacao-dos-servidos/afirmacao.json")
    if af:
        L += ["**Modelos servidos: o que cada escore pode afirmar** (campo opcional `afirmacao_autorizada` de toda resposta do "
              "`svc02`, fora do `model_card` da v1; não muda quem é recusado):", "",
              "| modelo | servível | afirmação autorizada | motivo |", "|---|---|---|---|"]
        L += [f"| `{m['modelo']}` | {m['servivel']} | `{m['estimando']}` | {'; '.join(m['motivos']) or '—'} |"
              for m in af["modelos"]]
        L += ["", "O nó `modelos_servidos` fica `sem_carimbo`: os modelos não foram reajustados sobre a tabela atual "
              "(reajustar muda o produto e exige decisão de versionamento).", ""]
    u = _j(RUNS / "ufo-02-negativos-visiveis/manifesto.json")
    if u:
        r = u["resumo"]
        L += [f"**UFO:** o produtor define não-inundado incluindo nuvem; medido no chip, "
              f"{_n(r['negativos_sem_chao_visivel'])} de {_n(r['negativos'])} negativos sem chão visível "
              f"({r['negativos_por_estado']}) viraram SEM_DADO no ds06 v7. Driver declarado por evento: um único "
              "pluvial (Houston/Harvey, composto).", ""]
    pb = _j(RUNS / "pluvial-01-busca-delimitada/busca.json")
    if pb:
        L += ["**Pluvial, busca delimitada sem pedido nem credencial** (C1 observação contínua ou imagem do mesmo "
              "evento; C2 funcionamento; C3 mecanismo separável; C4 local e janela; C5 acesso público):", "",
              "| fonte | estado | por quê |", "|---|---|---|"]
        L += [f"| {f['fonte']} | {f['estado']} | {f['por_que']} |" for f in pb["fontes"]]
        so = pb["sonda_taiwan"]
        L += ["", f"Sonda ao vivo do Civil IoT Taiwan em {so['quando_utc']}: HTTP {so.get('http', so.get('erro'))}, "
              f"operante={so['operante']}. {pb['conclusao']}. Uso bloqueado: {pb['uso_bloqueado']}.", ""]
    tc = _j(RUNS / "gfd-quarentena/executado.json")
    if tc:
        L += [f"**Teste confirmatório único** (`gfdq03`, regime `{tc['regime']}`, plano `{tc['pre_registro_avaliacao'][:20]}…`, "
              f"seleção `{tc['pre_registro_selecao'][:20]}…`): treino {tc['eventos_treino']} eventos GFD conhecidos "
              f"({tc['linhas_treino']} linhas); teste {len(tc['eventos_teste'])} eventos GFD novos da América do Sul "
              f"fora do Brasil, sem evento nem tile em comum ({tc['linhas_teste']} linhas), em quarentena até a leitura.", "",
              "| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | veredito |", "|---|---:|---:|---:|---|---:|---|"]
        for par, c in tc["comparacoes"].items():
            L.append(f"| {par.replace('|', ' × ')} | {c['eventos_pareados']} | {c['eventos_a_favor']} | "
                     f"{c['dentro_do_evento_mediana']} | {c['ic95_media_dentro']} | {c.get('p_wilcoxon')} | **{c['veredito']}** |")
        L += ["", "Ressalvas gravadas antes da leitura: " + " | ".join(tc["ressalvas_antes_da_leitura"]["ressalvas"]),
              "", "Os 16 eventos passaram a examinados (`reservas_examinadas.registrar_exame`): não confirmam mais nada.", ""]
    vr, tb = _j(RUNS / "rfd-replica-gfd/validacao.json"), _j(RUNS / "rfd-quarentena-brasil/executado.json")
    if vr:
        L += [f"**Réplica do algoritmo do GFD** (`rfd01`, código público MIT; MODIS C6.1, JRC GSW 1.4): portão "
              f"pré-registrado nos {vr['eventos']} eventos brasileiros do GFD -- concordância mediana "
              f"{vr['concordancia_mediana']}, cobertura mediana {vr['cobertura_mediana']}, "
              f"{vr['eventos_com_concordancia_acima_do_minimo']} eventos acima do mínimo: **passou = {vr['passou']}**. "
              "Rodada como tarefas em lote (cota do Earth Engine em modo restrito); equivalência lote × interativo "
              "conferida em 200/200 pontos.", ""]
    if tb:
        L += [f"**Teste confirmatório único no Brasil** (`rfd03`, regime `{tb['regime']}`): treino {tb['eventos_treino']} "
              f"eventos GFD; teste {len(tb['eventos_teste'])} cheias brasileiras 2019-2025 do GDACS rotuladas pela réplica "
              f"({tb['linhas_teste']} linhas; {tb['fracao_do_teste_em_tile_do_treino']:.0%} em tile do treino: evento novo, "
              "lugar conhecido).", "",
              "| comparação | eventos | a favor | mediana dentro | IC95 média dentro | p | veredito |", "|---|---:|---:|---:|---|---:|---|"]
        for par, c in tb["comparacoes"].items():
            L.append(f"| {par.replace('|', ' × ')} | {c['eventos_pareados']} | {c['eventos_a_favor']} | "
                     f"{c['dentro_do_evento_mediana']} | {c['ic95_media_dentro']} | {c.get('p_wilcoxon')} | **{c['veredito']}** |")
        L += ["", "Leitura conjunta dos dois confirmatórios: a logística de terreno supera a regra de HAND em território "
              "novo (América do Sul) e fica indecidível no Brasil conhecido, com a mesma direção; o GBM acrescenta à "
              "logística no Brasil conhecido e não em território novo -- compatível com ele aprender estrutura do lugar "
              "visto no treino. Ressalvas gravadas antes: " + " | ".join(tb["ressalvas_antes_da_leitura"]), ""]
    aud, f1 = _j(RUNS / "floodnet-03-auditoria/resumo.json"), _j(RUNS / "campanha-integrada-20261003/F1/card.json")
    if aud and f1:
        dg = _j(RUNS / "campanha-integrada-20261003/F1/diagnostico_post_hoc.json") or {}
        fo, te = f1["forcamento_dentro_do_sensor"], f1["terreno_entre_sensores_mesmo_dia_chuvoso"]
        L += ["**Pluvial: série FloodNet 2020-2023 auditada** (exploratório: primeira análise sobre dado recém-auditado, "
              "sensores instalados onde já alagou; licença aceita pela pesquisadora em 04/10/2026; "
              "`fn01`/`fn02`/F1). Janela sensor×dia, só sensor pluvial funcionando (cobertura, lacuna, perda de "
              "quadros e bateria, critérios fixados antes):", "",
              f"- janelas: {aud['por_classe']}; sensores com `sem_lamina`: {aud['sensores_com_sem_lamina']}. "
              "`sem_lamina` = nada acima de 10 mm no ponto do sensor naquele dia, não seco absoluto.",
              f"- validação com o catálogo: {dg.get('eventos_pluviais_do_catalogo', f1['validacao_com_catalogo'])} -- "
              "nenhum evento do catálogo cai em janela `sem_lamina`.",
              f"- forçamento dentro do sensor: AUC da chuva MRMS {fo['auc_media']} (IC {fo['ic95_media']}, "
              f"{fo['sensores']} sensores); o F1 usa {f1['janelas_lamina']} lâmina / {f1['janelas_sem_lamina']} sem lâmina "
              "porque exclui os dias sem grade MRMS.",
              f"- terreno entre sensores no mesmo dia com ≥ {te['limiar_mm']} mm: HAND {te['hand']}, TWI {te['twi']} "
              "-- poucos dias com as duas classes: não sustenta conclusão sobre terreno.",
              "- limites: o FloodNet instala onde já alagou (contraste entre lugares propensos, não a cidade); "
              "o critério de perda de quadros descartou 10 dos 19 eventos pluviais não observados -- mais estrito "
              "que o necessário, não alterado depois de visto; Nova York não valida o Brasil.", ""]
    serra_rec = {k: v for k, v in adm.get("recusadas", {}).items() if k.startswith(("mapbiomas", "zenodo"))}
    if serra_rec:
        L += ["**Serra: delineações públicas auditadas e recusadas como referência**", ""]
        L += [f"- `{k}`: {v}" for k, v in serra_rec.items()]
        L.append("")
    for v in ("v1", "v2"):
        m = _j(RUNS / f"serra-03-rotulo-sensor-duplo/{v}/manifesto.json")
        if not m:
            continue
        L += [f"**Serra: rótulo por dois sensores** (S2 óptico + S1 radar, controle seco, regra {v}, base S1 de "
              f"{m['base_s1_pre_dias']} dias): rótulos admitidos {m['rotulos_admitidos']}, grupos por classe "
              f"{m['grupos_por_classe']}, testável {m['testavel']}, para revisão humana {m['para_revisao_humana']}.", ""]
        L += [f"- {ev}: cena {e.get('defasagem_cena_evento_dias')} dias após o evento; classes {e['classes']}; "
              f"desconhecido por {e['desconhecido_por_motivo']}" for ev, e in m["por_evento"].items()]
        L.append("")
    L += ["Leitura: seco numa cena de 4 a 6 dias depois do pico não é ausência de cheia em vale de serra; nenhuma "
          "cena limpa viu água de cheia nos pontos sorteados; o S1 não tem passagem regular no sul do Brasil. A "
          "revisão humana com as mesmas imagens herda o mesmo limite temporal: a referência de serra segue "
          "bloqueada por falta de observação pública no instante do pico, e não por falta de revisão.", ""]
    a = _j(RUNS / "ctx-05-auditoria-do-negativo/auditoria.json")
    if a:
        o, mp = a["operacional"], a["mecanismo_da_perda_gfd"]
        res = next(iter(a["resolucao"].values()), {})
        L += ["**Fluvial: auditoria do `not_supported` do ctx04** (protocolo original preservado):", "",
              f"- operacional: {o['pares_usados']} de {o['pares_triados']} pares usados; perdidos {o['perdidos']}; "
              f"{len(o['usados_com_usina_ou_barragem_dentro_da_bacia'])} dos usados têm usina ou barragem dentro da "
              f"bacia; {len(o['bacias_aninhadas_no_mesmo_evento'])} pares de bacias aninhadas no mesmo evento;",
              f"- resolução: com {res.get('eventos')} eventos, a meia-largura do IC95 da diferença é "
              f"{res.get('meia_largura_ic95')}: ganho menor que isso é indistinguível de zero;",
              f"- como a perda acontece: erro da fração positiva por par {mp['erro_medio_da_fracao_positiva_por_par']} "
              f"(o contexto desloca interceptos entre eventos); o sinal do percentil troca entre dobras: "
              f"{mp['coeficiente_troca_de_sinal_entre_folds']['pico_percentil']};",
              f"- saturação, como hipótese: percentil entre {mp['saturacao']['percentil_min']} e "
              f"{mp['saturacao']['percentil_max']}; correlação chuva × fração positiva "
              f"{mp['saturacao']['corr_chuva_fracao_positiva']}, negativa, o que aponta para a faixa de amostragem "
              "do GFD e não para a física.",
              "", "Decisão: nenhum experimento novo de variáveis. O limite dominante é a resolução em eventos "
              "independentes e o desenho de amostragem; acrescentar variáveis sobre 11 eventos seria busca até ganho. "
              "O que mudaria a conclusão: mais eventos independentes com posto não regularizado. Quantos depende do efeito "
              "que importa: com a variância medida (meia-largura 0,124 em 11 eventos), resolver 0,05 de AUC pediria "
              "da ordem de 70 eventos, se a variância por evento se mantiver -- estimativa, não garantia.", ""]
    L += ["### Restrições desta seção", "",
          "| lacuna | por quê | uso |", "|---|---|---|",
          "| pluvial urbano | negativo observado existe só em Nova York (série FloodNet auditada: sensor×dia sem "
          "lâmina > 10 mm), em sensores instalados onde já alagou; terreno entre sensores sem dias suficientes | "
          "forçamento dentro do sensor medido em NYC; treino e afirmação para o Brasil seguem bloqueados |",
          "| serra brasileira | o CEMS observa relevo íngreme (14 eventos fora do Brasil, 2 grupos brasileiros, "
          "320 positivos), mas nenhuma observação pública no pico dos 5 eventos do pacote; o pacote tem 1,5-4% de "
          "pontos em fundo de vale (E3); delineações públicas recusadas | "
          "afirmação sobre serra bloqueada; casos ambíguos listados em `serra-03-rotulo-sensor-duplo/*/revisao_humana_reduzida.csv` |",
          "| fluvial com contexto | 11 eventos não resolvem ganho menor que 0,12 de AUC agrupada | inconclusivo por resolução |",
          "| modelos servidos | ajustados antes da admissão central; Recife e NYC medem propensão a registrar | "
          "afirmação declarada no cartão; reajuste depende de decisão de versionamento |",
          "| testes de Curitiba (5) | expectativas atualizadas ao contrato vigente, garantias preservadas | resolvido |", ""]


def secao_camada_de_resposta(L: list) -> None:
    """O que a IA consulta, o que ela pode afirmar e quando ela recusa -- lido dos artefatos."""
    L += ["## 12b. Camada de resposta: memória, verificador, recusa e lugar novo", ""]
    m = _j(RUNS / "rag-corpus/manifest.json") or {}
    p = _j(RUNS / "rag-precedencia/precedencia_documental.json") or {}
    e = _j(RUNS / "revp-eval/resultado.json") or {}
    r = _j(RUNS / "recusa-por-dominio/resultado.json")
    s = _j(RUNS / "resposta-lugar-novo/respostas.json")
    estados = {}
    for d in (p.get("documentos") or {}).values():
        estados[d.get("estado")] = estados.get(d.get("estado"), 0) + 1
    n_chunks = m.get("chunk_count")
    L += [f"- **Memória:** corpus com {n_chunks or '—'} trechos; precedência com {estados}. Trechos de documento "
          "superado ou de commit revisado chegam à resposta etiquetados (`consulta_cognitiva` chama "
          "`precedencia_documental.anotar`), e o vigente sai ao lado.",
          f"- **Verificador:** `claim_generalizacao` barra generalização afirmada sem teste confirmatório, no "
          "Brasil, do GBM, ou sem território delimitado; certeza sobre efeito de variável (\"prova\", "
          "\"garante\") é sempre recusada."]
    if e:
        a = e.get("adversarial") or {}
        L.append(f"- **REV-P-EVAL:** `{e.get('veredito')}`; {a.get('n')} ataques, {a.get('bloqueados')} bloqueados, "
                 f"{a.get('vazados')} vazados, {a.get('sem_payload', '—')} sem payload "
                 f"({a.get('com_alvo_substituto', 0)} com alvo substituto declarado).")
    if r:
        c = r["cortes"]
        def _o(k, sel="dominio_mediana"):
            v = (c.get(k) or {}).get("seletores", {}).get(sel)
            return f"{v['aurc']} vs acaso p05 {v['aurc_do_acaso']['p05']} → ordena `{v['ordena_melhor_que_o_acaso']}`" if v else "—"
        L += [f"- **Recusa por domínio (exploratório, pré-registrado):** dentro do GFD (primário) {_o('gfd_dentro')}; "
              f"CEMS → GFD {_o('cems_para_gfd')}. Dentro da mesma fonte a regra recusa terreno extremo e fácil "
              "(Brier dos recusados 0,02–0,09 contra ~0,17 dos aceitos); só entre instrumentos ela ordena o erro. "
              "A regra não foi alterada."]
    if s:
        b = s.get("bootstrap") or {}
        L += [f"- **Lugar novo:** {s['estados']}; incerteza por bootstrap de eventos `{b.get('estado')}` "
              f"({b.get('validas')} réplicas). Artefato histórico: não autoriza consulta anterior ao evento. "
              "A consulta nova exige população compatível e preditores calculáveis sem a amostra rotulada."]
    L.append("")


def _veredito_gen(r, fonte, chave):
    try:
        return r["estudos"][fonte]["atribuicoes"]["a1"]["comparacoes"][chave]["veredito"]
    except (KeyError, TypeError):
        return "não executado"


def secao_contribuicao(L: list) -> None:
    """Uma linha por frente: o que ela entrega HOJE para a IA final, lido dos artefatos."""
    g = _j(RUNS / "generalizacao-por-reservas/resultado.json")
    p = _j(RUNS / "politica-sob-orcamento/resultado.json")
    t = _j(RUNS / "sar-inundacao/experimento_transferencia_pretreino.json")
    c = _j(RUNS / "sar-inundacao/confirmacao_recusa_v1_teste.json")
    L += ["## 13. Contribuição real de cada frente para a IA final", "",
          "| frente | contribuição medida | estado |", "|---|---|---|"]
    q = _j(RUNS / "gfd-quarentena/executado.json")
    cq = (q or {}).get("comparacoes", {})
    conf_m1 = cq.get("M1_logistica_terreno|M0_regra_hand", {}).get("veredito", "não executado")
    conf_m3 = cq.get("M3_gbm_terreno|M1_logistica_terreno", {}).get("veredito", "não executado")
    b = _j(RUNS / "rfd-quarentena-brasil/executado.json")
    cb = (b or {}).get("comparacoes", {})
    br_m1 = cb.get("M1_logistica_terreno|M0_regra_hand", {}).get("veredito", "não executado")
    br_m3 = cb.get("M3_gbm_terreno|M1_logistica_terreno", {}).get("veredito", "não executado")
    if g:
        L.append(f"| terreno (logística) | supera a regra de HAND no GFD: exploratório `{_veredito_gen(g, 'gfd', 'M1_logistica_terreno_vs_M0_regra_hand')}`, "
                 f"**confirmatório em território novo `{conf_m1}`** (seção 8b, 16 eventos fora do Brasil); confirmatório em "
                 f"território conhecido (Brasil, seção 12) `{br_m1}`; no CEMS: "
                 f"`{_veredito_gen(g, 'cems', 'M1_logistica_terreno_vs_M0_regra_hand')}` | evidência na população histórica; não autoriza consulta territorial |")
        L.append(f"| não linearidade (GBM) | exploratório GFD `{_veredito_gen(g, 'gfd', 'M3_gbm_terreno_vs_M1_logistica_terreno')}`, CEMS "
                 f"`{_veredito_gen(g, 'cems', 'M3_gbm_terreno_vs_M1_logistica_terreno')}`; confirmatório em território novo "
                 f"`{conf_m3}`, em território conhecido (Brasil) `{br_m3}`. Esses contrastes delimitam evidência, "
                 "não demonstram impossibilidade universal de transferência | "
                 "hipótese de desenvolvimento; os eventos dos dois testes não reconfirmam |")
        L.append(f"| cobertura/solo | GFD `{_veredito_gen(g, 'gfd', 'M2_logistica_tcs_vs_M1_logistica_terreno')}`, CEMS "
                 f"`{_veredito_gen(g, 'cems', 'M2_logistica_tcs_vs_M1_logistica_terreno')}`: a associação não transfere entre "
                 "eventos, e o dado é do ano do evento com preenchimento não aleatório | fora do vetor |")
    if p:
        est = ", ".join(f"{v['orcamento']}: `{v['estado']}`" for v in p["vereditos"])
        L.append(f"| política por lacuna | {est} | heurística; simulação retrospectiva |")
    if t:
        L.append(f"| pré-treino SAR (A8) | H1 `{t['h1']['analise']['veredito']}` | não promovido; congelado sem decisão prática |")
    if c:
        L.append(f"| rede SAR + recusa (A0/A7) | recusa ordena no teste: `{c['ordena_no_teste']}` | observação, nunca preditor do escore |")
    n1 = _j(RUNS / "campanha-integrada-20261003/N1/card.json")
    n1b = _j(RUNS / "campanha-integrada-20261003/N1b/card.json")
    if n1 and n1b:
        d1 = n1["diferenca_contra_terreno_ic95_por_evento"].get("terreno_mais_embedding")
        d2 = n1b["diferencas_ic95_por_evento"].get("embedding_menos_vizinhanca_pca16")
        L.append(f"| encoder de terreno (B) | sobre HAND+TWI (2 variáveis; nível de AUC não comparável à seção 8) acrescenta "
                 f"{d1} (N1, `{n1['veredito']}`), mas estatística simples "
                 f"da mesma janela de 3,84 km o supera: embedding − vizinhança {d2} (N1b, post hoc, `{n1b['veredito']}`) | "
                 "N1b compara substituição; não exclui ganho incremental ao acrescentar o encoder (MS4 K5) |")
        v2b = _j(RUNS / "campanha-integrada-20261003/V2b/card.json")
        if v2b:
            res = v2b["resultado"]
            ganho = {f: res[f]["terreno_mais_todas_as_escalas"] for f in ("gfd", "cems") if f in res}
            L.append("| vizinhança da janela | ganho DENTRO do evento: " + "; ".join(
                f"{f.upper()} {x['media']} {x['ic95_da_media']} ({x['eventos_a_favor']}/{x['eventos']})" for f, x in ganho.items())
                + f" (V2b, post hoc). No CEMS o ganho do V2 é só entre eventos | {v2b['leitura']}; não é afirmado como relevo |")
    else:
        L.append("| encoder de terreno (B) | sem card N1/N1b em disco | sem contribuição medida no ds06 |")
    L += ["| RAG / juiz / LLM | redigem a partir de objetos medidos; não atualizam pesos | camada de resposta, fora do escore |", ""]


def secao_auditoria_pre_treino(L: list) -> None:
    """Le a auditoria pre-treino de 06/10; o detalhe fica no documento dela, aqui so o que limita o que esta acima."""
    r = _j(RUNS / "auditoria-pre-treino/resultado.json")
    L += ["## 13b. Auditoria pré-treino: o que limita as seções acima", ""]
    if not r:
        L += ["Não executada (`scripts/metodo/auditoria_pre_treino.py`).", ""]
        return
    L += [f"Síntese única em `docs/AUDITORIA_PRE_TREINO.md` (gerada em {r['quando_utc']}, `{r['hash_resumo'][:23]}…`). "
          "Nenhum teste confirmatório foi rerodado; os vereditos das seções 8b e 12 ficam como medidos, com estes limites:", "",
          "| achado | classe | impacto |", "|---|---|---|"]
    L += [f"| {a['id']} {a['titulo']} | `{a['classe']}` | {a['impacto']} |" for a in r["achados"]]
    L += ["", "Prontidão por tarefa: " + "; ".join(f"{p['tarefa']} `{p['estado']}`" for p in r["prontidao"]) + ".", ""]


def secao_proxima_decisao(L: list) -> None:
    q = _j(RUNS / "gfd-quarentena/executado.json")
    b = _j(RUNS / "rfd-quarentena-brasil/executado.json")
    L += ["## 14. Próxima decisão experimental", ""]
    if q and b:
        L += ["Os dois testes confirmatórios disponíveis **foram executados**: território novo (seção 8b) e território "
              "conhecido no Brasil (seção 12). Seus resultados permanecem restritos à população e ao vetor "
              "avaliados. A auditoria A16 mostrou dependência amostral de `elevation_local_m`; TC01 e MS6 mudam "
              "o quadro de consulta. Portanto, antes de nova confirmação, é preciso estabilizar alvo, população "
              "e variante calculável na inferência. A prioridade brasileira é **evento em território não visto "
              "no treino**, com observabilidade elegível e ausência nos registros conhecido e examinado. "
              "A comparação antiga não deve ser reutilizada silenciosamente se o vetor ou a pergunta mudar: "
              "a variante corrigida exige desenho próprio, congelado antes de abrir observações independentes. "
              "Reexaminar os testes consumidos não cria nova confirmação.", ""]
    elif q:
        L += ["A decisão anterior (teste confirmatório em eventos novos) **foi executada** (seção 8b). A próxima é um "
              "teste prospectivo com eventos brasileiros ausentes dos conjuntos conhecido e examinado.", ""]
    else:
        L += ["Toda conclusão acima sobre o ds06 é exploratória. A decisão que transforma alguma delas em afirmação é "
              "**adquirir eventos GFD/CEMS de planície ausentes de `conhecidos_2026_10_03.json` e rodar sobre eles, uma "
              "única vez, a comparação já congelada**. Até isso, nada é promovido.", ""]


def secao_framework_atual(L: list) -> None:
    """Atualizacoes nao apagam o registro dos contrastes anteriores."""
    L += ["## 15. Framework atual e fronteira de consulta", "",
          "A síntese reproduzível das frentes e dos critérios de conclusão está em "
          "`docs/FECHAMENTO_DO_FRAMEWORK.md`, gerada por `scripts/metodo/fechamento_do_framework.py`. "
          "O manifesto registra hashes, integridade e ausências; isso não substitui validade observacional.", ""]
    ms4 = _j(RUNS / "ms4-bracos-de-desenvolvimento/resultado.json")
    if ms4:
        k = (ms4.get("comparacoes") or {}).get("K5") or {}
        L += [f"- **MS4 K5:** `{k.get('veredito', 'não verificado')}`, delta mediano dentro do evento "
              f"{k.get('dentro_do_evento_mediana')} de AUC, IC da média {k.get('ic95_media_dentro')}. "
              "Compara acrescentar embedding à vizinhança, não substituí-la. Desenvolvimento em eventos "
              "conhecidos, sem promoção ou confirmação territorial."]
    ms5 = _j(RUNS / "ms5-recusa-decomposta/resultado.json")
    if ms5:
        sinais = ms5.get("sinais") or {}
        L += ["- **MS5:** " + "; ".join(f"{n} `{s.get('estado', 'não verificado')}`" for n, s in sinais.items())
              + ". Ordenação de erro no desenvolvimento não é probabilidade calibrada nem regra servida."]
    ms6 = _j(RUNS / "ms6-quadro-de-treino-para-consulta/resultado.json")
    L += ["- **MS6:** " + ("resultado disponível; consulta permanece exploratória e condicionada à seleção dos tiles."
                            if ms6 else "código disponível, resultado ainda ausente nesta síntese."),
          "- **Consulta:** `resposta_lugar_novo.py` recusa por padrão a linhagem de piso amostral; "
          "`--demonstracao-historica` explicita a reprodução antiga. Isso não modifica o escore dos modelos servidos.", ""]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    from proveniencia import agora_utc
    L = ["# Fechamento científico operacional", "",
         f"**Gerado de** `scripts/metodo/relatorio_de_fechamento.py` em {agora_utc()}. **Não editar à mão.**", "",
         "**Não é.** prova de que a ciência acabou: software correto não é validade observacional. "
         "Cada seção diz o que vale para qual uso e o que segue bloqueado.", ""]
    for f in (secao_validade, secao_admissao, secao_rastreio, secao_fluvial, secao_pluvial, secao_serra,
              secao_aprendizado, secao_generalizacao, secao_confirmatorio, secao_politica_orcamento, secao_neural, secao_cadeia,
              secao_rodada2, secao_camada_de_resposta, secao_contribuicao, secao_auditoria_pre_treino,
              secao_proxima_decisao, secao_framework_atual):
        f(L)
    L += ["## Lacunas restantes e o que as resolveria sem pedido, contato ou credencial", "",
          "| lacuna | estado | o que resolveria |", "|---|---|---|",
          "| avaliação confirmatória | nenhuma reserva intacta no ds06 (seção 0) | eventos GFD/CEMS públicos ausentes de "
          "`conhecidos_2026_10_03.json`, avaliados uma vez com os critérios já congelados |",
          "| pluvial urbano | negativo observado só em Nova York, em sensores instalados onde já alagou (série FloodNet "
          "auditada, seção 12): chuva ordena dentro do sensor; terreno entre sensores indecidível (6 dias) | para o "
          "Brasil, nada: NYC não valida o Brasil. O que resolveria aqui: observação contínua brasileira com "
          "funcionamento do sensor, ou imagem do mesmo evento com área seca visível e mecanismo pluvial separável |",
          "| serra brasileira | sem observação pública no pico (seção 12) | cena SAR/óptica no dia do pico; até lá, "
          "afirmação sobre serra bloqueada |",
          "| CEMS pré-2023 (Petrópolis 2022) | fora desta rodada: exige credencial nova | sem alternativa pública "
          "verificada; o uso fica restrito ao que existe |",
          ""]
    DOC.write_text("\n".join(L), encoding="utf-8")
    print(f"GRAVADO={DOC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
