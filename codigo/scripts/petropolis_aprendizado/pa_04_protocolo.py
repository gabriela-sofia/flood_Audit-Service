"""Congela o protocolo v1 antes de qualquer ajuste ou escore no teste.

Grava protocolo_v1.json (lido pelo treino, que recusa rodar sem ele ou com
amostras diferentes das congeladas) e protocolo_v1.md (leitura humana).
Registra tambem o que ja havia sido examinado ate aqui.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pa_comum import OUT, ROOT, Etapa, gravar_json, ler_json, rel, sha256

REF = OUT / "referencias"

PROTOCOLO = {
    "id": "petropolis-aprendizado-v1-protocolo-1",
    "pergunta": ("conhecimento de encostas aprendido em area observada (Idai) transfere para Petropolis e "
                 "melhora com poucas observacoes locais historicas, avaliado no episodio de fev/2022 fora dos "
                 "exemplos de atualizacao?"),
    "tarefa_primaria": {
        "nome": "T1 ordenacao presenca-fundo de cicatrizes de 2022",
        "alvo": "celulas de 30 m dentro de poligonos INPE/Charter tipo Deslizamento (ruptura+percurso, nao separados)",
        "fundo": "celulas do suporte fora de qualquer feicao do inventario; NAO rotulado; nao e encosta estavel",
        "suporte": "envoltoria convexa das 85 feicoes (11,88 km2)",
        "entradas": "7 atributos de terreno FABDEM anteriores ao evento; nenhuma imagem pos-evento; sem zoneamento 2025",
        "natureza_do_escore": "ordenacao relativa; nao probabilidade fisica; AUC presenca-fundo nao e acuracia territorial",
    },
    "desfecho_primario": {
        "nome": "AUC_obj",
        "definicao": ("para cada cicatriz s: media, sobre suas celulas, da fracao do fundo com escore menor "
                      "(empates contam 1/2); AUC_obj = media sobre as 81 cicatrizes"),
        "por_que": "cada cicatriz pesa igual; cicatrizes grandes nao dominam; unidade e o objeto, nao o pixel",
    },
    "desfechos_secundarios": [
        "AP celular com prevalencia 0,0222 declarada; AUC celular",
        "recuperacao de cicatrizes por orcamento de area q em {0,1; 0,2; 0,3}: fracao de cicatrizes com percentil medio >= 1-q",
        "omissoes por comunidade (FCU que intersectam o suporte de teste)",
        "T_hist: 622 sitios historicos de deslizamento contra o dominio de relato; so bracos sem rotulo local",
        "origem: AUC no Idai reserva_origem (fonte vs ausencia mapeada no evento; referencia de duas classes no suporte)",
    ],
    "atributos": ["decl_m3", "decl_max5", "curv_plan_m3", "curv_perf_m3", "logsca_m3", "relevo7", "tpi11"],
    "padronizacao": "media e desvio do ajuste Idai, comum a todos os bracos",
    "bracos": {
        "A0_regra": "escore = decl_m3 (ordenacao por declive)",
        "A1_log": "logistica penalizada ajustada so no Idai (ajuste); lambda escolhido no desenvolvimento Idai em {0,01;0,1;1;10;100}",
        "A1_hgb": "HistGradientBoosting so no Idai; grade max_depth {3,6} x learning_rate {0,05;0,1}; 300 iteracoes; min_samples_leaf 50; escolha no desenvolvimento Idai",
        "A2_log_l1/A2_log_l25": "logistica so com exemplos locais; prior zero; precisao lambda em {1, 25}",
        "A3_log_l1/A3_log_l25": "logistica local com prior centrado nos coeficientes A1_log; mesma precisao do A2 pareado",
        "A3perm_log_l25": "controle: prior centrado nos coeficientes A1_log permutados entre atributos (permutacao por semente, nunca identidade)",
        "A2_hgb": "boosting so local: 50 iteracoes, learning_rate 0,05, max_depth 3, min_samples_leaf 10",
        "A3_hgb": "A1_hgb + 50 iteracoes locais com os mesmos parametros (warm start; bins do doador)",
    },
    "objetivo_logistico": "soma ponderada da log-perda (positivo peso 1, fundo peso n_pos/n_fundo) + lambda/2 ||beta-beta0||^2; intercepto livre",
    "orcamentos": {"B": [10, 30, 100, 300], "unidade": "sitios historicos unicos (celula de 30 m) amostrados do pool",
                   "fundo_por_ajuste": "3B celulas do dominio de relato fora da exclusao e a >60 m de relato de deslizamento",
                   "sementes": 10, "nota": "sementes medem estabilidade computacional; nao sao episodios",
                   "B300_cobre": "75% do pool de 402 sitios: amostras de sementes se sobrepoem fortemente",
                   "B_todos": "402 sitios, 10 sorteios de fundo; usado na resposta territorial 'depois' e avaliado em T1"},
    "contrastes_primarios": {
        "orcamento": 100, "lambda": 25,
        "C1_transferencia": "A1_log - A0_regra",
        "C2_aprendizado_local": "A3_log_l25 - A1_log",
        "C3_valor_conhecimento_externo": "A3_log_l25 - A2_log_l25",
        "C4_controle_estrutura": "A3_log_l25 - A3perm_log_l25",
    },
    "regra_de_interpretacao": {
        "margem": 0.02,
        "sustentado": "media nas sementes >= +0,02 e IC95 bootstrap exclui 0",
        "contrario": "media <= -0,02 e IC95 exclui 0",
        "inconclusivo": "demais casos",
        "justificativa_margem": ("menor diferenca tratada como material nos contrastes anteriores do projeto (v3 bloco +0,025; "
                                 "MT01 +0,039) e abaixo do erro-padrao esperado de AUC com 81 objetos (~0,03); "
                                 "nao foi escolhida olhando resultados desta campanha"),
        "bootstrap": "2000 reamostragens: cicatrizes com reposicao e fundo por blocos de 500 m com reposicao; descritivo de um episodio",
    },
    "retencao": "Delta AUC no Idai reserva_origem: A3 contra A1 da mesma familia; custo se <= -0,02",
    "ablacoes_descritivas": "A1_log sem cada bloco (declive, curvatura, fluxo, relevo): Idai reserva, T1, T_hist",
    "diagnosticos": {
        "D1_vies_de_fundo": "A2_log_l1 B=300 com fundo do municipio inteiro em vez do dominio de relato",
        "D2_agua": "escore de agua svc-03 (fluvial_serra) avaliado em T1: escore baixo de agua nao e seguranca de encosta",
        "D3_zoneamento_2025": "nao avaliado como preditor (posterior ao episodio); so contexto",
        "D4_dominio": "distribuicoes Idai x Petropolis e fracao das FCU dentro de P1-P99 do doador",
    },
    "resposta_territorial": {
        "componente_encosta": "A1_log ('antes', transferido) e A3_log_l25 B_todos ('depois', atualizado), media dos 10 sorteios",
        "percentil": "relativo ao dominio municipal valido",
        "estado": ("inferido_avaliado_no_episodio_2022 se o contraste correspondente for sustentado; senao inferido_nao_validado; "
                   "regra fixada aqui, antes do teste"),
    },
    "limite_de_busca": "uma execucao do protocolo; qualquer mudanca e emenda com novo id; resultados v1 preservados",
    "bloqueios_conhecidos": [
        "ausencias fisicas observadas em Petropolis: inexistentes; so fundo",
        "dados de Ferreira et al. e Alcantara et al.: mediante solicitacao; sem contato externo",
        "USGS Maria: anti-robo; nao contornado",
        "vetor DRM de 611 cicatrizes: citado em PDF, nao vetorial no acervo",
        "deteccao pos-evento: sem imagem pos-evento qualificada nesta linhagem",
    ],
}


def main() -> int:
    destino = OUT / "protocolo" / "protocolo_v1.json"
    if destino.exists():
        print("protocolo v1 ja congelado; mudancas exigem emenda")
        return 1
    e = Etapa("pa_04_protocolo")
    refs = ler_json(REF / "referencias.json")
    scripts = sorted((ROOT / "scripts/petropolis_aprendizado").glob("pa_*.py"))
    conteudo = dict(PROTOCOLO)
    conteudo["congelado_em"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conteudo["amostras_sha256"] = refs["hashes"]
    conteudo["scripts_sha256_no_congelamento"] = {rel(p): sha256(p) for p in scripts}
    conteudo["ja_examinado"] = {
        "contagens": {k: v for k, v in refs.items() if k != "hashes"},
        "terreno": "percentis dos 7 atributos nas duas regioes (terreno/terreno.json); declive mediano Idai 8 graus, Petropolis 20",
        "nada_ajustado": "nenhum modelo ajustado; nenhum escore calculado no teste 2022, em T_hist ou no Idai antes deste congelamento",
        "visto_antes_desta_linhagem": ("contagens e intersecoes do Charter com FCU (estudo multiperigos v1, v3); "
                                       "pool de 412 sitios por consulta exploratoria antes de fixar status ALTA/MEDIA"),
    }
    gravar_json(destino, conteudo)
    md = ["# Protocolo v1 — aprendizado de encostas em Petrópolis", "",
          f"Congelado em {conteudo['congelado_em']}, antes de qualquer ajuste ou escore no teste.", ""]
    for k, v in PROTOCOLO.items():
        md.append(f"## {k}")
        if isinstance(v, dict):
            for a, b in v.items():
                md.append(f"- **{a}**: {b}")
        elif isinstance(v, list):
            md += [f"- {x}" for x in v]
        else:
            md.append(str(v))
        md.append("")
    (OUT / "protocolo" / "protocolo_v1.md").write_text("\n".join(md), encoding="utf-8")
    e.saida(destino, OUT / "protocolo" / "protocolo_v1.md")
    e.fechar(protocolo=conteudo["id"])
    print("protocolo congelado:", sha256(destino))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
