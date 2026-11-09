"""Congela o protocolo v1 de Recife antes de qualquer ajuste ou escore no teste."""
from __future__ import annotations

from datetime import datetime, timezone

from ra_comum import OUT, ROOT, Etapa, gravar_json, ler_json, rel, sha256

PROTOCOLO = {
    "id": "recife-aprendizado-v1-protocolo-1",
    "pergunta": ("o nucleo Q transferido ordena as ruas com relato de alagamento dentro dos bairros de Recife; observacoes "
                 "locais e atributos urbanos (PE3D, drenagem, saida) acrescentam em bairros nao usados na atualizacao?"),
    "tarefa": {
        "nome": "T1 presenca-fundo de ruas com relato SEDEC de agua, ordenadas dentro do bairro",
        "unidade": "objeto rua-bairro (logradouro oficial recortado pelo bairro)",
        "alvo": "rua com >= 1 relato administrativo de agua (Alagamentos / Imoveis alagados) 2014-2024 geocodificado",
        "fundo": "demais ruas dos mesmos bairros; NAO rotulado; nao e rua seca",
        "populacao": "5.477 ruas de 38 bairros com relato geocodificado; 96 positivas",
        "natureza": "priorizacao de registro administrativo de alagamento; nao probabilidade fisica, profundidade ou especificidade",
    },
    "particao": {"unidade": "bairro inteiro", "atualizacao": "19 bairros, 48 positivas, 3.252 ruas",
                 "teste": "19 bairros, 48 positivas, 2.225 ruas",
                 "dependencia": "rede de drenagem e microbacias atravessam bairros; o teste e espacial por bairro, nao por sistema"},
    "desfecho_primario": {"nome": "AUC_bairro",
                          "definicao": "para cada rua positiva do teste, fracao das ruas de fundo do MESMO bairro com escore menor (empates 1/2); media sobre as positivas",
                          "por_que": "o registro depende do bairro; a pergunta e onde, dentro do bairro, a agua foi relatada"},
    "secundarios": ["AUC agrupada no teste", "AP com prevalencia", "recuperacao: fracao das positivas no quintil superior do proprio bairro",
                    "omissoes por bairro e por FCU", "retencao na origem limpa (6 unidades de retencao v10, AUC dentro do evento)"],
    "blocos": {"Q30": ["f_hand", "f_sca", "f_declive", "f_construido", "f_areia", "f_dist_dren", "f_dist_canal"],
               "DREN": ["dren_entradas_100m", "dren_dist_rede_m", "dren_frac_ligadas_5m", "dren_frac_canaleta"],
               "SAIDA": ["saida_log_dist_canal"],
               "PE3D": ["pe3d_rel50", "pe3d_rel200", "pe3d_depr_max", "pe3d_frac_depr10", "pe3d_decl_long", "pe3d_desnivel_canal"],
               "chuva": "nao usada: tarefa estatica multi-ano sem hora; registrado como nao avaliado"},
    "padronizacao": {"Q30": "media/desvio/mediana do proprio nucleo Q (escala externa comum)",
                     "urbanos": "media/desvio das ruas de atualizacao (nunca do teste); NaN -> mediana da atualizacao + indicador"},
    "bracos": {
        "A0_regra": "escore = -f_hand (HAND de 30 m: rua mais baixa em relacao ao canal primeiro)",
        "A0_pe3d": "escore = -pe3d_rel50 (rua mais baixa que o entorno de 50 m), so no subconjunto PE3D",
        "A1_Q": "nucleo Qsep_sel_fis congelado (q_escore), sem rotulo local",
        "A2_q_l1/l25": "logistica local nos 7 atributos Q (escala Q), prior zero",
        "A3_q_l1/l25": "logistica local com prior centrado nos coeficientes principais de Q",
        "A3perm_l25": "controle: prior centrado nos coeficientes Q permutados entre atributos",
        "A2u_l25": "logistica local com Q30+DREN+SAIDA, prior zero (controle com os mesmos atributos)",
        "A4_l25": "Q30 com prior em Q + DREN + SAIDA com prior zero",
        "A4_sem_DREN / A4_sem_SAIDA": "ablacoes do A4",
        "A5_l25 (subconjunto PE3D)": "A4 + PE3D com prior zero; A5_sem_PE3D = A4 no mesmo subconjunto; A2p = local com todos os blocos",
        "A2_hgb": "HistGradientBoosting local com Q30+DREN+SAIDA (200 iter, lr 0,05, max_depth 3, min_samples_leaf 10)",
        "A4_hgb": "HistGradientBoosting com q_escore + DREN + SAIDA (mesmos parametros)",
    },
    "objetivo_logistico": "soma ponderada da log-perda (positivo 1, fundo n_pos/n_fundo) + lambda/2 ||beta - beta0||^2; intercepto livre",
    "orcamentos": {"B": [10, 25, "todas"], "unidade": "ruas positivas de atualizacao sorteadas (consultas)",
                   "fundo": "todas as ruas de fundo dos bairros que contem ao menos uma positiva sorteada",
                   "sementes": 10, "todas": "48 positivas, deterministico",
                   "nota": "B=100/400 nao existe: so ha 48 positivas de atualizacao"},
    "contrastes_primarios": {"B": "todas", "lambda": 25,
                             "C1_transferencia": "A1_Q - A0_regra", "C2_aprendizado_local": "A3_q_l25 - A1_Q",
                             "C3_valor_conhecimento_externo": "A3_q_l25 - A2_q_l25", "C4_controle": "A3_q_l25 - A3perm_l25",
                             "C5_contexto_urbano": "A4_l25 - A3_q_l25", "C5_controle_local": "A2u_l25 - A2_q_l25",
                             "C6_PE3D (subconjunto)": "A5_l25 - A4_l25 no teste PE3D"},
    "regra": {"margem": 0.02, "sustentado": "media >= 0,02 e IC95 exclui 0", "contrario": "media <= -0,02 e IC95 exclui 0",
              "inconclusivo": "demais", "bootstrap": "2000 reamostragens de bairros do teste com reposicao (descritivo; 19 bairros)",
              "justificativa": "mesma margem de Petropolis e dos contrastes anteriores; nao escolhida por estes resultados"},
    "hiperparametros": "nenhuma busca no teste; lambdas fixos {1, 25}; boosting com parametros fixos acima",
    "retencao": "coeficientes Q apos a atualizacao aplicados as 6 unidades de retencao (efeitos principais; interacoes de Q < 1e-4)",
    "resposta_territorial": "candidata com terreno, captacao, rede, saida, agua_observada, inferencia_aprendida, contexto_social, fontes, limites, proxima_observacao",
    "limite_de_busca": "uma execucao; mudancas viram emenda versionada",
    "bloqueios_conhecidos": ["sem negativos observados (fundo)", "episodio 2022 sem endereco utilizavel",
                             "Soledade: 5 medidas num ponto, conferencia visual pendente", "chuva subdiaria e mare nao adquiridas (Cemaden com CAPTCHA)",
                             "cotas de fundo/secoes insuficientes: capacidade hidraulica desconhecida; SWMM nao executado"],
}


def main() -> int:
    destino = OUT / "protocolo" / "protocolo_v1.json"
    if destino.exists():
        print("protocolo ja congelado")
        return 1
    e = Etapa("ra_03_protocolo")
    refs = ler_json(OUT / "referencias/referencias.json")
    atr = ler_json(OUT / "atributos/atributos.json")
    conteudo = {**PROTOCOLO, "congelado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "referencias_sha256": refs["hashes"], "atributos_sha256": atr["sha_atributos"],
                "scripts_sha256": {rel(p): sha256(p) for p in sorted((ROOT / "scripts/recife_aprendizado").glob("ra_*.py"))},
                "ja_examinado": {"contagens_geocodificacao": {k: v for k, v in refs.items() if k != "hashes"},
                                 "atributos": "faltantes e contagens (atributos.json); nenhuma relacao atributo x rotulo examinada",
                                 "antes_desta_linhagem": ("v6-v10 usaram a referencia Sentinel-2 E2 de 2022 na grade de Recife; "
                                                          "o modelo servido recife_pluvial usa pontos SEDEC de monitoramento; "
                                                          "nenhum deles usou ruas geocodificadas")}}
    gravar_json(destino, conteudo)
    md = ["# Protocolo v1 — alagamento urbano em Recife", "", f"Congelado em {conteudo['congelado_em']}.", ""]
    for k, v in PROTOCOLO.items():
        md.append(f"## {k}")
        if isinstance(v, dict):
            md += [f"- **{a}**: {b}" for a, b in v.items()]
        elif isinstance(v, list):
            md += [f"- {x}" for x in v]
        else:
            md.append(str(v))
        md.append("")
    (OUT / "protocolo" / "protocolo_v1.md").write_text("\n".join(md), encoding="utf-8")
    e.saida(destino)
    e.fechar(protocolo=conteudo["id"])
    print("protocolo congelado:", sha256(destino))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
