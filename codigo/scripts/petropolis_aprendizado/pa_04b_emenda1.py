"""Congela a emenda 1 antes de executa-la. Desenhada DEPOIS dos resultados do protocolo v1.

Motivo (diagnostico do v1, nao ajuste olhando o teste para ganhar): os relatos
historicos UFRJ/GEOCART ficaram abaixo do acaso no teste secundario e as
atualizacoes com eles inverteram o modelo. Medianas: relatos em posicao baixa e
concava (tpi -9 m, declive 14 graus) contra cicatrizes de 2022 a 25 graus. Hipotese:
o endereco marca o RECEPTOR do dano, nao a ruptura.

Esta emenda nao substitui nem reinterpreta o v1. Ela acrescenta:
  D5  diagnostico de receptor no teste historico (desenvolvimento, sem 2022);
  E1  aprendizado local com o tipo certo de observacao (cicatrizes mapeadas),
      validacao cruzada espacial por bairro DENTRO do episodio de 2022, que ja foi
      o teste do v1: resultado e desenvolvimento, nao confirmacao.
"""
from __future__ import annotations

from datetime import datetime, timezone

from pa_comum import OUT, ROOT, Etapa, gravar_json, rel, sha256

EMENDA = {
    "id": "petropolis-aprendizado-v1-emenda-1",
    "base": "petropolis-aprendizado-v1-protocolo-1 (resultados preservados, nao reinterpretados)",
    "motivo": ("v1: C2 contrario, relatos historicos abaixo do acaso; diagnostico de semantica posicional "
               "(dano/receptor x ruptura). A emenda testa essa explicacao e testa o aprendizado local com "
               "observacoes de natureza compativel com o alvo."),
    "D5_receptor": {
        "tarefa": "T_hist (622 sitios historicos x dominio de relato elegivel), sem rotulo de 2022",
        "escores": {"A1_cel": "A1_log na propria celula (igual ao v1)",
                    "A1_montante7": "maximo do escore A1_log entre celulas que drenam para o sitio em ate 7 passos D8 (~210 m)",
                    "A0_montante7": "maximo da declividade decl_m3 a montante em ate 7 passos D8"},
        "estratos": "precisao do geocodigo: ROOFTOP+RANGE_INTERPOLATED x GEOMETRIC_CENTER+APPROXIMATE",
        "leitura": ("se os relatos forem receptores, A1_montante7 > 0,5 enquanto A1_cel < 0,5; isso apoia a "
                    "explicacao, nao prova a cadeia causal de cada relato"),
    },
    "E1_cicatrizes_cv_bairro": {
        "unidades": "81 cicatrizes de deslizamento 2022; dobras = 5 bairros do inventario (Alto da Serra 21, Caxambu 10, Cel. Veiga 14, Centro 22, Chacara Flora 14)",
        "atribuicao_de_celulas": "cada celula do suporte vai para o bairro da feicao (nao inundacao) de centroide mais proximo",
        "ajuste_dobra_k": "cicatrizes e fundo das outras dobras; fundo a >150 m de qualquer celula da dobra k",
        "avaliacao": "celulas da dobra k; percentil de cada cicatriz contra o fundo da propria dobra; AUC_obj_oof = media sobre as 81",
        "pesos": "cada cicatriz soma peso 1 (dividido entre suas celulas); fundo soma o mesmo total",
        "orcamentos_K": [5, 15, "todas"], "sementes": 10, "nota_K": "K = cicatrizes de ajuste sorteadas entre as dobras de treino; 'todas' e deterministico",
        "fundo": "todas as celulas elegiveis das dobras de treino (sem sorteio)",
        "bracos": ["A0_regra", "A1_log", "A2_log_l1", "A2_log_l25", "A3_log_l1", "A3_log_l25", "A3perm_log_l25", "A1_hgb", "A2_hgb", "A3_hgb"],
        "primario": {"K": "todas", "lambda": 25,
                     "E1_C2": "A3_log_l25 - A1_log", "E1_C3": "A3_log_l25 - A2_log_l25",
                     "E1_C4": "A3_log_l25 - A3perm_log_l25", "E1_local_vs_regra": "A2_log_l25 - A0_regra"},
        "regra": "a mesma do v1 (margem 0,02; IC95 bootstrap de cicatrizes e blocos de 500 m, percentil dentro da dobra)",
        "retencao": "Idai reserva_origem para os ajustes K=todas (media das 5 dobras)",
        "alcance": ("transferencia espacial entre bairros DENTRO de um episodio, mesmo mapeador e mesma chuva; "
                    "dobras adjacentes; nao e episodio independente nem confirmacao"),
    },
    "limite_de_busca": "uma execucao desta emenda; nenhuma variante de dobras, raios ou orcamentos depois de ver E1",
}


def main() -> int:
    destino = OUT / "protocolo" / "emenda_1.json"
    if destino.exists():
        print("emenda 1 ja congelada")
        return 1
    e = Etapa("pa_04b_emenda1")
    conteudo = dict(EMENDA)
    conteudo["congelada_em"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conteudo["visto_antes"] = "todos os resultados do protocolo v1 (resultados/*.csv, resultado_treino.json)"
    conteudo["scripts_sha256"] = {rel(p): sha256(p) for p in sorted((ROOT / "scripts/petropolis_aprendizado").glob("pa_*.py"))}
    gravar_json(destino, conteudo)
    with open(OUT / "protocolo" / "emendas_e_incidentes.md", "a", encoding="utf-8") as f:
        f.write(f"\n2. {conteudo['congelada_em']}: **emenda 1** congelada (`protocolo/emenda_1.json`) depois de ver os "
                "resultados do v1. Acrescenta D5 (diagnostico de receptor no teste historico) e E1 (validacao "
                "cruzada por bairro com cicatrizes de 2022). Nao altera nem reinterpreta o v1. E1 e desenvolvimento.\n")
    e.saida(destino)
    e.fechar(emenda=conteudo["id"])
    print("emenda 1 congelada:", sha256(destino))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
