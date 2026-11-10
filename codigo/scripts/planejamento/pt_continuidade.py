"""Diagnóstico continuidade SAR v2.1 × v3.1: o que mudou, o que reproduziu, o que fica defensável.

Nenhum treino. Lê os protocolos, as análises e o código congelado de cada versão e:
  1. tabela de fatores (entrada, máscara/suporte, populações, ordens, sementes, referência/limiar, perdas,
     normalização, orçamento, checkpoint, retenção/adaptação, regra estatística, otimizador);
  2. REAGREGA no mesmo suporte: as 8 unidades comuns (5 contextos Sen1Floods11 de retenção + 3 eventos de
     avaliação), mesmo contraste de MÉTODO (LwF legado − continuação; o "lwf" da v2.1 é o LwF não mascarado) e mesmas
     métricas presentes nas duas (IoU no limiar fixo 0,2 e AP); a v3.1 acrescenta IoU calibrado, que a v2.1 não tem;
  3. concordância por evento entre versões (sinal e Spearman das diferenças);
  4. separa mudança de tarefa/população/régua de mudança de método.
Unidade de inferência: evento (ordens e sementes já agregadas dentro do evento pelas próprias análises).
"""
from __future__ import annotations

import math

import numpy as np

from pt_comum import LINHAGEM, REPO, gravar, ler, rel, sha

AC = REPO / "local_runs" / "aprendizado-continuo"
COMUNS = ["sen1:11:2019-05-22", "sen1:2:2018-09-18", "sen1:6:2017-06-28", "sen1:9:2019-09-17", "sen1:3:2016-08-12",
          "sen1:10:2017-05-30", "sen1:7:2018-10-31", "sen1:4:2018-08-05"]


def _ic(v: np.ndarray, nivel=.95) -> list:
    from scipy import stats
    h = stats.t.ppf(.5 + nivel / 2, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return [float(v.mean() - h), float(v.mean() + h)]


def _resumo(d: dict, unidades: list) -> dict:
    from scipy import stats
    v = np.array([d[u] for u in unidades if u in d])
    w = float(stats.wilcoxon(v, method="exact").pvalue) if len(v) > 1 and np.any(v != 0) else None
    return {"n": len(v), "media": float(v.mean()), "a_favor": f"{int((v > 0).sum())}/{len(v)}", "ic95_t": _ic(v),
            "p_wilcoxon_exato": w}


def fatores(p2: dict, p3: dict) -> list[dict]:
    c2, c3 = p2["config"], p3["config"]
    n_pix = {s: sum(1 for ctx in a.values() for v in ctx.values() if v == 0) for s, a in p3["pixels_rotulados_por_anotacao"].items()}
    return [
        {"fator": "entrada e modalidade", "v2_1": f"S1 {c2['modo']}, recorte {c2['recorte']}, base {c2['base']}", "v3_1": f"S1 {c3['modo']}, recorte {c3['recorte']}, base {c3['base']}", "mudou": False},
        {"fator": "contextos (etapas)", "v2_1": f"{len(p2['contextos_canonicos'])} Sen1Floods11", "v3_1": f"{len(p3['contextos_canonicos'])} (+EMSR424, CEMS)", "mudou": True},
        {"fator": "anotacoes por contexto", "v2_1": c2["anotacoes_por_contexto"], "v3_1": c3["anotacoes_por_contexto"], "mudou": True},
        {"fator": "anotacoes sem pixel rotulado (v3.1)", "v2_1": "nao registrado", "v3_1": n_pix, "mudou": None},
        {"fator": "ordens x sementes", "v2_1": f"{len(p2['ordens'])} ordens; sementes {c2['sementes']} (bloco A: 1 semente por ordem, 10 repeticoes por evento)",
         "v3_1": f"{len(p3['ordens'])} ordens x sementes {c3['sementes']} (12 repeticoes)", "mudou": True},
        {"fator": "orcamento", "v2_1": f"{c2['passos_por_etapa']} passos/etapa, lote {c2['batch']}, lr {c2['lr']}", "v3_1": f"{c3['passos_por_etapa']} passos/etapa, lote {c3['batch']}, lr {c3['lr']} (uma etapa a mais)", "mudou": True},
        {"fator": "otimizador por etapa", "v2_1": "Adam novo (otimizador='novo'; braco replay_otimizador_herdado separado)", "v3_1": "Adam novo (mesmo nucleo continuidade_v2)", "mudou": False},
        {"fator": "LwF", "v2_1": "lwf (KL em todos os pixels)", "v3_1": f"lwf_legado (todos os pixels) e lwf_mascarado (so pixel valido); lambda {p3['lwf']['lambda']}, T {p3['lwf']['T']}", "mudou": True},
        {"fator": "checkpoint", "v2_1": f"validacao: eventos {p2['validacao']}", "v3_1": "validacao_checkpoint: chips Nigeria/Somalia (so CE)", "mudou": True},
        {"fator": "limiar/referencia", "v2_1": f"fixo {p2['limiar']}", "v3_1": f"legado {c3['limiar_legado']} e calibrado por modelo (EMSR698, Escocia)", "mudou": True},
        {"fator": "desfecho primario", "v2_1": "IoU no limiar 0,2, 8 eventos", "v3_1": "AP (1000 faixas), 9 unidades", "mudou": True},
        {"fator": "regra", "v2_1": p2["regras"]["decisao"], "v3_1": p3["primaria"]["ganho_ou_perda"] + "; familia de 4 contrastes", "mudou": True},
        {"fator": "contraste primario", "v2_1": "lwf - continuacao", "v3_1": "lwf_mascarado - continuacao (lwf_legado - continuacao so secundario)", "mudou": True},
        {"fator": "retencao/adaptacao", "v2_1": "5 reservas de retencao + 3 avaliacao", "v3_1": "6 retencao (inclui EMSR424) + 3 transferencia", "mudou": True},
        {"fator": "normalizacao", "v2_1": "etapa1", "v3_1": "etapa1 (nucleo comum)", "mudou": False},
    ]


def executar() -> dict:
    p2, p3 = ler(AC / "sar_v2_1/protocolo.json"), ler(AC / "sar_v3_1/protocolo.json")
    a2, a3 = ler(AC / "sar_v2_1/analise.json"), ler(AC / "sar_v3_1/analise.json")
    c2 = a2["familias"]["F3_metodos"]["contrastes"]["lwf - continuacao"]
    s3 = a3["secundaria"]
    v2 = {"iou_0_2": c2["iou"]["oito_eventos"]["diferencas_por_evento"], "ap": c2["ap"]["oito_eventos"]["diferencas_por_evento"]}
    v3 = {"iou_0_2": s3["lwf_legado - continuacao"]["iou_legado"]["diferencas"], "ap": s3["lwf_legado - continuacao"]["ap"]["diferencas"],
          "iou_calibrado": s3["lwf_legado - continuacao"]["iou_calibrado"]["diferencas"]}
    v3m = {"iou_0_2": s3["lwf_mascarado - continuacao"]["iou_legado"]["diferencas"], "ap": s3["lwf_mascarado - continuacao"]["ap"]["diferencas"],
           "iou_calibrado": s3["lwf_mascarado - continuacao"]["iou_calibrado"]["diferencas"]}
    from scipy import stats
    reag = {}
    for m in ("iou_0_2", "ap"):
        x = np.array([v2[m][u] for u in COMUNS])
        y = np.array([v3[m][u] for u in COMUNS])
        reag[m] = {"v2_1_lwf_menos_continuacao": _resumo(v2[m], COMUNS), "v3_1_lwf_legado_menos_continuacao": _resumo(v3[m], COMUNS),
                   "v3_1_lwf_mascarado_menos_continuacao": _resumo(v3m[m], COMUNS),
                   "mesmo_sinal_por_evento": f"{int((np.sign(x) == np.sign(y)).sum())}/8",
                   "spearman_entre_versoes": float(stats.spearmanr(x, y).statistic),
                   "razao_das_medias_v3_v2": float(y.mean() / x.mean())}
    reag["iou_calibrado_so_v3_1"] = {"lwf_legado": _resumo(v3["iou_calibrado"], COMUNS), "lwf_mascarado": _resumo(v3m["iou_calibrado"], COMUNS),
                                     "nota": "a v2.1 nao calibrou limiar; nao ha par na v2.1"}
    # níveis absolutos da continuação nas 8 unidades (a referência do contraste)
    r2 = a2["resumo_por_braco"]["continuacao"]["ap_final_por_evento"]
    r3 = a3["resumo"]["continuacao"]["ap_por_evento"]
    nivel = {"ap_continuacao_v2_1_media8": float(np.mean([r2[u] for u in COMUNS])),
             "ap_continuacao_v3_1_media8": float(np.mean([r3[u] for u in COMUNS])),
             "ap_lwf_v2_1_media8": float(np.mean([a2["resumo_por_braco"]["lwf"]["ap_final_por_evento"][u] for u in COMUNS])),
             "ap_lwf_legado_v3_1_media8": float(np.mean([a3["resumo"]["lwf_legado"]["ap_por_evento"][u] for u in COMUNS]))}
    leitura = {
        "reproduzido": ("LwF nao mascarado > continuacao no IoU do limiar fixo 0,2: v2.1 +{:.4f} (8/8) e v3.1 +{:.4f} ({}) nas mesmas 8 unidades"
                        .format(reag["iou_0_2"]["v2_1_lwf_menos_continuacao"]["media"], reag["iou_0_2"]["v3_1_lwf_legado_menos_continuacao"]["media"],
                                reag["iou_0_2"]["v3_1_lwf_legado_menos_continuacao"]["a_favor"])),
        "nao_reproduzido_em_tamanho": ("a vantagem em AP (sem limiar) caiu de +{:.4f} para +{:.4f}; a continuacao subiu {:+.4f} e o LwF {:+.4f} em AP medio"
                                       .format(reag["ap"]["v2_1_lwf_menos_continuacao"]["media"], reag["ap"]["v3_1_lwf_legado_menos_continuacao"]["media"],
                                               nivel["ap_continuacao_v3_1_media8"] - nivel["ap_continuacao_v2_1_media8"],
                                               nivel["ap_lwf_legado_v3_1_media8"] - nivel["ap_lwf_v2_1_media8"])),
        "efeito_do_limiar": ("com limiar calibrado por modelo a vantagem do LwF legado cai para +{:.4f}: parte do ganho no limiar fixo 0,2 e ponto de operacao "
                             "(o LwF desloca a probabilidade em relacao ao limiar), nao separacao".format(reag["iou_calibrado_so_v3_1"]["lwf_legado"]["media"])),
        "mudanca_de_tarefa_vs_metodo": ("o metodo comparavel (LwF nao mascarado) nao piorou; mudaram a regua primaria (IoU 0,2 -> AP), o contraste primario "
                                        "(LwF mascarado, metodo novo), a populacao (5 anotacoes por contexto em vez de 12; EMSR424 como sexta etapa), o checkpoint e o "
                                        "limiar. 'O ganho nao se repetiu' descreve a decisao da v3.1, nao uma refutacao do resultado da v2.1"),
        "defensavel": ("em desenvolvimento, com dados ja examinados e Adam novo por etapa (heranca de pesos, nao retomada de otimizador), LwF reduz a perda "
                       "da continuacao ingenua no IoU do limiar legado de forma consistente nas duas versoes; o ganho em AP e positivo nas duas mas so supera a "
                       "margem de 0,025 na v2.1; com limiar calibrado o ganho e pequeno. Nao ha resultado confirmatorio fora do desenvolvimento"),
        "experimento_adicional": ("nao executado: a diferenca concreta (limiar fixo vs calibrado; AP vs IoU) ja esta isolada nas secundarias pareadas da "
                                  "propria v3.1; uma terceira versao seria repetir para recuperar o ganho antigo")}
    out = {"tipo": "diagnostico_continuidade_v2_1_v3_1", "unidades_comuns": COMUNS, "fatores": fatores(p2, p3), "reagregacao_mesmo_suporte": reag,
           "niveis_absolutos": nivel, "leitura": leitura,
           "fontes": {rel(AC / f): sha(AC / f) for f in ("sar_v2_1/protocolo.json", "sar_v2_1/analise.json", "sar_v3_1/protocolo.json", "sar_v3_1/analise.json")}}
    gravar(LINHAGEM / "continuidade" / "diagnostico.json", out)
    return out
