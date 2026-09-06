"""Ontologia temporal do REV-P.

Este modulo define o que "validade temporal" significa antes de qualquer
experimento novo. Ele nao treina modelo, nao altera contrato e nao promove
challenger; apenas torna explicitas as dimensoes que um teste temporal precisa
separar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TERMOS_TEMPORAIS: dict[str, dict[str, str]] = {
    "STATIC_SUSCEPTIBILITY": {
        "definicao": "Predisposicao espacial relativamente persistente.",
        "exemplos_revp": "HAND, TWI, slope, elevation.",
        "nao_equivale_a": "evento, chuva, data administrativa ou modelo.",
    },
    "EVENT_FORCING": {
        "definicao": "Condicao temporal que ativa ou testa a predisposicao.",
        "exemplos_revp": "rain_max_24h, rain_decay_index.",
        "nao_equivale_a": "suscetibilidade estatica nem rotulo observado.",
    },
    "EVENT_TIME": {
        "definicao": "Instante ou periodo do evento hidrologico.",
        "exemplos_revp": "data_evento ou event_date.",
        "nao_equivale_a": "data de registro ou data em que a base foi montada.",
    },
    "OBSERVATION_TIME": {
        "definicao": "Instante em que a evidencia ou rotulo foi observado.",
        "exemplos_revp": "data de vistoria, cena, survey ou atendimento.",
        "nao_equivale_a": "tempo fisico do evento quando ha atraso ou janela.",
    },
    "RECORD_TIME": {
        "definicao": "Instante administrativo de entrada do registro.",
        "exemplos_revp": "protocolo SIAC, publicacao administrativa, ingestao.",
        "nao_equivale_a": "evento hidrologico nem observacao independente.",
    },
    "TRAIN_TIME": {
        "definicao": "Intervalo de evidencia disponivel ao ajuste.",
        "exemplos_revp": "dados observaveis ate t.",
        "nao_equivale_a": "tabela completa ou estatistica global.",
    },
    "TEST_TIME": {
        "definicao": "Intervalo futuro reservado para avaliacao.",
        "exemplos_revp": "eventos posteriores a t.",
        "nao_equivale_a": "fold aleatorio, espacial ou diagnostico embaralhado.",
    },
    "LABEL_PROCESS": {
        "definicao": "Processo que produz o rotulo observado.",
        "exemplos_revp": "SEDEC, SIAC 156, Environment Agency, Copernicus EMS.",
        "nao_equivale_a": "fenomeno fisico real sem erro de observacao.",
    },
    "SAMPLING_PROCESS": {
        "definicao": "Processo que determina quais observacoes entram na amostra.",
        "exemplos_revp": "exclusao qualificada, ausencia de registro, AOI EMS.",
        "nao_equivale_a": "prevalencia real do fenomeno no territorio.",
    },
    "MODEL_TIME": {
        "definicao": "Versao e data do modelo ou artefato avaliado.",
        "exemplos_revp": "SVC-01, SVC-07, SUSC-20O, MOD-PROSP-02.",
        "nao_equivale_a": "tempo do evento ou tempo do registro.",
    },
}


TEMPO_FISICO = [
    "STATIC_SUSCEPTIBILITY",
    "EVENT_FORCING",
    "EVENT_TIME",
]

TEMPO_OBSERVACIONAL = [
    "OBSERVATION_TIME",
    "RECORD_TIME",
    "LABEL_PROCESS",
    "SAMPLING_PROCESS",
    "MODEL_TIME",
]


CHECKLIST_TEMPORAL = [
    "treino_estritamente_anterior",
    "preprocessing_fitado_apenas_no_treino",
    "features_disponiveis_no_tempo_correto",
    "sem_estatistica_global_futura",
    "sem_evento_compartilhado",
    "sem_aoi_compartilhada_indevida",
    "label_comparavel",
    "sampling_comparavel",
    "classes_avaliaveis",
    "eventos_independentes_suficientes",
    "metrica_definida",
]


IDENTIFICABILIDADE = {
    "IDENTIFICAVEL",
    "PARCIALMENTE_IDENTIFICAVEL",
    "NAO_IDENTIFICAVEL",
}

SAIDAS_TEMPORAIS = {
    "TEMPORAL_SUPPORT",
    "TEMPORAL_FAILURE",
    "TEMPORAL_INCONCLUSIVE",
    "TEMPORAL_TEST_INVALID",
}


def construir_ontologia() -> dict[str, Any]:
    """Retorna a ontologia temporal em forma serializavel."""
    return {
        "tipo": "ontologia_temporal_revp",
        "versao": "1",
        "pergunta_alvo": (
            "quando o REV-P e treinado com evidencia disponivel ate t, que "
            "evidencia permite afirmar que suas relacoes fisico-hidrologicas "
            "e sua capacidade de ordenacao permanecem transportaveis para "
            "eventos posteriores?"
        ),
        "termos": TERMOS_TEMPORAIS,
        "tempo_fisico": TEMPO_FISICO,
        "tempo_observacional": TEMPO_OBSERVACIONAL,
        "dimensoes_nao_equivalentes": [
            ["EVENT_TIME", "OBSERVATION_TIME", "RECORD_TIME"],
            ["STATIC_SUSCEPTIBILITY", "EVENT_FORCING"],
            ["LABEL_PROCESS", "SAMPLING_PROCESS"],
            ["TRAIN_TIME", "TEST_TIME", "MODEL_TIME"],
        ],
        "alvo_do_teste_temporal": {
            "treino": "dados observaveis ate t",
            "teste": "eventos posteriores a t",
            "proibido": "informacao futura direta ou indireta",
            "resposta": (
                "a relacao aprendida antes de t transporta para eventos "
                "posteriores?"
            ),
        },
        "checklist_executavel": CHECKLIST_TEMPORAL,
        "identificabilidade": sorted(IDENTIFICABILIDADE),
        "saidas_review_only": sorted(SAIDAS_TEMPORAIS),
        "nao_e": [
            "nao altera modelo servido",
            "nao altera maturidade",
            "nao promove challenger",
            "nao transforma ausencia de registro em ausencia de enchente",
            "nao declara concept drift apenas por queda de performance",
        ],
    }


def avaliar_checklist(checks: dict[str, Any]) -> dict[str, Any]:
    """Classifica a validade de um teste temporal por checklist categorico.

    Valores aceitos por item:
    - True: criterio atendido.
    - False: criterio falhou.
    - None, "limited", "unknown": criterio limitado ou nao identificavel.
    """
    ausentes = [c for c in CHECKLIST_TEMPORAL if c not in checks]
    falhas = [c for c, v in checks.items() if v is False]
    limitados = [
        c for c, v in checks.items()
        if v is None or str(v).lower() in {"limited", "unknown", "indeterminado"}
    ]

    if falhas or ausentes:
        identificabilidade = "NAO_IDENTIFICAVEL"
        valido = False
    elif limitados:
        identificabilidade = "PARCIALMENTE_IDENTIFICAVEL"
        valido = True
    else:
        identificabilidade = "IDENTIFICAVEL"
        valido = True

    return {
        "temporal_test_valid": valido,
        "temporal_identifiability": identificabilidade,
        "falhas": falhas,
        "limitados": limitados,
        "ausentes": ausentes,
        "sem_score_magico": True,
    }


def avaliar_evidencia_temporal(
    *,
    temporal_test_valid: bool,
    temporal_identifiability: str,
    performance_stability: str | None,
    ci95_inclui_acaso: bool | None = None,
    leakage_detected: bool = False,
) -> dict[str, Any]:
    """Portao temporal modo de revisao.

    A saida nunca promove modelo. Ela separa explicitamente teste invalido de
    falha temporal.
    """
    if temporal_identifiability not in IDENTIFICABILIDADE:
        raise ValueError(f"identificabilidade desconhecida: {temporal_identifiability}")

    if leakage_detected or not temporal_test_valid:
        status = "TEMPORAL_TEST_INVALID"
        motivo = "teste temporal invalido ou com vazamento/criterio ausente"
    elif temporal_identifiability == "NAO_IDENTIFICAVEL":
        status = "TEMPORAL_TEST_INVALID"
        motivo = "a pergunta temporal nao e identificavel neste desenho"
    elif temporal_identifiability == "PARCIALMENTE_IDENTIFICAVEL":
        status = "TEMPORAL_INCONCLUSIVE"
        motivo = "ha evidencias temporais, mas ha confundimento nao separado"
    elif performance_stability == "stable":
        status = "TEMPORAL_SUPPORT"
        motivo = "estabilidade de ordenacao demonstrada sob desenho identificavel"
    elif performance_stability == "unstable" and ci95_inclui_acaso is False:
        status = "TEMPORAL_FAILURE"
        motivo = "instabilidade demonstrada sob desenho identificavel"
    else:
        status = "TEMPORAL_INCONCLUSIVE"
        motivo = "evidencia temporal insuficiente para suporte ou falha"

    return {
        "status": status,
        "motivo": motivo,
        "review_only": True,
        "altera_contrato": False,
        "altera_modelo_servido": False,
    }


def gravar_ontologia(saida: Path) -> dict[str, Any]:
    """Grava `ontologia.json` e retorna o dicionario produzido."""
    ontologia = construir_ontologia()
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "ontologia.json").write_text(
        json.dumps(ontologia, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return ontologia


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    gravar_ontologia(repo / "local_runs" / "temporalidade")
    print("local_runs/temporalidade/ontologia.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
