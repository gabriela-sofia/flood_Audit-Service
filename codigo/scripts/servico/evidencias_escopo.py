"""Estados cientificos rastreaveis para consulta, sem promover modelo ou regiao.

Cada registro aponta a um artefato e ao hash atual. Ausencia de resultado nao
vira resultado negativo. O registro descreve estudos, nao valida a AOI pedida.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FONTES = {
    "selecao_contextual": ("local_runs/planejamento-tese/v1/contexto_v2/resultado.json", "primarias_v2", "desenvolvimento retrospectivo com selecao interna admissivel"),
    "suporte_abstencao": ("local_runs/planejamento-tese/v2/suporte_aninhado/resultado.json", "estado", "avaliacao aninhada sobre dados examinados"),
    "aquisicao_local": ("local_runs/planejamento-tese/v2/aquisicao_laco/resultado.json", "vereditos", "simulacao retrospectiva de aquisicao local"),
    "n2_spearman": ("local_runs/planejamento-tese/v2/n2/resultado.json", "contrastes", "seis seeds pareados; desfecho Spearman de fracao de agua"),
    "sar_terreno_classico": ("local_runs/planejamento-tese/v3-aoi/sar_terreno/resultado.json", "primaria_DEL", "Otsu e regra HAND a 30m; datas auditadas separadamente"),
    "sar_terreno_datas": ("local_runs/planejamento-tese/v3-aoi/sar_terreno/auditoria_datas.json", "so_data_exata_DEL", "restricao de datas posterior ao resultado; descritiva"),
    "sar_terreno_aprendido": ("local_runs/evolucao-escopo/v1/fusao/resultado.json", "contrastes", "sondas logisticas; componente retido; desenvolvimento"),
    "encoder_terreno_iou": ("local_runs/evolucao-escopo/v1/transferencia_encoder/reanalise_v2/resultado.json", "contrastes_iou", "sensibilidades posteriores de IoU vazia em suporte comum; validacao conhecida; sem teste/Bolivia"),
    "social_datado": ("local_runs/evolucao-escopo/v1/social_datado/resultado.json", "produtos_datados", "limites censitarios condicionais por produto/data"),
    "fisica_local_recuperada": ("local_runs/evolucao-escopo/v1/fisica_local/resultado.json", "delta_AUC_media_por_componente", "variante local retrospectiva sem solo/cobertura; componente retido"),
    "contexto_local_recuperado": ("local_runs/evolucao-escopo/v1/contexto_local/resultado.json", "contrastes", "selecao de seis doadores entre sete; orcamento pareado; retrospectivo"),
}


def consultar(raiz: Path = REPO):
    registros = []
    for id_, (nome, chave, natureza) in FONTES.items():
        p = raiz / nome
        r = {"evidencia_id": id_, "arquivo": nome, "ponteiro": chave, "natureza": natureza,
             "nao_e": "validacao desta AOI, promocao ou conclusao causal"}
        if not p.is_file():
            r.update(estado="sem_resultado_local", fonte_sha256=None, resultado=None)
        else:
            bruto = p.read_bytes()
            try:
                obj = json.loads(bruto)
                if chave not in obj:
                    raise ValueError("ponteiro ausente na fonte")
                r.update(estado="resultado_local_disponivel", fonte_sha256="sha256:"+hashlib.sha256(bruto).hexdigest(), resultado=obj[chave])
            except (ValueError, UnicodeError) as exc:
                r.update(estado="fonte_invalida", fonte_sha256="sha256:"+hashlib.sha256(bruto).hexdigest(), resultado=None, motivo=str(exc))
        registros.append(r)
    return {"versao": "registro_evidencias_escopo_v1", "registros": registros,
            "maturidade": "estudos_locais", "promovido": False,
            "regra": "cada resultado conserva sua tarefa, populacao, unidade e natureza; recuperar nao promove",
            "nao_e": "avaliacao territorial do lugar consultado; este registro e do estado da pesquisa"}


def verificar_afirmacao(afirmacao: dict, registro: dict):
    """Usar fonte certa e hash certo e necessario; resultado alegado deve ser identico."""
    motivos = []
    if registro.get("estado") != "resultado_local_disponivel":
        motivos.append("resultado_indisponivel")
    for chave in ("evidencia_id", "fonte_sha256", "resultado"):
        if afirmacao.get(chave) != registro.get(chave):
            motivos.append(chave+"_divergente")
    return {"aprovada": not motivos, "motivos": motivos,
            "limite": "compara afirmacao estruturada; nao julga semanticamente toda prosa livre"}
