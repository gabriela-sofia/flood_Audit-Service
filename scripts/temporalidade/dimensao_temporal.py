"""
A dimensao temporal do REV-P, derivada de artefato.

O QUE ESTE ARQUIVO FAZ, e o que ele deliberadamente NAO faz:

ele responde, para cada par REGIAO x MODELO que o REV-P conhece, a pergunta
"qual e o suporte temporal demonstravel para o uso deste modelo nesta regiao?".
A resposta sai como estado da ontologia geral de `suporte_cientifico`, com
escopo, protocolo, evidencia, limitacoes, proveniencia e consequencia.

Ele NAO tem resultado escrito por regiao. Nenhum lugar deste arquivo diz
"curitiba -> inconclusive". Os estados saem de tres coisas lidas de artefato:

  1. a CAPACIDADE temporal da regiao   (ha eventos datados, ambas as classes,
                                        negativos comparaveis, folds?)
  2. os TESTES temporais executados    (existe corte prospectivo cujo dominio
                                        de avaliacao e esta regiao, com este
                                        modelo?)
  3. a ESTABILIDADE dos cortes         (as AUCs por corte e os IC95 deles)

e sao julgados por `ontologia_temporal.avaliar_evidencia_temporal`, que ja
existia e ja separa teste invalido de falha. Aqui nao ha um segundo juri.

-------------------------------------------------------------------------
O ESCOPO E REGIAO x MODELO, E ISSO NAO E DETALHE
-------------------------------------------------------------------------

Curitiba tem um modelo PROPRIO (o diagnostico do SUSC-20O/20Q, que nunca foi
servido) e e SERVIDA por `fluvial_planicie`, que nunca a viu no ajuste. Sao
dois escopos, e o suporte temporal de um nao vale para o outro:

  - o modelo proprio de Curitiba tem holdout prospectivo executado ali;
  - o `fluvial_planicie` tem holdout prospectivo executado no POOL fluvial,
    que nao contem Curitiba.

Guardar isso por regiao -- `curitiba.temporal = X` -- faria o segundo herdar
a evidencia do primeiro, ou o contrario. Por isso a chave e o par, e por isso
o modelo transferido ganha uma linha separada para o proprio dominio de
ajuste: e la que a evidencia dele existe, e e de la que ela nao sai.

-------------------------------------------------------------------------
FORCING NAO E DRIFT DE MODELO
-------------------------------------------------------------------------

A chuva continua sendo cenario/forcing. `forcing_shift_detected` entra como
EVIDENCIA e como LIMITACAO, nunca como conclusao: mudanca no forcing nao
demonstra que a relacao aprendida deixou de valer. Essa separacao e o que
impede a leitura "Curitiba provou concept drift", que nenhum desenho deste
projeto identifica.

Uso:
    python scripts/temporalidade/dimensao_temporal.py
    python scripts/temporalidade/dimensao_temporal.py --matriz
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))
sys.path.insert(0, str(REPO / "scripts" / "temporalidade"))

import auditoria_temporal as audit  # noqa: E402
import ontologia_temporal as onto  # noqa: E402
import suporte_cientifico as sup  # noqa: E402
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402

RUNS = REPO / "local_runs"
MODELOS_SERVIDOS = RUNS / "svc-01-modelos"
TEMPORALIDADE = RUNS / "temporalidade"
OUT = RUNS / "temporalidade-integracao"
HISTORICO = OUT / "historico"

PROTOCOLO_ID = "revp_temporal_v1"
ACASO = 0.5

# Menos de tres cortes nao e serie, e coincidencia. O numero nao e novo: e o
# `MIN_CORTES_TEMPORAIS` que o SVC-07 ja usava para o portao P5, e reusa-lo
# aqui e o que impede duas definicoes de "serie temporal" no mesmo repositorio.
MINIMO_DE_CORTES = 3

# O corte prospectivo que o protocolo de challengers congelou antes de rodar
# (`protocolo_experimental.CORTES_TEMPORAIS`). Ele esta aqui como CONSTANTE
# LIDA, e nao como escolha desta frente: mudar o corte e refazer o protocolo,
# e nao ajustar um numero num script de auditoria.
ANO_DE_CORTE_DO_PROTOCOLO = 2026

# Datasets de challenger que NAO sao regiao: sao o pool de ajuste de um modelo
# de transferencia. O nome do dataset e o nome do modelo servido, e o dominio
# de avaliacao e o proprio pool -- nunca a regiao que o modelo passa a servir.
DOMINIO_DE_AJUSTE = "dominio_de_ajuste"


# --------------------------------------------------------------- evidencia

@dataclass
class Corte:
    """Um corte prospectivo: treino ate `corte`, teste depois dele."""

    rotulo: str
    auc: float | None
    ic95: list | None = None
    n_teste: int | None = None

    @property
    def ic_lo(self) -> float | None:
        if not self.ic95 or len(self.ic95) != 2:
            return None
        try:
            return float(self.ic95[0])
        except (TypeError, ValueError):
            return None

    @property
    def ic_hi(self) -> float | None:
        if not self.ic95 or len(self.ic95) != 2:
            return None
        try:
            return float(self.ic95[1])
        except (TypeError, ValueError):
            return None

    def para_dict(self) -> dict:
        return {"corte": self.rotulo, "auc": self.auc, "ic95": self.ic95,
                "n_teste": self.n_teste}


# ---------------------------------------------- ciclo de vida do teste temporal
#
# Um contador ambiguo produz conclusao errada, e este projeto ja pagou por
# isso: `n_folds_validos == 0` foi lido como "o teste temporal reprovou", e a
# primeira versao desta frente escreveu "corte executado sem fold avaliavel"
# para um caso em que NADA foi executado. Sao coisas diferentes, e o unico
# jeito de nao voltar a confundi-las e nomea-las.
#
#   DECLARED      o protocolo declara o esquema e o corte.
#   MATERIALIZED  `gerar_folds` produziu ao menos um par (treino, teste).
#   EXECUTED      ao menos um fold sobreviveu aos filtros e gerou predicao.
#   EVALUABLE     a metrica ficou definida (as duas classes no teste).
#   INTERPRETED   a evidencia foi lida pela ontologia de suporte.
#
# Cada nivel pressupoe o anterior. Recife para em DECLARED: o corte fixado foi
# o ano de 2026 e a base termina em 2025, entao nem par de indices existe.

DECLARED = "DECLARED"
MATERIALIZED = "MATERIALIZED"
EXECUTED = "EXECUTED"
EVALUABLE = "EVALUABLE"
INTERPRETED = "INTERPRETED"

CICLO_DE_VIDA = (DECLARED, MATERIALIZED, EXECUTED, EVALUABLE, INTERPRETED)

# O nome epistemico de cada parada, que e o que vai para `razao_tecnica`. Ele
# diz o que NAO aconteceu, e nunca o que o modelo fez.
PARADA = {
    "protocolo_nao_declara_teste_neste_escopo": DECLARED,
    "corte_declarado_nao_materializou_nenhum_fold": DECLARED,
    "folds_materializados_todos_invalidos": MATERIALIZED,
    "folds_executados_sem_metrica_definida": EXECUTED,
}


@dataclass
class ExecucaoTemporal:
    """Ate onde o teste chegou, e por que parou.

    Os contadores vem do artefato do harness (`n_folds_validos`,
    `n_folds_descartados`) e da materializacao real dos folds. Nenhum deles,
    sozinho, distingue "ninguem tentou" de "tentou e nao fechou" -- o par
    distingue, e e por isso que os dois viajam juntos.
    """

    estado: str = DECLARED
    protocolo_declara: bool = False
    folds_materializados: int = 0
    folds_validos: int = 0
    folds_descartados: int = 0
    metrica_definida: bool = False
    motivo_da_parada: str | None = None
    corte_declarado: str | None = None

    @property
    def chegou_a(self) -> int:
        return CICLO_DE_VIDA.index(self.estado)

    def alcancou(self, nivel: str) -> bool:
        return self.chegou_a >= CICLO_DE_VIDA.index(nivel)

    def para_dict(self) -> dict:
        return {
            "estado": self.estado,
            "protocolo_declara": self.protocolo_declara,
            "folds_materializados": self.folds_materializados,
            "folds_validos": self.folds_validos,
            "folds_descartados": self.folds_descartados,
            "metrica_definida": self.metrica_definida,
            "motivo_da_parada": self.motivo_da_parada,
            "corte_declarado": self.corte_declarado,
        }


def classificar_execucao(*, protocolo_declara: bool, folds_materializados: int,
                         folds_validos: int, folds_descartados: int,
                         metrica_definida: bool,
                         corte_declarado: str | None = None) -> ExecucaoTemporal:
    """Contadores -> estado do ciclo de vida. Sem interpretacao de desempenho.

    Repare que nenhuma AUC entra aqui. Ate `EVALUABLE` a pergunta e sobre o
    DESENHO, e nao sobre o resultado -- misturar as duas e o que fazia zero
    fold virar evidencia contra o modelo.
    """
    if not protocolo_declara:
        return ExecucaoTemporal(
            estado=DECLARED, protocolo_declara=False,
            motivo_da_parada="protocolo_nao_declara_teste_neste_escopo",
            corte_declarado=corte_declarado)
    if folds_materializados <= 0:
        return ExecucaoTemporal(
            estado=DECLARED, protocolo_declara=True,
            folds_descartados=folds_descartados,
            motivo_da_parada="corte_declarado_nao_materializou_nenhum_fold",
            corte_declarado=corte_declarado)
    if folds_validos <= 0:
        return ExecucaoTemporal(
            estado=MATERIALIZED, protocolo_declara=True,
            folds_materializados=folds_materializados,
            folds_descartados=folds_descartados,
            motivo_da_parada="folds_materializados_todos_invalidos",
            corte_declarado=corte_declarado)
    if not metrica_definida:
        return ExecucaoTemporal(
            estado=EXECUTED, protocolo_declara=True,
            folds_materializados=folds_materializados,
            folds_validos=folds_validos, folds_descartados=folds_descartados,
            motivo_da_parada="folds_executados_sem_metrica_definida",
            corte_declarado=corte_declarado)
    return ExecucaoTemporal(
        estado=EVALUABLE, protocolo_declara=True,
        folds_materializados=folds_materializados,
        folds_validos=folds_validos, folds_descartados=folds_descartados,
        metrica_definida=True, corte_declarado=corte_declarado)


def execucao_de_cortes(cortes: list, corte_declarado: str) -> ExecucaoTemporal:
    """Ciclo de vida de uma serie que ja veio pronta no artefato publicado.

    Aqui os folds nao sao materializados por este codigo -- chegam como linhas
    de resultado. O que se pode afirmar e o que a linha mostra: quantos cortes
    existem e quantos tem metrica.
    """
    com_auc = [c for c in cortes
               if c.auc is not None and not (isinstance(c.auc, float)
                                             and np.isnan(c.auc))]
    return classificar_execucao(
        protocolo_declara=bool(cortes),
        folds_materializados=len(cortes),
        folds_validos=len(cortes),
        folds_descartados=0,
        metrica_definida=bool(com_auc),
        corte_declarado=corte_declarado)


@dataclass
class EvidenciaTemporal:
    """Tudo que se sabe sobre um escopo antes de julgar.

    `execucao` diz ATE ONDE o teste chegou; `dados_permitiriam_teste` diz se a
    base teria como sustentar um. Os dois levam a `not_testable` por caminhos
    diferentes, e a diferenca e operacional: "nao ha dado local" pede
    aquisicao, "o corte declarado nao materializa fold" pede um corte novo
    congelado antes de rodar.
    """

    identificabilidade: str
    dados_permitiriam_teste: bool
    execucao: ExecucaoTemporal = field(default_factory=ExecucaoTemporal)
    cortes: list = field(default_factory=list)
    vazamento: bool = False
    forcing_shift_detected: bool | None = None
    notas: list = field(default_factory=list)

    @property
    def teste_executado(self) -> bool:
        """Derivado do ciclo de vida -- nunca uma flag escrita a mao."""
        return self.execucao.alcancou(EVALUABLE) and bool(self.cortes_com_auc())

    def cortes_com_auc(self) -> list:
        return [c for c in self.cortes if c.auc is not None and
                not (isinstance(c.auc, float) and np.isnan(c.auc))]


def estabilidade_dos_cortes(cortes: list,
                            minimo: int = MINIMO_DE_CORTES) -> tuple:
    """Le a serie de cortes e devolve (performance_stability, inclui_acaso).

    `stable` exige que TODO corte discrimine: IC95 inteiro acima do acaso. Um
    unico corte cujo intervalo encosta em 0,5 ja tira a serie de `stable`,
    porque a pergunta e transporte para eventos posteriores -- e a pior janela
    e a que responde por ela.

    Serie com menos de `minimo` cortes devolve `None`: nao ha estabilidade a
    afirmar nem a negar com um ponto so, e devolver `stable` ali seria
    transformar ausencia de serie em suporte.
    """
    validos = [c for c in cortes if c.auc is not None and
               not (isinstance(c.auc, float) and np.isnan(c.auc))]
    if len(validos) < minimo:
        return None, None
    com_ic = [c for c in validos if c.ic_lo is not None]
    if len(com_ic) < len(validos):
        # Corte sem IC nao permite dizer se a queda e ruido. Sem isso a serie
        # nao sustenta `stable` nem `unstable` com honestidade.
        return None, None
    pior = min(com_ic, key=lambda c: c.ic_lo)
    inclui_acaso = bool(pior.ic_lo <= ACASO <= (pior.ic_hi or ACASO))
    estavel = all(c.ic_lo > ACASO for c in com_ic)
    return ("stable" if estavel else "unstable"), inclui_acaso


def derivar_suporte(ev: EvidenciaTemporal) -> dict:
    """Evidencia -> estado de suporte, passando pelo juri que ja existia.

    Nao ha regra nova aqui: a classificacao e de `ontologia_temporal`, e a
    unica coisa que este bloco acrescenta e a traducao para o vocabulario
    geral e a razao tecnica -- que preserva POR QUE cada estado saiu.
    """
    estabilidade, inclui_acaso = estabilidade_dos_cortes(ev.cortes)
    n_cortes = len(ev.cortes_com_auc())

    if not ev.teste_executado or n_cortes == 0:
        # A razao vem do CICLO DE VIDA, e nao de um booleano. Cada parada tem
        # um nome que diz o que faltou -- e nenhum deles diz que o modelo
        # reprovou, porque nenhum teste chegou a produzir evidencia sobre ele.
        # A ausencia de DADO vem antes da parada do ciclo de vida, e a ordem e
        # a mensagem: em Petropolis o protocolo tambem nao declara teste, mas
        # dizer isso primeiro sugeriria que escrever um protocolo resolveria.
        # Nao resolveria -- nao ha ponto rotulado. A restricao que liga e a
        # que o leitor precisa ver.
        if not ev.dados_permitiriam_teste:
            razao = "sem_dado_para_montar_teste"
        elif ev.execucao.motivo_da_parada:
            razao = ev.execucao.motivo_da_parada
        else:
            razao = "dado_existe_mas_nenhum_teste_executado_neste_escopo"
        teste_valido = False
    elif ev.vazamento:
        razao = "vazamento_detectado_no_desenho"
        teste_valido = False
    else:
        teste_valido = ev.identificabilidade != "NAO_IDENTIFICAVEL"
        if not teste_valido:
            razao = "desenho_nao_identifica_a_pergunta_temporal"
        elif n_cortes < MINIMO_DE_CORTES:
            razao = f"serie_curta_{n_cortes}_corte(s)_minimo_{MINIMO_DE_CORTES}"
        elif estabilidade is None:
            razao = "cortes_sem_intervalo_para_separar_queda_de_ruido"
        elif estabilidade == "stable":
            razao = "todos_os_cortes_discriminam_acima_do_acaso"
        elif inclui_acaso:
            razao = "algum_corte_nao_se_separa_do_acaso"
        else:
            razao = "instabilidade_precisa_sob_desenho_identificavel"

    veredito = onto.avaliar_evidencia_temporal(
        temporal_test_valid=teste_valido,
        temporal_identifiability=ev.identificabilidade,
        performance_stability=estabilidade,
        ci95_inclui_acaso=inclui_acaso,
        leakage_detected=ev.vazamento,
    )
    return {
        "support": sup.do_vocabulario_temporal(veredito["status"]),
        "status_temporal_legado": veredito["status"],
        "razao_tecnica": razao,
        "motivo_do_juri": veredito["motivo"],
        "performance_stability": estabilidade,
        "ci95_inclui_acaso": inclui_acaso,
        "n_cortes_avaliaveis": n_cortes,
    }


# ------------------------------------------------------------- leitura de artefato

def _proveniencia(caminho: Path, papel: str) -> dict:
    return {
        "artefato": audit.rel(caminho),
        "existe": caminho.exists(),
        "sha256": sha256_arquivo(caminho),
        "papel": papel,
    }


def capacidade() -> pd.DataFrame:
    """Capacidade temporal por fonte, da tabela unica admitida.

    Usa `admitidos` em vez do pool fluvial porque o pool exclui Recife: uma
    matriz que nao consegue nem PERGUNTAR pela regiao com modelo proprio
    responderia por omissao, que e o modo de falha desta frente.
    """
    adm = audit.read_csv(audit.ADMITIDOS)
    return audit.capacidade_por_regiao(pd.DataFrame(), adm)


def identificabilidade_da_base(d: pd.DataFrame, source_key: str = "",
                               folds_sustentaveis: int = 0) -> tuple:
    """Identificabilidade temporal de uma base qualquer, pelo classificador oficial.

    Existe para que quem monta cortes fora desta frente -- hoje o portao P5 do
    SVC-07 -- pergunte pela MESMA regra, e nao invente a propria. A alternativa
    seria o reajuste chamar de "estavel" uma base que a auditoria classifica
    como nao identificavel, que e como se tem duas definicoes de temporalidade
    no mesmo repositorio sem ninguem perceber.
    """
    if d is None or len(d) == 0 or "classe" not in d.columns:
        return "NAO_IDENTIFICAVEL", {}
    labels = pd.to_numeric(d["classe"], errors="coerce")
    datas = audit.parse_date(d["data_evento"]) if "data_evento" in d.columns \
        else pd.Series(pd.NaT, index=d.index)
    neg = d[labels == 0]
    comparaveis = (int(neg["nivel_negativo"].map(audit.is_negative_comparavel).sum())
                   if "nivel_negativo" in d.columns and not neg.empty else 0)
    por_data = (d.dropna(subset=[]).assign(_d=datas.dt.date, _y=labels)
                .dropna(subset=["_d"]).groupby("_d")["_y"].nunique()
                if datas.notna().any() else pd.Series(dtype=int))
    stats = {
        "source_key": source_key,
        "positivos": int((labels == 1).sum()),
        "negativos": int((labels == 0).sum()),
        "negativos_comparaveis": comparaveis,
        "n_dates": int(datas.dt.date.dropna().nunique()),
        "eventos_independentes": (int(d["grupo_cv"].nunique())
                                  if "grupo_cv" in d.columns else 0),
        "datas_com_ambas_classes": int((por_data >= 2).sum()) if len(por_data) else 0,
        "folds_sustentaveis": int(folds_sustentaveis),
        "anos_cobertos": int(datas.dt.year.dropna().nunique()),
    }
    return audit.classify_temporal_identifiability(stats), stats


def _capacidade_da_fonte(cap: pd.DataFrame, chave: str) -> dict:
    if cap.empty:
        return {}
    linha = cap[cap["source_key"].astype(str) == chave]
    return linha.iloc[0].to_dict() if not linha.empty else {}


def materializacao_do_corte(dataset: str, ano_corte: int) -> dict:
    """Quantos folds o corte declarado MATERIALIZA nesta base -- sem ajustar nada.

    Isto e geometria de indices, e nao experimento: `gerar_folds` so parte a
    base em treino e teste. Nenhum modelo e ajustado, nenhuma metrica sai
    daqui, e por isso a funcao pode rodar sobre um corte que o protocolo nao
    congelou -- e o que permite dizer "o corte de 2026 nao materializa fold e
    o de 2024 materializaria" sem produzir um resultado novo que ninguem
    congelou antes.

    Devolve `{}` quando a base do harness nao esta disponivel: ausencia de
    base nao pode virar "zero folds", que e uma afirmacao sobre o desenho.
    """
    try:
        sys.path.insert(0, str(REPO / "scripts" / "challengers"))
        import matriz_validacao as mv
    except Exception:
        return {}
    if dataset not in getattr(mv, "DATASETS", {}):
        return {}
    try:
        c = mv.carregar(dataset)
        folds = mv.gerar_folds(c, mv.HOLDOUT, ano_corte=ano_corte)
    except Exception as e:  # base ausente ou ilegivel
        return {"erro": str(e)[:120]}

    materializados, com_duas_classes = 0, 0
    for _rotulo, tr, te in folds:
        materializados += 1
        if (len(np.unique(c.y[tr])) >= 2) and (len(np.unique(c.y[te])) >= 2):
            com_duas_classes += 1
    return {
        "ano_corte": int(ano_corte),
        "folds_materializados": materializados,
        "folds_com_duas_classes": com_duas_classes,
        "anos_na_base": sorted(int(a) for a in
                               c.datas.dt.year.dropna().unique()),
    }


def cortes_dos_experimentos(dataset: str, modelo: str) -> tuple:
    """Cortes prospectivos do harness, e ATE ONDE o teste chegou.

    Devolve `(cortes, execucao)`. O segundo e o que impede a leitura errada
    que este projeto ja fez duas vezes: em Recife existe linha de
    `holdout_temporal` em `metrics.csv` com `n_folds_validos == 0` E
    `n_folds_descartados == 0`. Ler isso como "o teste reprovou" e falso, e
    ler como "executou e nao deu fold avaliavel" tambem e -- o corte declarado
    e o ano de 2026, a base de Recife termina em 2025, e `gerar_folds` nao
    produz nem par de indices. O teste para em DECLARED.
    """
    folds = audit.read_csv(audit.EXPERIMENTOS / "folds.csv")
    metrics = audit.read_csv(audit.EXPERIMENTOS / "metrics.csv")

    def _filtrar(df):
        if df.empty or "dataset" not in df.columns:
            return df
        return df[(df["dataset"].astype(str) == dataset)
                  & (df["modelo"].astype(str) == modelo)
                  & (df["esquema"].astype(str) == "holdout_temporal")]

    m = _filtrar(metrics)
    declarado = not m.empty

    cortes = []
    for _, r in _filtrar(folds).iterrows():
        cortes.append(Corte(rotulo=str(r.get("fold")),
                            auc=audit._float_or_none(r.get("auc")),
                            ic95=None,
                            n_teste=audit._int_or_none(r.get("n_test"))))
    # O IC95 do esquema esta em `metrics.csv` -- e por conjunto/modelo/esquema,
    # e nao por fold. Ele entra so quando ha um fold, que e o caso do holdout.
    if len(cortes) == 1 and declarado:
        lo = audit._float_or_none(m.iloc[0].get("auc_ic95_lo"))
        hi = audit._float_or_none(m.iloc[0].get("auc_ic95_hi"))
        if lo is not None and hi is not None:
            cortes[0].ic95 = [lo, hi]

    validos = audit._int_or_none(m.iloc[0].get("n_folds_validos")) if declarado else 0
    descartados = (audit._int_or_none(m.iloc[0].get("n_folds_descartados"))
                   if declarado else 0)
    metrica = bool(declarado and audit._float_or_none(m.iloc[0].get("auc")) is not None)

    ano = ANO_DE_CORTE_DO_PROTOCOLO
    mat = materializacao_do_corte(dataset, ano) if declarado else {}
    # Sem a base para conferir, a materializacao e DEDUZIDA dos contadores do
    # artefato -- e a deducao e conservadora: `validos + descartados` e o piso
    # do que foi materializado, nunca uma afirmacao de que nada existiu.
    materializados = mat.get("folds_materializados",
                             (validos or 0) + (descartados or 0))

    execucao = classificar_execucao(
        protocolo_declara=declarado,
        folds_materializados=materializados,
        folds_validos=validos or 0,
        folds_descartados=descartados or 0,
        metrica_definida=metrica,
        corte_declarado=f"holdout_temporal ano_corte={ano}")
    if mat:
        execucao.corte_declarado = (
            f"holdout_temporal ano_corte={ano}; anos na base: "
            f"{mat.get('anos_na_base', [])[:1]}..{mat.get('anos_na_base', [])[-1:]}"
            if mat.get("anos_na_base") else execucao.corte_declarado)
    return cortes, execucao


def cortes_do_mod_prosp(conjunto: str, variante: str) -> list:
    folds = audit.read_csv(audit.MOD_PROSP / "folds.csv")
    if folds.empty:
        return []
    f = folds[(folds.get("conjunto").astype(str) == conjunto)
              & (folds.get("variante").astype(str) == variante)]
    cortes = []
    for _, r in f.iterrows():
        ic = r.get("ic95")
        if isinstance(ic, str):
            try:
                ic = json.loads(ic)
            except json.JSONDecodeError:
                ic = None
        cortes.append(Corte(rotulo=str(r.get("corte")),
                            auc=audit._float_or_none(r.get("auc_prospectivo")),
                            ic95=ic if isinstance(ic, list) else None,
                            n_teste=audit._int_or_none(r.get("n_teste"))))
    return cortes


def cortes_do_modelo_local_curitiba() -> list:
    """Walk-forward do SUSC-20Q mais o holdout do SUSC-20O.

    Nenhum deles carrega IC95 no artefato publicado. Isso nao e detalhe: sem
    IC nao da para separar queda de ruido, e a serie nao pode sustentar nem
    `stable` nem `unstable` -- que e uma das razoes de o estado ficar
    inconclusivo, e nao "falhou".
    """
    wf = audit.read_csv(audit.CURITIBA_PUBLIC / "results"
                        / "v20q_walk_forward_cutoffs.csv")
    cortes = []
    for _, r in wf.iterrows():
        cortes.append(Corte(rotulo=f"teste_{r.get('test_year')}",
                            auc=audit._float_or_none(r.get("holdout_auc")),
                            ic95=None,
                            n_teste=audit._int_or_none(r.get("n_test"))))
    return cortes


def forcing_shift_de_curitiba() -> bool | None:
    """Le o drift de chuva do artefato -- e para aqui.

    Devolve se o forcing mudou. NAO devolve, e nao pode devolver, se o modelo
    falhou: sao perguntas diferentes, e juntar as duas e exatamente o erro que
    produziu a frase "Curitiba provou concept drift".
    """
    drift = audit.read_csv(audit.EXPERIMENTOS / "feature_drift.csv")
    if drift.empty or "feature" not in drift.columns:
        return None
    chuva = drift[drift["feature"].astype(str).str.contains("rain", na=False)]
    if chuva.empty:
        return None
    coluna = next((c for c in ("psi", "PSI", "smd") if c in chuva.columns), None)
    if coluna is None:
        return None
    return bool(pd.to_numeric(chuva[coluna], errors="coerce").abs().max() >= 0.25)


# ------------------------------------------------------------- montagem da matriz

def _janela(cap: dict) -> str | None:
    ini, fim = cap.get("inicio"), cap.get("fim")
    for v in (ini, fim):
        if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) in {
                "nan", "NaT", ""}:
            return None
    return f"{ini}..{fim}"


def _janela_expansiva(inicio, cortes: list) -> str | None:
    """Janela de treino de uma serie EXPANSIVA, que nao e um intervalo fixo.

    Cada corte treina desde o inicio ate ele proprio, entao publicar so o
    primeiro corte diria que o modelo foi ajustado uma vez, com dois anos de
    dado -- quando foram oito ajustes, cada um com mais historico que o
    anterior. A forma composta e a unica leitura correta da serie.
    """
    if not cortes:
        return None
    fim = str(cortes[-1].rotulo).replace("teste_", "")
    base = _janela({"inicio": inicio, "fim": fim})
    if base is None:
        return None
    return f"{base} (janela expansiva, {len(cortes)} cortes)"


def janela_de_treino_do_harness(dataset: str, modelo: str) -> str | None:
    """Janela de TREINO dos folds do holdout, quando o artefato a carrega.

    Ela nao e a janela de dados da regiao, e a diferenca so parece detalhe ate
    o modelo ser de transferencia: `fluvial_planicie` treina no pool e serve
    Curitiba, entao publicar a janela de Curitiba como `training_window` diria
    que ele viu Curitiba no ajuste, que e falso.
    """
    folds = audit.read_csv(audit.EXPERIMENTOS / "folds.csv")
    if folds.empty or "train_period_ini" not in folds.columns:
        return None
    f = folds[(folds["dataset"].astype(str) == dataset)
              & (folds["modelo"].astype(str) == modelo)
              & (folds["esquema"].astype(str) == "holdout_temporal")]
    if f.empty:
        return None
    ini, fim = audit._min_col(f, "train_period_ini"), audit._max_col(f, "train_period_fim")
    return _janela({"inicio": ini, "fim": fim})


def _evidencia_de_cortes(cortes: list, extras: list | None = None) -> list:
    ev = [c.para_dict() for c in cortes]
    return ev + list(extras or [])


def _limitacoes_da_capacidade(cap: dict) -> list:
    lim = []
    if not cap:
        return ["nenhuma linha de capacidade temporal para esta fonte"]
    if int(cap.get("positivos") or 0) == 0 or int(cap.get("negativos") or 0) == 0:
        lim.append("a fonte nao tem as duas classes rotuladas")
    if int(cap.get("negativos_comparaveis") or 0) <= 0:
        lim.append("nenhum negativo com qualidade temporal comparavel")
    elif int(cap.get("negativos_comparaveis") or 0) < int(cap.get("negativos") or 0):
        lim.append(
            f"{cap.get('negativos_comparaveis')} de {cap.get('negativos')} "
            "negativos tem qualidade temporal comparavel")
    if int(cap.get("datas_com_ambas_classes") or 0) <= 0:
        lim.append("nenhuma data tem as duas classes de modo avaliavel")
    if int(cap.get("folds_sustentaveis") or 0) <= 0:
        lim.append("a fonte nao sustenta fold prospectivo segundo a viabilidade")
    return lim


def _avaliacao(regiao: str, modelo: str, ev: EvidenciaTemporal, cap: dict,
               evidencia_extra: list, limitacoes: list, prov: list,
               evidence_version: str, dominio: str, quando: str,
               training_window: str | None = None) -> sup.AvaliacaoDeSuporte:
    d = derivar_suporte(ev)
    escopo = sup.EscopoDeSuporte(
        region_id=regiao,
        model_id=modelo,
        protocol_id=PROTOCOLO_ID,
        evidence_version=evidence_version,
        training_window=training_window,
        evaluation_window=(ev.cortes[-1].rotulo if ev.cortes else None),
        evaluation_domain=dominio,
        region_data_window=_janela(cap),
        # `forcing_source` e a fonte de CHUVA, e nao o nome do dataset -- os
        # dois ja se confundiram uma vez neste projeto, e a confusao esconde
        # que Recife mistura fontes de chuva. `observation_process` fica de
        # fora quando nao ha descricao propria do processo de rotulo: a
        # justificativa da capacidade e outra coisa, e ela entra na evidencia
        # com o nome dela.
        forcing_source=audit._rain_source_for_dataset(
            str(cap.get("source_key") or cap.get("dataset") or "")),
        dataset_version=cap.get("dataset"),
        observation_process=cap.get("observation_process"),
    )
    todas = list(limitacoes)
    if ev.forcing_shift_detected:
        todas.append(
            "shift de forcing (chuva) detectado no periodo: e mudanca de "
            "cenario, e nao demonstracao de que a relacao aprendida deixou "
            "de valer")
    todas.extend(ev.notas)
    return sup.AvaliacaoDeSuporte(
        dimension=sup.TEMPORAL,
        support=d["support"],
        scope=escopo,
        razao_tecnica=d["razao_tecnica"],
        evidence=_evidencia_de_cortes(ev.cortes, evidencia_extra + [
            {"capacidade_justificativa": cap.get("justificativa")},
            {"identificabilidade": ev.identificabilidade},
            {"teste_executado": ev.teste_executado},
            {"dados_permitiriam_teste": ev.dados_permitiriam_teste},
            {"performance_stability": d["performance_stability"]},
            {"ci95_inclui_acaso": d["ci95_inclui_acaso"]},
            {"status_temporal_legado": d["status_temporal_legado"]},
            {"motivo_do_juri": d["motivo_do_juri"]},
            {"forcing_shift_detected": ev.forcing_shift_detected},
        ]),
        limitations=todas,
        provenance=prov,
        avaliado_em=quando,
    )


def _modelos_servidos() -> dict:
    modelos = {}
    if not MODELOS_SERVIDOS.exists():
        return modelos
    for p in sorted(MODELOS_SERVIDOS.glob("*.json")):
        if p.name == "indice.json":
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        modelos[m["nome"]] = m
    return modelos


def _versao_da_evidencia(prov_base: list) -> str:
    """Hash dos artefatos de entrada, sem o prefixo `sha256:`.

    O prefixo sai porque esta string vira NOME DE ARQUIVO no historico, e no
    Windows um `:` no nome cria um alternate data stream em vez de um arquivo
    -- o snapshot ficava com zero byte e ninguem percebia.
    """
    return sha256_json([{k: p.get(k) for k in ("artefato", "sha256")}
                        for p in prov_base]).split(":")[-1][:16]


def construir_matriz() -> sup.MatrizDeSuporte:
    """A matriz inteira, derivada. Nenhum estado por regiao escrito a mao."""
    quando = agora_utc()
    cap = capacidade()
    servidos = _modelos_servidos()

    prov_base = [
        _proveniencia(audit.ADMITIDOS, "capacidade temporal por fonte"),
        _proveniencia(audit.EXPERIMENTOS / "folds.csv",
                      "cortes prospectivos do harness de challengers"),
        _proveniencia(audit.EXPERIMENTOS / "metrics.csv",
                      "IC95 por conjunto/modelo/esquema"),
        _proveniencia(audit.MOD_PROSP / "folds.csv",
                      "janela expansiva prospectiva do piloto ingles"),
        _proveniencia(audit.CURITIBA_PUBLIC / "results"
                      / "v20q_walk_forward_cutoffs.csv",
                      "walk-forward do modelo local de Curitiba"),
        _proveniencia(MODELOS_SERVIDOS / "indice.json",
                      "modelos servidos e regioes servidas"),
    ]
    evidence_version = _versao_da_evidencia(prov_base)
    matriz = sup.MatrizDeSuporte(gerado_em=quando)

    forcing_curitiba = forcing_shift_de_curitiba()

    # ---------------------------------------------------- modelos servidos
    for nome, m in sorted(servidos.items()):
        if not m.get("servivel"):
            continue
        do_ajuste = {f.lower() for f in (m.get("fontes_do_ajuste") or {})}
        for regiao in m.get("regioes_servidas", []):
            regiao = str(regiao).lower()
            c = _capacidade_da_fonte(cap, regiao)
            proprio = regiao in do_ajuste

            if proprio:
                # Modelo proprio: o dataset do harness leva o nome do modelo e
                # o dominio de avaliacao E a regiao.
                cortes, execucao = cortes_dos_experimentos(nome, "firth_servido")
                dominio = regiao
                extra = [{"dataset_do_harness": nome},
                         {"modelo_avaliado": "firth_servido"},
                         {"execucao": execucao.para_dict()}]
                notas = []
                if execucao.motivo_da_parada:
                    notas.append(
                        f"o teste temporal parou em {execucao.estado}: "
                        f"{execucao.motivo_da_parada}. Nao ha AUC baixa aqui "
                        "-- ha teste que nao chegou a produzir evidencia "
                        "sobre o modelo")
                mat = materializacao_do_corte(nome, ANO_DE_CORTE_DO_PROTOCOLO)
                if mat.get("folds_materializados") == 0 and mat.get("anos_na_base"):
                    notas.append(
                        f"o corte declarado e o ano {ANO_DE_CORTE_DO_PROTOCOLO}"
                        f" e a base termina em {mat['anos_na_base'][-1]}: "
                        "nenhum par (treino, teste) chega a existir. Um corte "
                        "avaliavel exigiria protocolo novo, congelado antes de "
                        "rodar -- esta auditoria nao produz resultado com corte "
                        "que ninguem congelou")
                    extra.append({"materializacao_do_corte_declarado": mat})
            else:
                # Transferencia: os cortes do modelo existem no POOL de ajuste,
                # e nao aqui. Copia-los seria herdar suporte nao medido.
                cortes = []
                execucao = classificar_execucao(
                    protocolo_declara=False, folds_materializados=0,
                    folds_validos=0, folds_descartados=0,
                    metrica_definida=False)
                dominio = regiao
                extra = [{"modelo_de_transferencia": True},
                         {"dominio_do_teste_do_modelo": f"{DOMINIO_DE_AJUSTE}:{nome}"},
                         {"por_que_sem_corte": (
                             "os cortes prospectivos deste modelo foram "
                             "avaliados no pool de ajuste, que nao contem "
                             "esta regiao")}]
                notas = [
                    "o suporte temporal medido no dominio de ajuste deste "
                    "modelo nao transfere para esta regiao: e escopo diferente"]

            ev = EvidenciaTemporal(
                identificabilidade=str(c.get("temporal_identifiability")
                                       or "NAO_IDENTIFICAVEL"),
                dados_permitiriam_teste=bool(
                    c and int(c.get("positivos") or 0) > 0
                    and int(c.get("negativos") or 0) > 0),
                execucao=execucao,
                cortes=cortes,
                forcing_shift_detected=(forcing_curitiba if regiao == "curitiba"
                                        else None),
                notas=notas,
            )
            matriz.adicionar(_avaliacao(
                regiao, nome, ev, c, extra, _limitacoes_da_capacidade(c),
                prov_base, evidence_version, dominio, quando,
                training_window=(janela_de_treino_do_harness(
                    nome, "firth_servido") if proprio else None)))

        # ------------------------- o dominio de ajuste do modelo transferido
        #
        # A linha existe para que a evidencia do modelo tenha onde morar sem
        # contaminar a regiao servida. Quem quiser saber "o fluvial_planicie
        # transporta no tempo?" tem resposta; quem perguntar "ele transporta
        # no tempo EM CURITIBA?" recebe a outra linha, e elas nao se misturam.
        servidas = {str(r).lower() for r in m.get("regioes_servidas", [])}
        if servidas - do_ajuste:
            cortes, execucao = cortes_dos_experimentos(nome, "firth_servido")
            fontes = ",".join(sorted(do_ajuste)) or "desconhecido"
            c_pool = {"dataset": f"pool:{fontes}",
                      "justificativa": "pool de ajuste, nao regiao"}
            ev = EvidenciaTemporal(
                identificabilidade=("PARCIALMENTE_IDENTIFICAVEL" if cortes
                                    else "NAO_IDENTIFICAVEL"),
                dados_permitiriam_teste=True,
                execucao=execucao,
                cortes=cortes,
                notas=["dominio de ajuste do modelo; nao e uma regiao servida"],
            )
            matriz.adicionar(_avaliacao(
                f"{DOMINIO_DE_AJUSTE}:{nome}", nome, ev, c_pool,
                [{"fontes_do_ajuste": sorted(do_ajuste)},
                 {"execucao": execucao.para_dict()}],
                ["o pool mistura fontes; o confundimento fonte x periodo nao "
                 "foi decomposto"],
                prov_base, evidence_version, f"{DOMINIO_DE_AJUSTE}:{nome}",
                quando,
                training_window=janela_de_treino_do_harness(
                    nome, "firth_servido")))

    # ------------------------------- modelos avaliados que nao sao servidos
    #
    # Eles entram porque a pergunta cientifica existe para eles, e sair de
    # fora faria a matriz responder so por aquilo que o produto ja serve --
    # perdendo justamente o controle ingles e o modelo proprio de Curitiba.
    for conjunto, variante, rotulo in (("TERRENO", "bloco", "TERRENO/bloco"),
                                       ("COMPLETO", "bloco", "COMPLETO/bloco")):
        cortes = cortes_do_mod_prosp(conjunto, variante)
        if not cortes:
            continue
        c_uk = _capacidade_da_fonte(cap, "uk")
        ev = EvidenciaTemporal(
            identificabilidade=str(c_uk.get("temporal_identifiability")
                                   or "NAO_IDENTIFICAVEL"),
            dados_permitiriam_teste=True,
            execucao=execucao_de_cortes(cortes, f"{conjunto}/{variante}"),
            cortes=cortes,
            notas=["piloto de exclusao qualificada medido num unico pais"],
        )
        matriz.adicionar(_avaliacao(
            "uk", f"MOD-PROSP-02:{rotulo}", ev, c_uk,
            [{"conjunto": conjunto}, {"variante": variante}],
            _limitacoes_da_capacidade(c_uk)
            + ["nao autoriza uso operacional: e piloto"],
            prov_base, evidence_version, "uk", quando,
            training_window=_janela_expansiva(c_uk.get("inicio"), cortes)))

    cortes_cur = cortes_do_modelo_local_curitiba()
    if cortes_cur:
        c_cur = _capacidade_da_fonte(cap, "curitiba")
        ev = EvidenciaTemporal(
            identificabilidade=str(c_cur.get("temporal_identifiability")
                                   or "NAO_IDENTIFICAVEL"),
            dados_permitiriam_teste=True,
            execucao=execucao_de_cortes(cortes_cur, "SUSC-20Q walk-forward"),
            cortes=cortes_cur,
            forcing_shift_detected=forcing_curitiba,
            notas=["modelo diagnostico local, nunca servido pelo contrato"],
        )
        matriz.adicionar(_avaliacao(
            "curitiba", "SUSC-20O/20Q:logistica_l2_diagnostica", ev, c_cur,
            [{"artefatos": ["v20o_all_reports.json",
                            "v20q_walk_forward_cutoffs.csv"]}],
            _limitacoes_da_capacidade(c_cur)
            + ["os cortes publicados nao carregam IC95: queda e ruido nao "
               "estao separados",
               "processo administrativo e geocodificacao entram no rotulo"],
            prov_base, evidence_version, "curitiba", quando,
            training_window=_janela_expansiva(c_cur.get("inicio"), cortes_cur)))

    return matriz


# ---------------------------------------------------------- impacto no contrato

def analisar_impacto_no_contrato(matriz: sup.MatrizDeSuporte,
                                 shadow_df=None) -> dict:
    """Responde as sete perguntas do versionamento, sem esconder nenhuma.

    A pergunta que importa e a 5: alguma resposta que a v1 emite hoje deixaria
    de ser emitida se o suporte temporal virasse portao? Se sim, ativar G6 e
    mudanca semantica -- e o documento de versionamento diz, com todas as
    letras, que mudanca de comportamento de recusa exige `v2`.
    """
    from svc02_contrato_inferencia import (VERSAO_CONTRATO, carregar_modelos,
                                           maturidade_da_regiao,
                                           regioes_com_referencia_local)

    servidos = carregar_modelos()
    com_ref = regioes_com_referencia_local()
    afetadas = []
    for nome, m in sorted(servidos.items()):
        if not m.get("servivel"):
            continue
        for regiao in m.get("regioes_servidas", []):
            regiao = str(regiao).lower()
            aval = matriz.consultar(regiao, nome)
            estado = (aval or {}).get("support")
            emitiria_hoje = True   # v1 emite se G1-G5 fecharem
            emitiria_com_g6 = bool(estado and sup.suporte_demonstrado(estado))
            afetadas.append({
                "regiao": regiao,
                "modelo": nome,
                "maturidade": maturidade_da_regiao(regiao, m, com_ref),
                "temporal_support": estado,
                "razao_tecnica": (aval or {}).get("razao_tecnica"),
                "v1_pode_emitir_escore": emitiria_hoje,
                "com_g6_ativo_emitiria": emitiria_com_g6,
                "g6_mudaria_a_resposta": emitiria_hoje and not emitiria_com_g6,
            })

    muda = [a for a in afetadas if a["g6_mudaria_a_resposta"]]

    # Quando o shadow rodou, ele MANDA. A tabela acima e projecao por escopo;
    # o shadow e resposta real do contrato sobre requisicao real, e a pergunta
    # do versionamento e sobre resposta, nao sobre projecao.
    casos_medidos = None
    if shadow_df is not None and len(shadow_df):
        casos_medidos = {
            "casos_avaliados": int(len(shadow_df)),
            "casos_que_mudariam": int(shadow_df["changed"].sum()),
            "regioes_que_mudariam": sorted(
                {str(r) for r in shadow_df.loc[shadow_df["changed"], "region"]}),
        }
        v2 = casos_medidos["casos_que_mudariam"] > 0
    else:
        v2 = bool(muda)

    from svc02_contrato_inferencia import GATE_TEMPORAL_ATIVO, PORTOES_PUBLICOS
    return {
        "public_gates": list(PORTOES_PUBLICOS),
        "g6_internal_exists": True,
        "g6_public": False,
        "g6_active": bool(GATE_TEMPORAL_ATIVO),
        "v1_decision_changes": False,
        "affected_cases": casos_medidos,
        "tipo": "contract_impact_dimensao_temporal",
        "versao": "1",
        "gerado_em": matriz.gerado_em,
        "current_contract": VERSAO_CONTRATO,
        "temporal_dimension_implemented": True,
        "temporal_dimension_modo": "review_only",
        "1_adiciona_somente_informacao": True,
        "2_altera_decisao": False,
        "3_altera_status": False,
        "4_altera_primeira_falha_reportada": False,
        "5_muda_resposta_v1_anteriormente_correta": False,
        "6_exige_g6": v2,
        "7_exige_v2": v2,
        "changes_v1_decisions": False,
        "g6_candidate": {
            "nome": "suporte_temporal",
            "posicao_estudada": "apos G5, antes da emissao do escore",
            "entrada": "avaliacao de suporte temporal do escopo regiao x modelo",
            "nao_testa": "AUC temporal contra limiar",
            "testa": ("ha suporte temporal suficiente, segundo a politica "
                      "cientifica aplicavel a este modelo e escopo?"),
            "implementado": True,
            "ativo": False,
            "constante": "GATE_TEMPORAL_ATIVO em svc02_contrato_inferencia.py",
        },
        "arquiteturas_estudadas": {
            "A_g6_na_escada": {
                "descricao": "G6 = suporte_temporal, sexto portao da sequencia",
                "custo": ("muda a sequencia externa que a v1 promete e o "
                          "conjunto de `gate_que_nao_fechou`; um cliente da v1 "
                          "passa a receber nome de portao que nao existia"),
                "beneficio": "a temporalidade ganha o mesmo peso de G1-G5",
            },
            "B_composicao_antes_da_emissao": {
                "descricao": ("suporte cientifico composto avaliado depois de "
                              "G5, sem alterar a sequencia externa G1-G5"),
                "custo": ("a recusa nao tem nome de portao na escada, e o "
                          "cliente precisa ler um campo novo para saber por que"),
                "beneficio": ("preserva a escada da v1 e cabe no precedente de "
                              "`modelo_dentro_dos_criterios`, que ja recusa "
                              "depois de G5"),
            },
            "escolhida_para_a_v2": "A",
            "por_que": (
                "a arquitetura B ja existe no codigo como "
                "`modelo_dentro_dos_criterios` e e justamente o caso em que a "
                "recusa fica sem nome na escada -- o frontend precisou de uma "
                "lista `PORTOES_APOS_A_ESCADA` so para desenha-la. Repetir "
                "esse padrao para uma condicao com peso estrutural esconderia "
                "a dimensao no exato campo que o cliente nao le. Se a "
                "temporalidade tem peso comparavel a G1-G5, ela entra na "
                "escada -- e entrar na escada e `v2`, o que e a razao de a "
                "ativacao nao acontecer neste commit."),
        },
        "regioes_afetadas": afetadas,
        "regioes_que_g6_recusaria": (casos_medidos["regioes_que_mudariam"]
                                     if casos_medidos
                                     else [a["regiao"] for a in muda]),
        "derivado_de": ("shadow sobre requisicoes reais" if casos_medidos
                        else "projecao por escopo (shadow nao executado)"),
        "v2_required": v2,
        "veredito": "V2_REQUIRED" if v2 else "SEM_MUDANCA_DE_VERSAO",
        "reason": (
            "com a evidencia atual, ativar o portao temporal faria a v1 deixar "
            "de emitir escore para " + ", ".join(
                f"{a['regiao']} ({a['temporal_support']})" for a in muda)
            + ". Mudar comportamento de recusa e mudanca semantica, e "
              "docs/CONTRATO_VERSIONAMENTO.md exige `v2` para isso."
        ) if v2 else (
            "nenhuma resposta que a v1 emite hoje mudaria com o portao "
            "temporal ativo"),
        "enquanto_v1": (
            "a dimensao temporal viaja como campo novo e opcional "
            "(`suporte_cientifico`), que um cliente da v1 pode ignorar sem "
            "ler nada errado. Nenhuma decisao da v1 depende dela."),
        "nao_e": ("nao ativa portao, nao muda maturidade, nao promove modelo e "
                  "nao altera o Firth servido"),
    }


# -------------------------------------------------------------------- saida

def gravar(matriz: sup.MatrizDeSuporte, impacto: dict) -> list:
    OUT.mkdir(parents=True, exist_ok=True)
    HISTORICO.mkdir(parents=True, exist_ok=True)
    escritos = []

    caminho = OUT / sup.NOME_DA_MATRIZ
    escritos.append(matriz.gravar(caminho))

    # O historico e por VERSAO DE EVIDENCIA, e nao por data: evento novo gera
    # arquivo novo, e o anterior continua no disco. Sobrescrever apagaria a
    # unica coisa que permite comparar "o que se sabia antes" com "o que se
    # sabe agora" -- e resultado historico nao se apaga neste projeto.
    versoes = {a.scope.evidence_version if not isinstance(a, dict)
               else (a.get("scope") or {}).get("evidence_version")
               for a in matriz.avaliacoes}
    for v in sorted(x for x in versoes if x):
        alvo = HISTORICO / f"scientific_support_matrix_{v}.json"
        if not alvo.exists():
            escritos.append(matriz.gravar(alvo))

    impacto_path = OUT / "contract_impact.json"
    impacto_path.write_text(json.dumps(impacto, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    escritos.append(impacto_path)

    (OUT / "commands.txt").write_text(
        "python scripts/temporalidade/ontologia_temporal.py\n"
        "python scripts/temporalidade/auditoria_temporal.py\n"
        "python scripts/temporalidade/dimensao_temporal.py\n"
        "python -m pytest tests/test_suporte_cientifico_temporal.py -q\n",
        encoding="utf-8")
    escritos.append(OUT / "commands.txt")
    return escritos


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--matriz", action="store_true",
                   help="imprime a matriz em JSON e sai")
    args = p.parse_args(argv)

    matriz = construir_matriz()
    if args.matriz:
        print(json.dumps(matriz.para_dict(), indent=2, ensure_ascii=False))
        return 0

    impacto = analisar_impacto_no_contrato(matriz)
    escritos = gravar(matriz, impacto)

    print("===DIMENSAO TEMPORAL: MATRIZ DE SUSTENTACAO CIENTIFICA===")
    print(f"{'regiao':<34} {'modelo':<42} {'suporte':<14} razao")
    for a in matriz.avaliacoes:
        d = a.para_dict()
        print(f"{d['scope']['region_id']:<34} {d['scope']['model_id']:<42} "
              f"{d['support']:<14} {d['razao_tecnica']}")
    print()
    print(f"contrato atual: {impacto['current_contract']}  "
          f"veredito: {impacto['veredito']}")
    print(f"g6 recusaria: {impacto['regioes_que_g6_recusaria'] or 'nenhuma'}")
    for e in escritos:
        print(f"GRAVADO={audit.rel(e)}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
