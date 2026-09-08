"""Contrato cientifico unico para precipitacao no REV-P.

Chuva e ``EVENT_FORCING``. Ela nao e terreno, rotulo, data do evento nem
evidencia observada. Este modulo define as duas variaveis historicas sem
decidir que ambas devam permanecer no estimador servido.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from math import isfinite
from typing import Iterable, Sequence

METHOD_VERSION = "rain-method-v1.0.0"
PROVIDER_ID = "open_meteo"
PRODUCT_ID = "era5_land"
PROVIDER_URL = "https://archive-api.open-meteo.com/v1/archive"
SPATIAL_RESOLUTION_DEGREES = 0.1
TEMPORAL_RESOLUTION = "daily"
TIMEZONE = "auto"
LOOKBACK_DAYS = 14
DECAY_K = 0.85
MISSING_POLICY = "fail_closed"


@dataclass(frozen=True)
class RainVariable:
    name: str
    mathematical_definition: str
    unit: str
    source: str
    spatial_resolution: str
    temporal_resolution: str
    timezone: str
    window: str
    decay_parameter: float | None
    reference_date: str
    queried_period: str
    expected_physical_sign: int
    missing_treatment: str
    extrapolation_flags: tuple[str, ...]
    method_version: str


VARIABLES = {
    "rain_max_24h": RainVariable(
        name="rain_max_24h",
        mathematical_definition="max(P_d) for d in [event_date-14, event_date-1]",
        unit="mm",
        source="Open-Meteo archive API / ERA5-Land",
        spatial_resolution="0.1 degree native grid (approximately 11 km)",
        temporal_resolution=TEMPORAL_RESOLUTION,
        timezone=TIMEZONE,
        window="14 complete local days before event_date",
        decay_parameter=None,
        reference_date="event_date; event day excluded",
        queried_period="event_date-14 through event_date-1 inclusive",
        expected_physical_sign=1,
        missing_treatment=MISSING_POLICY,
        extrapolation_flags=("below_training_p5", "above_training_p95"),
        method_version=METHOD_VERSION,
    ),
    "rain_decay_index": RainVariable(
        name="rain_decay_index",
        mathematical_definition="sum(v_i * k**(n-1-i)); oldest i=0, newest i=n-1",
        unit="mm",
        source="Open-Meteo archive API / ERA5-Land",
        spatial_resolution="0.1 degree native grid (approximately 11 km)",
        temporal_resolution=TEMPORAL_RESOLUTION,
        timezone=TIMEZONE,
        window="14 complete local days before event_date",
        decay_parameter=DECAY_K,
        reference_date="event_date; event day excluded",
        queried_period="event_date-14 through event_date-1 inclusive",
        expected_physical_sign=1,
        missing_treatment=MISSING_POLICY,
        extrapolation_flags=("below_training_p5", "above_training_p95"),
        method_version=METHOD_VERSION,
    ),
}

# Nomes preservados por compatibilidade do contrato publico. A definicao deixa
# explicito que o terceiro e um extremo bruto e extrapolativo, nao um valor
# equivalente a observacao dentro do dominio do ajuste.
SCENARIOS = {
    "mediana_observada": {
        "rain_max_24h": 13.6,
        "rain_decay_index": 23.18,
        "definition": "median of observed Recife event distribution",
        "training_percentile": 50.0,
    },
    "p90_observado": {
        "rain_max_24h": 34.4,
        "rain_decay_index": 54.45,
        "definition": "90th percentile of observed Recife event distribution",
        "training_percentile": 90.0,
    },
    "maximo_observado": {
        "rain_max_24h": 100.2,
        "rain_decay_index": 115.22,
        "definition": "raw observed maximum; extrapolative relative to model p5-p95 domain",
        "training_percentile": 100.0,
    },
}


# O PAPEL FISICO DE CADA VARIAVEL, e a lacuna que ele revela.
#
# Um evento de inundacao tem dois ingredientes, e a hidrologia os separa desde
# Kohler & Linsley (1951):
#
#   ANTECEDENTE  quanta agua o solo ja guardava quando o evento comecou. E
#                memoria, decai por drenagem e evapotranspiracao, e e o que o
#                API mede.
#   GATILHO      quanta agua caiu NO evento. E o forcing que precisa exceder a
#                capacidade restante para a agua aparecer na superficie.
#
# As DUAS variaveis deste projeto sao ANTECEDENTES. Ambas sao calculadas em
# [event_date-14, event_date-1], com o dia do evento EXCLUIDO -- entao nenhuma
# delas mede o gatilho. `rain_max_24h` nao e "o pico de chuva do evento" que o
# nome sugere: e o maior acumulado diario da janela anterior a ele.
#
# Isso importa para ler o sinal. O `+1` de `rain_max_24h` foi justificado no
# SVC-01 como "mais chuva, mais agua", que e verdade para a chuva DO evento e
# nao decorre da definicao operacional da variavel: um pico grande ha dez dias
# pode ter escoado inteiro, e o que resta dele ja esta contabilizado no API.
# Dai a correlacao de 0,71 entre as duas e o coeficiente do pico proximo de
# zero com IC cruzando -- elas competem pela mesma informacao.
#
# O gatilho e ADQUIRIVEL: basta pedir ao provedor a janela [event_date,
# event_date]. Ele nao existe hoje porque nunca foi pedido, e nao porque a
# fonte nao o tenha.
PAPEL_FISICO = {
    "rain_max_24h": {
        "papel": "ANTECEDENTE",
        "mede": "maior acumulado diario na janela anterior ao evento",
        "nao_mede": "a chuva do dia do evento",
        "sinal_declarado_no_ajuste": 1,
        "sinal_decorre_da_definicao": False,
        "por_que": ("o +1 vale para a chuva DO evento; para o maior pico "
                    "anterior a ele o sentido nao decorre da fisica, porque "
                    "o que sobrou daquele pico ja esta no indice de decaimento"),
    },
    "rain_decay_index": {
        "papel": "ANTECEDENTE",
        "mede": "umidade acumulada com decaimento (API de Kohler & Linsley)",
        "nao_mede": "a chuva do dia do evento",
        "sinal_declarado_no_ajuste": 1,
        "sinal_decorre_da_definicao": True,
        "por_que": ("solo mais saturado quando o evento comeca deixa menos "
                    "capacidade de absorcao: o sinal segue da definicao"),
    },
}

GATILHO_AUSENTE = {
    "existe_no_modelo": False,
    "o_que_seria": "acumulado de precipitacao no proprio dia do evento",
    "janela_que_o_produziria": "[event_date, event_date]",
    "adquirivel": True,
    "por_que_nao_existe": ("a janela canonica exclui o dia do evento; o "
                           "gatilho nunca foi pedido ao provedor"),
    "consequencia": ("o modelo tem duas variaveis de umidade antecedente, "
                     "correlacionadas em 0,71, e nenhuma variavel do forcing "
                     "que dispara o evento"),
    "cuidado_ao_adquirir": (
        "a chuva do dia do evento e posterior a decisao de rotular o ponto: "
        "usa-la exige verificar que o rotulo nao foi definido a partir dela, "
        "sob risco de circularidade"),
}


def physical_role(variable: str) -> dict:
    """Papel fisico da variavel, e se o sinal esperado decorre da definicao."""
    info = PAPEL_FISICO.get(str(variable).strip())
    if info is None:
        return {"conhecido": False, "variavel": variable,
                "variaveis": sorted(PAPEL_FISICO)}
    return {"conhecido": True, "variavel": variable, **info}


def trigger_window(event_date: str | date) -> tuple[str, str]:
    """A janela do gatilho: o proprio dia do evento.

    Existe para que adquirir o gatilho seja uma chamada declarada, e nao uma
    alteracao ad hoc da janela canonica -- que continua sendo a antecedente.
    """
    dt = event_date if isinstance(event_date, date) else date.fromisoformat(
        str(event_date)[:10])
    return (dt.isoformat(), dt.isoformat())


# Ate onde o produto diario alcanca, por mecanismo. Nao e opiniao sobre a
# fisica: e o casamento entre a resolucao TEMPORAL do que se mede e a escala
# de tempo em que o processo responde.
#
# Caine (1980) estabeleceu que deslizamentos rasos e corridas de detritos sao
# governados por limiares de INTENSIDADE x DURACAO, I = a * D^-b, com duracoes
# criticas de uma a poucas horas. Um acumulado de 24 h nao distingue 80 mm
# caidos em duas horas de 80 mm distribuidos no dia inteiro -- e e essa
# diferenca que decide o evento em encosta.
#
# Ochoa-Rodriguez et al. (2015) mediram o efeito equivalente no espaco para
# hidrologia urbana: abaixo de certa resolucao espacial e temporal, a entrada
# de chuva deixa de sustentar o modelo hidrodinamico. E a explicacao publicada
# do que este projeto observou empiricamente na grade de 0,1 grau.
ADEQUACAO_POR_MECANISMO = {
    "FLUVIAL_ENXURRADA": {
        "adequado": True,
        "por_que": ("resposta de bacia fluvial se acumula em escala diaria; o "
                    "acumulado de 24 h e o API cobrem a janela do processo"),
    },
    "PLUVIAL_URBANO": {
        "adequado": None,
        "por_que": ("drenagem urbana responde em escala sub-diaria; o "
                    "acumulado de 24 h cobre o volume mas nao a intensidade "
                    "que satura a rede"),
        "referencia": "Ochoa-Rodriguez et al. (2015), Journal of Hydrology",
    },
    "HIDROGEOMORFOLOGICO_SERRA": {
        "adequado": False,
        "por_que": ("encosta responde a intensidade em 1-3 h, e o limiar e "
                    "intensidade-duracao (I = a*D^-b), nao acumulado diario: "
                    "80 mm em duas horas e 80 mm no dia inteiro sao o mesmo "
                    "numero neste produto e eventos diferentes no terreno"),
        "referencia": "Caine (1980), Geografiska Annaler A",
        "exigiria": ("serie sub-diaria (radar ou pluviografo) para construir "
                     "o par intensidade-duracao"),
    },
}


def mechanism_adequacy(mechanism: str | None) -> dict:
    """A chuva diaria alcanca o mecanismo desta regiao?

    Declarar isto e o que impede o produto de oferecer, para relevo de serra,
    um forcing cuja resolucao temporal nao cobre o processo que decide o
    evento. A resposta nao muda escore nenhum: ela qualifica o que o numero
    pode significar.
    """
    if not mechanism:
        return {"conhecido": False,
                "mecanismos": sorted(ADEQUACAO_POR_MECANISMO)}
    info = ADEQUACAO_POR_MECANISMO.get(str(mechanism).strip().upper())
    if info is None:
        return {"conhecido": False, "mecanismo": mechanism,
                "mecanismos": sorted(ADEQUACAO_POR_MECANISMO)}
    return {"conhecido": True, "mecanismo": mechanism,
            "resolucao_temporal_do_produto": TEMPORAL_RESOLUTION, **info}


def scenario_values(name: str) -> dict[str, float]:
    """Devolve so valores de variaveis; metadados nao entram no estimador."""
    s = SCENARIOS.get(name)
    if s is None:
        return {}
    return {k: float(s[k]) for k in VARIABLES}


def scenario_evidence(name: str, training_domain: dict | None = None) -> dict:
    """Descreve relacao entre cenario e dominio sem descartar celula em silencio."""
    if name not in SCENARIOS:
        return {"known": False, "scenario": name, "available": list(SCENARIOS)}
    meta = SCENARIOS[name]
    values = scenario_values(name)
    by_variable = {}
    for var, value in values.items():
        bounds = (training_domain or {}).get(var)
        if not bounds or len(bounds) != 2:
            by_variable[var] = {
                "value": value, "training_domain_p5_p95": None,
                "inside_training_domain": None, "distance_to_domain": None,
            }
            continue
        lo, hi = map(float, bounds)
        inside = lo <= value <= hi
        distance = 0.0 if inside else (lo - value if value < lo else value - hi)
        by_variable[var] = {
            "value": value,
            "training_domain_p5_p95": [lo, hi],
            "inside_training_domain": inside,
            "distance_to_domain": round(distance, 6),
        }
    flags = [v for v, d in by_variable.items()
             if d["inside_training_domain"] is False]
    return {
        "known": True,
        "scenario": name,
        "definition": meta["definition"],
        "training_percentile": meta["training_percentile"],
        "values": values,
        "by_variable": by_variable,
        "inside_training_domain": False if flags else (
            True if by_variable and all(d["inside_training_domain"] is True
                                        for d in by_variable.values()) else None),
        "extrapolation": bool(flags),
        "extrapolation_variables": flags,
    }


def requested_window(event_date: str | date,
                     lookback_days: int = LOOKBACK_DAYS) -> tuple[str, str]:
    dt = event_date if isinstance(event_date, date) else date.fromisoformat(
        str(event_date)[:10])
    return ((dt - timedelta(days=lookback_days)).isoformat(),
            (dt - timedelta(days=1)).isoformat())


def parse_event_dates(values):
    """Parse documented forms and mixed offsets into one UTC timeline.

    ``utc=True`` is deliberate: pandas otherwise raises for a Series mixing
    naive timestamps, ``Z`` and explicit UTC offsets.  Consumers compare event
    days rather than local clock times, so the canonical result stays
    timezone-aware and auditable instead of silently dropping such rows.
    """
    import pandas as pd
    return pd.to_datetime(values, format="mixed", errors="coerce", utc=True)


def derive_rain_features(values: Sequence[float], *, decay_k: float = DECAY_K,
                         expected_days: int | None = LOOKBACK_DAYS) -> dict:
    """Deriva variaveis de chuva de uma serie diaria da mais antiga para a mais nova.

    O modo cientifico e fail-closed: serie incompleta, nao finita ou negativa
    e evidencia indisponivel, nao precipitacao zero.
    """
    series = [float(v) for v in values]
    if not series:
        raise ValueError("empty precipitation series")
    if expected_days is not None and len(series) != expected_days:
        raise ValueError(f"expected {expected_days} days, received {len(series)}")
    if not 0.0 < decay_k <= 1.0:
        raise ValueError("decay_k must be in (0, 1]")
    if any(not isfinite(v) for v in series):
        raise ValueError("precipitation series contains non-finite value")
    if any(v < 0 for v in series):
        raise ValueError("negative precipitation is physically impossible")
    n = len(series)
    api = sum(v * decay_k ** (n - 1 - i) for i, v in enumerate(series))
    return {
        "rain_max_24h": round(max(series), 4),
        "rain_decay_index": round(api, 4),
        "days_in_window": n,
        "decay_k": decay_k,
        "method_version": METHOD_VERSION,
    }


TRIGGER_VARIABLE = "rain_event_day"


def derive_trigger_feature(values: Sequence[float]) -> dict:
    """O acumulado do proprio dia do evento.

    Um unico dia, entao nao ha decaimento a aplicar: o API de uma janela de um
    dia e o proprio valor, e publica-lo com esse nome so criaria uma terceira
    coluna igual a esta. O que este numero acrescenta ao modelo nao e mais uma
    medida de umidade -- e o FORCING, que as duas variaveis antecedentes nao
    medem.

    Fail-closed como o resto do contrato: serie vazia, nao finita ou negativa e
    indisponibilidade, nunca zero de precipitacao.
    """
    series = [float(v) for v in values]
    if len(series) != 1:
        raise ValueError(f"a janela do gatilho tem exatamente 1 dia, recebeu {len(series)}")
    valor = series[0]
    if not isfinite(valor):
        raise ValueError("precipitacao do dia do evento nao finita")
    if valor < 0:
        raise ValueError("negative precipitation is physically impossible")
    return {
        TRIGGER_VARIABLE: round(valor, 4),
        "days_in_window": 1,
        "role": "TRIGGER",
        "method_version": METHOD_VERSION,
    }


PEAK_VARIABLE = "rain_peak_1h"


def derive_peak_feature(times: Sequence[str], values: Sequence[float]) -> dict:
    """O maior acumulado de UMA HORA dentro do dia do evento.

    POR QUE ESTA VARIAVEL EXISTE, e por que ela nao e o gatilho de novo:

    `rain_event_day` mede VOLUME -- quanta agua caiu no dia. Esta mede
    INTENSIDADE -- quao rapido ela caiu. Sao ingredientes diferentes do mesmo
    evento, e a limitacao ja estava declarada em `autopareamento_*.json`:
    "acumulado diario nao distingue intensidade: 80 mm em duas horas e 80 mm no
    dia inteiro sao o mesmo numero neste produto".

    Para alagamento urbano a intensidade e o mecanismo. A agua aparece na rua
    quando a taxa de chegada excede a taxa de escoamento da galeria, e essa
    comparacao e mm/h contra mm/h. Volume diario nao a expressa.

    Fail-closed como o resto do contrato: serie vazia, nao finita ou negativa e
    indisponibilidade, nunca zero de precipitacao.
    """
    series = [float(v) for v in values]
    if not series:
        raise ValueError("serie horaria vazia")
    if len(times) != len(series):
        raise ValueError("horas e valores com tamanhos diferentes")
    for v in series:
        if not isfinite(v):
            raise ValueError("precipitacao horaria nao finita")
        if v < 0:
            raise ValueError("negative precipitation is physically impossible")
    topo = max(range(len(series)), key=lambda i: series[i])
    return {
        PEAK_VARIABLE: round(series[topo], 4),
        "hora_do_pico": str(times[topo]),
        "horas_na_janela": len(series),
        "role": "TRIGGER_INTENSITY",
        "method_version": METHOD_VERSION,
    }


PAPEL_FISICO[PEAK_VARIABLE] = {
    "papel": "GATILHO_INTENSIDADE",
    "mede": "maior acumulado de uma hora dentro do dia do evento",
    "nao_mede": "o volume do dia, que e o `rain_event_day`",
    "sinal_declarado_no_ajuste": 1,
    "sinal_decorre_da_definicao": True,
    "por_que": ("a agua aparece quando a taxa de chegada excede a de escoamento; "
                "intensidade maior, menos capacidade restante de drenagem"),
}


def quality_control(dates: Iterable[str], values: Iterable[float], *,
                    expected_start: str | None = None,
                    expected_end: str | None = None) -> dict:
    ds = [str(d)[:10] for d in dates]
    vs = list(values)
    issues = []
    if len(ds) != len(vs):
        issues.append("date_value_length_mismatch")
    if len(set(ds)) != len(ds):
        issues.append("duplicate_dates")
    if any(v is None for v in vs):
        issues.append("missing_precipitation")
    finite = []
    for v in vs:
        try:
            finite.append(float(v))
        except (TypeError, ValueError):
            finite.append(float("nan"))
    if any(not isfinite(v) for v in finite):
        issues.append("non_finite_precipitation")
    if any(isfinite(v) and v < 0 for v in finite):
        issues.append("negative_precipitation")
    if expected_start and (not ds or ds[0] != expected_start):
        issues.append("unexpected_start_date")
    if expected_end and (not ds or ds[-1] != expected_end):
        issues.append("unexpected_end_date")
    if ds != sorted(ds):
        issues.append("dates_not_monotonic")
    return {
        "ok": not issues,
        "issues": sorted(set(issues)),
        "n_dates": len(ds),
        "n_values": len(vs),
        "expected_start": expected_start,
        "expected_end": expected_end,
    }


def contract() -> dict:
    return {
        "method_version": METHOD_VERSION,
        "semantic_role": "EVENT_FORCING",
        "provider": PROVIDER_ID,
        "product": PRODUCT_ID,
        "provider_url": PROVIDER_URL,
        "spatial_resolution_degrees": SPATIAL_RESOLUTION_DEGREES,
        "temporal_resolution": TEMPORAL_RESOLUTION,
        "timezone": TIMEZONE,
        "lookback_days": LOOKBACK_DAYS,
        "missing_policy": MISSING_POLICY,
        "variables": {k: asdict(v) for k, v in VARIABLES.items()},
        "scenarios": SCENARIOS,
        "physical_role": PAPEL_FISICO,
        "missing_trigger": GATILHO_AUSENTE,
        "mechanism_adequacy": ADEQUACAO_POR_MECANISMO,
        "period_policy": (
            "scenario-based served inference; arbitrary requested period is not "
            "used unless rain is explicitly acquired and derived for that period"),
    }
