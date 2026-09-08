"""Serie diaria continua de chuva por celula do produto, para desenho autopareado.

POR QUE ESTE MODULO EXISTE, e por que ele nao duplica `aq_chamados07`:

`aq_chamados07` adquire a chuva das DATAS QUE JA TEM PONTO -- uma requisicao por
(celula, data) observada. Isso basta para pendurar chuva num ponto rotulado, e
nao basta para o desenho autopareado: nele cada caso e comparado consigo mesmo
em OUTRAS datas, e essas outras datas nao tem ponto nenhum. Pedir so o que ja
tem ponto e exatamente o que impede o contraste temporal de existir.

Aqui a unidade de aquisicao vira a CELULA, e nao o par (celula, data): uma
requisicao continua cobre a janela inteira do estudo, e dela sai a serie diaria
de que qualquer data de referencia precisa. Para Nova York sao dezesseis
requisicoes no lugar de 811, e o resultado contem estritamente mais.

-------------------------------------------------------------------------
AS TRES VARIAVEIS SAIEM DAS MESMAS FUNCOES DO CONTRATO
-------------------------------------------------------------------------

Nada e recalculado com formula local. Para cada dia `d` da janela util:

  rain_event_day     `derive_trigger_feature([P(d)])`          janela [d, d]
  rain_max_24h       `derive_rain_features(P[d-14 .. d-1])`    antecedente
  rain_decay_index   idem, indice de decaimento de Kohler & Linsley

Se a serie fatiada aqui produzisse numero diferente do que `aq_chamados07`
produziu pedindo a janela ao provedor, uma das duas estaria errada e ninguem
saberia qual. Por isso `--conferir` refaz a comparacao par a par contra
`chuva_por_celula_data.csv` e falha se houver uma divergencia sequer.

-------------------------------------------------------------------------
O QUE ELE NAO FAZ
-------------------------------------------------------------------------

Nao estima nada, nao rotula ponto e nao altera a tabela unica. Produz serie e
manifesto -- o insumo do `autopareamento_temporal.py`, que e quem pergunta se a chuva
transitoria muda o risco no mesmo lugar.

Uso:
    python scripts/chuva/serie_por_celula.py --online
    python scripts/chuva/serie_por_celula.py --conferir      # so replay
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import (  # noqa: E402
    LOOKBACK_DAYS, METHOD_VERSION, PRODUCT_ID, PROVIDER_ID,
    SPATIAL_RESOLUTION_DEGREES, TIMEZONE, TRIGGER_VARIABLE,
    derive_rain_features, derive_trigger_feature, parse_event_dates,
    quality_control,
)
from chuva.provedores import OpenMeteoEra5LandProvider, RainRequest  # noqa: E402

TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.csv"
PARES_AQ07 = REPO / "local_runs" / "aq-chuva-cidades" / "chuva_por_celula_data.csv"
OUT = REPO / "local_runs" / "chuva-serie-celula"
CACHE = OUT / "cache"

# As duas cidades com data POR OCORRENCIA e classes que dividem o dia. As
# outras seis tem data de lote (cena, ativacao) e nao sustentam autopareamento
# temporal -- ver `procedencia_das_datas.py`.
FONTES = ("nyc", "sao_paulo")
INTERVALO_MINIMO_S = 0.2


def celulas_do_estudo(fontes=FONTES) -> pd.DataFrame:
    """Celulas do produto e janela util de cada uma, a partir dos pontos."""
    d = pd.read_csv(TABELA, low_memory=False)
    d = d[d.fonte.isin(fontes) & d.classe.isin([0, 1])].copy()
    d["ts"] = parse_event_dates(d.data_evento)
    d = d[d.ts.notna() & d.lat.notna() & d.lon.notna()]
    r = SPATIAL_RESOLUTION_DEGREES
    d["cl_lat"] = (d.lat / r).round() * r
    d["cl_lon"] = (d.lon / r).round() * r
    d["dia"] = d.ts.dt.strftime("%Y-%m-%d")
    return (d.groupby(["fonte", "cl_lat", "cl_lon"], as_index=False)
            .agg(pontos=("classe", "size"), positivos=("classe", "sum"),
                 primeiro_dia=("dia", "min"), ultimo_dia=("dia", "max"))
            .sort_values(["fonte", "cl_lat", "cl_lon"]).reset_index(drop=True))


def celulas_dos_casos(fonte: str) -> pd.DataFrame:
    """Celulas e janela vindas dos CASOS brutos, e nao da tabela unica.

    A tabela unica so guarda o que a promocao admitiu -- em Nova York, 69 dias
    de uma amostra de 5.000 linhas. O bruto tem 2.402 dias, e e essa extensao
    que da contraste temporal ao autopareamento.
    """
    from chuva.casos_das_centrais import celulas_e_janela

    caminho = (REPO / "local_runs" / "chuva-casos-centrais"
               / f"casos_celula_dia_{fonte}.csv")
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho.name} ausente; rode "
            f"scripts/chuva/casos_das_centrais.py --fonte {fonte}")
    return celulas_e_janela(pd.read_csv(caminho))


def _serie_da_celula(prov: OpenMeteoEra5LandProvider, lat: float, lon: float,
                     inicio: str, fim: str, *, online: bool) -> tuple:
    """Mapa dia -> precipitacao, com o controle de qualidade do contrato."""
    aq = prov.acquire(RainRequest(latitude=float(lat), longitude=float(lon),
                                  start_date=inicio, end_date=fim,
                                  timezone=TIMEZONE), online=online)
    diario = aq.raw_payload.get("daily") or {}
    datas = [str(x)[:10] for x in (diario.get("time") or [])]
    valores = diario.get("precipitation_sum") or []
    qc = quality_control(datas, valores, expected_start=inicio, expected_end=fim)
    if not qc["ok"]:
        raise ValueError(f"controle de qualidade: {qc['issues']}")
    return dict(zip(datas, [float(v) for v in valores])), aq


def derivar_dia(serie: dict[str, float], dia: str) -> dict | None:
    """As tres variaveis do contrato para um dia, ou None se a janela nao fecha.

    Fail-closed: janela antecedente incompleta e indisponibilidade, nunca zero.
    """
    d = date.fromisoformat(dia)
    if dia not in serie:
        return None
    janela = []
    for k in range(LOOKBACK_DAYS, 0, -1):
        chave = (d - timedelta(days=k)).isoformat()
        if chave not in serie:
            return None
        janela.append(serie[chave])
    antecedentes = derive_rain_features(janela)
    gatilho = derive_trigger_feature([serie[dia]])
    return {
        "rain_max_24h": antecedentes["rain_max_24h"],
        "rain_decay_index": antecedentes["rain_decay_index"],
        TRIGGER_VARIABLE: gatilho[TRIGGER_VARIABLE],
    }


def construir(celulas: pd.DataFrame, *, online: bool) -> tuple[pd.DataFrame, list]:
    prov = OpenMeteoEra5LandProvider(CACHE, min_interval_seconds=INTERVALO_MINIMO_S)
    linhas, falhas = [], []
    for i, c in enumerate(celulas.itertuples(index=False), start=1):
        # A janela pedida comeca LOOKBACK dias antes do primeiro dia util, para
        # que o antecedente do primeiro dia tambem feche.
        inicio = (date.fromisoformat(c.primeiro_dia)
                  - timedelta(days=LOOKBACK_DAYS)).isoformat()
        try:
            serie, aq = _serie_da_celula(prov, c.cl_lat, c.cl_lon, inicio,
                                         c.ultimo_dia, online=online)
        except Exception as e:                                    # noqa: BLE001
            falhas.append({"fonte": c.fonte, "cl_lat": c.cl_lat,
                           "cl_lon": c.cl_lon, "motivo": type(e).__name__,
                           "detalhe": str(e)[:200]})
            continue
        dia = date.fromisoformat(c.primeiro_dia)
        ultimo = date.fromisoformat(c.ultimo_dia)
        while dia <= ultimo:
            valores = derivar_dia(serie, dia.isoformat())
            if valores is not None:
                linhas.append({
                    "fonte": c.fonte, "cl_lat": float(c.cl_lat),
                    "cl_lon": float(c.cl_lon), "dia": dia.isoformat(),
                    **valores, "cache_mode": aq.cache_mode,
                    "raw_sha256": aq.raw_sha256,
                })
            dia += timedelta(days=1)
        print(f"  {i}/{len(celulas)} celulas · dias acumulados={len(linhas)}",
              flush=True)
    return pd.DataFrame(linhas), falhas


def conferir_contra_aq07(serie: pd.DataFrame) -> dict:
    """A serie fatiada reproduz o que o provedor devolveu por janela pedida?

    Divergencia aqui nao e detalhe numerico: significa que dois caminhos do
    mesmo contrato produzem valores diferentes para a mesma celula-dia, e o
    projeto ja teve esse modo de falha com hash de proveniencia.
    """
    if not PARES_AQ07.exists():
        return {"conferido": False, "motivo": f"ausente: {PARES_AQ07.name}"}
    ref = pd.read_csv(PARES_AQ07)
    cols = ["rain_max_24h", "rain_decay_index", TRIGGER_VARIABLE]
    j = ref.merge(serie, on=["fonte", "cl_lat", "cl_lon", "dia"],
                  suffixes=("_aq07", "_serie"), how="inner")
    divergencias = []
    for col in cols:
        dif = (j[f"{col}_aq07"] - j[f"{col}_serie"]).abs()
        # 1e-4 e a propria precisao de arredondamento declarada no contrato.
        ruins = j[dif > 1e-4]
        for r in ruins.head(5).itertuples(index=False):
            divergencias.append({"variavel": col, "fonte": r.fonte,
                                 "dia": r.dia,
                                 "aq07": getattr(r, f"{col}_aq07"),
                                 "serie": getattr(r, f"{col}_serie")})
    return {
        "conferido": True,
        "pares_comparados": int(len(j)),
        "pares_no_aq07": int(len(ref)),
        "divergencias": len(divergencias),
        "exemplos": divergencias[:10],
        "criterio": "diferenca absoluta <= 1e-4 em cada variavel do contrato",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online", action="store_true",
                    help="autoriza chamadas ao provedor quando o cache nao tem")
    ap.add_argument("--conferir", action="store_true",
                    help="compara com aq_chamados07 e falha se divergir")
    ap.add_argument("--de-casos", default=None, choices=["nyc", "sao_paulo"],
                    help="celulas e janela dos casos brutos, nao da tabela unica")
    ns = ap.parse_args()

    if ns.de_casos:
        celulas = celulas_dos_casos(ns.de_casos)
        destino_nome = f"serie_diaria_por_celula_{ns.de_casos}.csv"
        manifesto_nome = f"manifesto_{ns.de_casos}.json"
    else:
        if not TABELA.exists():
            print(f"tabela unica ausente: {TABELA}")
            return 2
        celulas = celulas_do_estudo()
        destino_nome, manifesto_nome = ("serie_diaria_por_celula.csv",
                                        "manifesto.json")
    print(f"celulas do produto: {len(celulas)}"
          f"{' — online autorizado' if ns.online else ' — replay, sem rede'}")
    for f, s in celulas.groupby("fonte"):
        dias = ((pd.to_datetime(s.ultimo_dia) - pd.to_datetime(s.primeiro_dia))
                .dt.days + 1)
        print(f"  {f:10s} celulas={len(s):3d} janela_max={int(dias.max())} dias")

    serie, falhas = construir(celulas, online=ns.online)
    if serie.empty:
        print("nenhuma serie construida")
        if falhas:
            print(f"primeira falha: {falhas[0]}")
        return 1

    conferencia = conferir_contra_aq07(serie)

    OUT.mkdir(parents=True, exist_ok=True)
    destino = OUT / destino_nome
    serie.sort_values(["fonte", "cl_lat", "cl_lon", "dia"]).to_csv(destino,
                                                                   index=False)
    manifesto = {
        "artefato": "chuva_serie_por_celula",
        "schema_version": "1.0.0",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method_version": METHOD_VERSION,
        "provider": PROVIDER_ID,
        "produto": PRODUCT_ID,
        "por_que_por_celula": (
            "no desenho autopareado o caso e comparado consigo mesmo em outras "
            "datas, e essas datas nao tem ponto: pedir so as datas com ponto e "
            "o que impede o contraste temporal de existir"),
        "celulas_pedidas": int(len(celulas)),
        "celulas_com_serie": int(serie.groupby(
            ["fonte", "cl_lat", "cl_lon"]).ngroups),
        "celulas_com_falha": len(falhas),
        "dias_derivados": int(len(serie)),
        "por_fonte": {f: {"celulas": int(s.groupby(["cl_lat", "cl_lon"]).ngroups),
                          "dias": int(len(s)),
                          "primeiro_dia": s.dia.min(), "ultimo_dia": s.dia.max()}
                      for f, s in serie.groupby("fonte")},
        "por_cache_mode": serie.cache_mode.value_counts().to_dict(),
        "primeiras_falhas": falhas[:10],
        "equivalencia_com_aq07": conferencia,
        "variaveis": ["rain_max_24h", "rain_decay_index", TRIGGER_VARIABLE],
        "janela_antecedente_dias": LOOKBACK_DAYS,
        "fail_closed": ("dia cuja janela antecedente nao fecha nao entra na "
                        "serie; ausencia nunca vira zero de precipitacao"),
        "nao_e": ("nao e rotulo, nao e evidencia de ocorrencia e nao autoriza "
                  "claim causal: e a serie do forcing no produto unico"),
        "saida": str(destino.relative_to(REPO)),
        "sha256_saida": hashlib.sha256(destino.read_bytes()).hexdigest(),
    }
    (OUT / manifesto_nome).write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/chuva/serie_por_celula.py --online --conferir\n",
        encoding="utf-8")

    print(f"\ndias derivados: {len(serie):,} em "
          f"{manifesto['celulas_com_serie']} celulas")
    if conferencia.get("conferido"):
        print(f"equivalencia com aq_chamados07: {conferencia['pares_comparados']} "
              f"pares comparados, {conferencia['divergencias']} divergencias")
    if falhas:
        print(f"falhas: {len(falhas)} — primeira: {falhas[0]['motivo']}")
    print(f"MANIFESTO={OUT / manifesto_nome}")

    if ns.conferir and conferencia.get("divergencias"):
        print("FALHA: a serie fatiada diverge do que o provedor devolveu por janela")
        return 3
    if ns.conferir and not conferencia.get("conferido"):
        print("FALHA: nao havia com o que conferir")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
