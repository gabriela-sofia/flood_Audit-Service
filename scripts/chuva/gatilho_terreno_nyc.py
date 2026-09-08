"""O que Nova York ensina: quanto o terreno responde ao gatilho da chuva.

POR QUE NOVA YORK, E SO ELA:

Nova York nao e uma regiao servida do REV-P. Ela e a unica fonte onde existe o
contraste que as tres regioes brasileiras nao tem, e por isso e a unica que pode
ensinar o mecanismo. Medido em 02/09/2026 sobre a base de 01/09, contando
estratos (celula de chuva x dia) que contem AS DUAS classes:

    nyc         3.534 estratos   316.361 pontos   HAND varia 7,45 m dentro
    sao_paulo     172 estratos       668 pontos
    curitiba       38 estratos       116 pontos
    recife          2 estratos         4 pontos

Sem estrato informativo nao ha o que estimar. O gatilho ja esta adquirido para
99,99% da base -- o que falta nunca foi dado, e sim desenho.

-------------------------------------------------------------------------
O DESENHO, e por que ele neutraliza o processo de observacao
-------------------------------------------------------------------------

ESTRATO = (celula de 0,1 grau x dia). Dentro dele sao constantes, POR
CONSTRUCAO e nao por ajuste: a chuva do dia (o produto e o mesmo numero para a
celula inteira), a estacao, o dia da semana, a tendencia de longo prazo e o
volume de chamados daquela celula naquele dia.

CASO E CONTROLE SAO OS DOIS CHAMADOS. O positivo e um chamado hidrico; o
negativo e um chamado de OUTRO assunto -- `aq_chamados03` grava
`tipo_negativo = ausente`. Isso e `target-group background`: as duas pessoas
ligaram para a prefeitura no mesmo dia e na mesma celula. O que difere entre
elas nao e a propensao a telefonar, e sim o lugar onde estavam.

Restam, dentro do estrato: o terreno, e o assunto do chamado. E e exatamente
essa a pergunta do projeto.

CENTRAGEM POR ESTRATO como em `desenho_pareado.py`: cada variavel entra como
desvio da media do proprio estrato. E o analogo pratico da logistica
condicional, e a razao de reusar e nao inventar outra.

O EFEITO PRINCIPAL DA CHUVA NAO E ESTIMAVEL AQUI, e isso e proposital: ele e
constante no estrato e some na centragem. Quem o estima e o case-crossover
(`autopareamento_temporal.py`), que ja mediu razao de taxas 1,51 por 10 mm em Nova York.
Aqui a pergunta e outra, e e a que interessa ao mapa:

    o terreno separa MAIS os lugares quando chove mais?

Isso e literalmente a definicao de suscetibilidade do projeto -- predisposicao
do terreno SOB UM DADO FORCAMENTO -- e e a quantidade que pode viajar para
regiao sem inventario, porque e fisica e nao burocracia municipal.

-------------------------------------------------------------------------
CRITERIOS FIXADOS ANTES DE ESTIMAR
-------------------------------------------------------------------------

SINAL EXIGIDO por variavel, o mesmo do SVC-01:

    hand_m           negativo   mais alto acima do rio, menos alaga
    twi_dinf         positivo   mais encosta drenando para o ponto, mais alaga
    slope_deg        negativo   mais ingreme, a agua nao para
    elevation_rel_m  negativo   mais alto na regiao, menos alaga

LEITURA: um efeito so conta como aprendido se o IC95 por reamostragem de
ESTRATOS excluir zero E o sinal bater com a fisica. Coeficiente com sinal
invertido e resultado, nao e defeito a esconder -- ele vai publicado.

DOSE: a mesma coisa medida em faixas de chuva do dia. Se o terreno for o
mecanismo, o efeito cresce da faixa seca para a faixa chuvosa. Se nao crescer,
o desenho esta lendo outra coisa, e isso tambem vai publicado.

TRANSFERENCIA: validacao deixando UMA CELULA inteira de fora por vez. Fold que
parte um estrato deixaria a mesma celula-dia no treino e no teste.

O QUE ESTE MODULO NAO FAZ: nao ajusta modelo servido, nao promove variavel, nao
altera contrato e nao escreve em diretorio versionado.

Uso:
    python scripts/chuva/gatilho_terreno_nyc.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import (  # noqa: E402
    METHOD_VERSION, SPATIAL_RESOLUTION_DEGREES as RES, TRIGGER_VARIABLE,
    parse_event_dates,
)

TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.csv"
SERIE = REPO / "local_runs" / "chuva-serie-celula" / "serie_diaria_por_celula_nyc.csv"
OUT = REPO / "local_runs" / "nyc-terreno-sob-gatilho"

TERRENO = ["hand_m", "twi_dinf", "slope_deg", "elevation_rel_m"]
SINAL_EXIGIDO = {"hand_m": -1, "twi_dinf": +1, "slope_deg": -1,
                 "elevation_rel_m": -1}
ESTRATO = ["cl_lat", "cl_lon", "dia"]
SEMENTE = 20260902
N_BOOT = 400
MINIMO_POR_ESTRATO = 4
FAIXAS_DE_CHUVA = 4


def carregar() -> pd.DataFrame:
    d = pd.read_csv(TABELA, low_memory=False,
                    usecols=["fonte", "classe", "lat", "lon", "data_evento"] + TERRENO)
    d = d[(d.fonte == "nyc") & d.classe.isin([0, 1])].copy()
    d["ts"] = parse_event_dates(d.data_evento)
    d = d.dropna(subset=TERRENO + ["lat", "lon"])
    d = d[d.ts.notna()].copy()
    d["dia"] = d.ts.dt.strftime("%Y-%m-%d")
    d["cl_lat"] = ((d.lat / RES).round() * RES).round(4)
    d["cl_lon"] = ((d.lon / RES).round() * RES).round(4)

    serie = pd.read_csv(SERIE, usecols=["cl_lat", "cl_lon", "dia", TRIGGER_VARIABLE])
    serie["cl_lat"] = serie.cl_lat.round(4)
    serie["cl_lon"] = serie.cl_lon.round(4)
    serie = serie.drop_duplicates(ESTRATO)
    d = d.merge(serie, on=ESTRATO, how="left")
    return d[d[TRIGGER_VARIABLE].notna()].copy()


def estratos_utilizaveis(s: pd.DataFrame) -> pd.DataFrame:
    """So estrato com as duas classes e tamanho minimo entra: o resto nao informa."""
    g = s.groupby(ESTRATO).classe.agg(["size", "nunique"])
    bons = g[(g["nunique"] > 1) & (g["size"] >= MINIMO_POR_ESTRATO)].index
    return s.set_index(ESTRATO).loc[bons].reset_index()


def centrar(s: pd.DataFrame, feats: list[str]) -> np.ndarray:
    """Desvio da media do proprio estrato: remove o efeito fixo do estrato."""
    z = s[feats] - s.groupby(ESTRATO)[feats].transform("mean")
    return z.to_numpy(dtype=float)


def variacao_interna(s: pd.DataFrame, feats: list[str]) -> dict:
    saida = {}
    for f in feats:
        total = float(s[f].var())
        intra = float((s[f] - s.groupby(ESTRATO)[f].transform("mean")).var())
        saida[f] = round(intra / total, 4) if total else None
    return saida


def ajustar(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logistica centrada, sem intercepto: o estrato ja saiu na centragem."""
    from sklearn.linear_model import LogisticRegression
    # `C=np.inf` no lugar de `penalty=None`: a segunda esta depreciada e some na
    # sklearn 1.10. As duas produzem o MESMO ajuste sem penalizacao -- a propria
    # biblioteca avisa que ignora C quando a penalidade e nula --, e a troca foi
    # conferida reproduzindo os coeficientes ja publicados.
    m = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=2000)
    m.fit(X, y)
    return m.coef_[0]


def ic_por_estrato(s: pd.DataFrame, feats: list[str], rng) -> dict:
    """IC95 reamostrando ESTRATOS inteiros, nunca pontos."""
    chaves = s[ESTRATO].drop_duplicates().to_numpy()
    indice = {tuple(k): i for i, k in enumerate(chaves)}
    grupo = np.array([indice[t] for t in map(tuple, s[ESTRATO].to_numpy())])
    por_grupo = [np.flatnonzero(grupo == i) for i in range(len(chaves))]
    saida = []
    for _ in range(N_BOOT):
        sorteio = rng.integers(0, len(chaves), len(chaves))
        linhas = np.concatenate([por_grupo[i] for i in sorteio])
        amostra = s.iloc[linhas]
        if amostra.classe.nunique() < 2:
            continue
        try:
            saida.append(ajustar(centrar(amostra, feats),
                                 amostra.classe.to_numpy()))
        except Exception:                                        # noqa: BLE001
            continue
    B = np.array(saida)
    return {f: [round(float(np.percentile(B[:, j], 2.5)), 4),
                round(float(np.percentile(B[:, j], 97.5)), 4)]
            for j, f in enumerate(feats)}


def auc_deixando_celula_de_fora(s: pd.DataFrame, feats: list[str]) -> dict:
    from sklearn.metrics import roc_auc_score
    celulas = s[["cl_lat", "cl_lon"]].drop_duplicates().to_numpy()
    aucs = []
    for lat, lon in celulas:
        fora = (s.cl_lat == lat) & (s.cl_lon == lon)
        tr, te = s[~fora], s[fora]
        if tr.classe.nunique() < 2 or te.classe.nunique() < 2:
            continue
        beta = ajustar(centrar(tr, feats), tr.classe.to_numpy())
        p = centrar(te, feats) @ beta
        aucs.append(float(roc_auc_score(te.classe.to_numpy(), p)))
    return {"celulas_avaliadas": len(aucs),
            "auc_media": round(float(np.mean(aucs)), 4) if aucs else None,
            "auc_minima": round(float(np.min(aucs)), 4) if aucs else None,
            "auc_maxima": round(float(np.max(aucs)), 4) if aucs else None,
            "celulas_acima_do_acaso": int(np.sum(np.array(aucs) > 0.5)) if aucs else 0}


def por_faixa_de_chuva(s: pd.DataFrame, feats: list[str], rng) -> list[dict]:
    """O efeito do terreno medido separadamente em cada faixa de chuva do dia.

    Se o terreno for o mecanismo, o efeito cresce com a dose. Nao e interacao
    imposta por forma funcional: e a mesma estimativa repetida em faixas.
    """
    chuva_do_estrato = s.groupby(ESTRATO)[TRIGGER_VARIABLE].first()
    faixa = pd.qcut(chuva_do_estrato, FAIXAS_DE_CHUVA, labels=False,
                    duplicates="drop")
    s = s.merge(faixa.rename("faixa"), left_on=ESTRATO, right_index=True)
    saida = []
    for f in sorted(s.faixa.dropna().unique()):
        parte = s[s.faixa == f]
        if parte.classe.nunique() < 2:
            continue
        beta = ajustar(centrar(parte, feats), parte.classe.to_numpy())
        saida.append({
            "faixa": int(f),
            "chuva_mm": [round(float(parte[TRIGGER_VARIABLE].min()), 2),
                         round(float(parte[TRIGGER_VARIABLE].max()), 2)],
            "chuva_mediana_mm": round(float(parte[TRIGGER_VARIABLE].median()), 2),
            "estratos": int(parte.groupby(ESTRATO).ngroups),
            "pontos": int(len(parte)),
            "coef": {v: round(float(b), 4) for v, b in zip(feats, beta)},
            "ic95": ic_por_estrato(parte, feats, rng),
        })
    return saida


def main() -> int:
    rng = np.random.default_rng(SEMENTE)
    print("=== O QUE NOVA YORK ENSINA: TERRENO SOB GATILHO ===")
    bruto = carregar()
    print(f"chamados de Nova York com terreno e gatilho: {len(bruto):,}")

    s = estratos_utilizaveis(bruto)
    n_estratos = int(s.groupby(ESTRATO).ngroups)
    print(f"estratos informativos (celula x dia, as duas classes, >= "
          f"{MINIMO_POR_ESTRATO} pontos): {n_estratos:,}")
    print(f"pontos neles: {len(s):,} ({100*len(s)/len(bruto):.1f}% dos chamados)")
    print(f"positivos {int(s.classe.sum()):,} | negativos {int((1-s.classe).sum()):,}")

    intra = variacao_interna(s, TERRENO + [TRIGGER_VARIABLE])
    print("\nfracao da variancia que sobra DENTRO do estrato:")
    for f, v in intra.items():
        marca = "  <- constante por construcao" if f == TRIGGER_VARIABLE else ""
        print(f"  {f:18s} {v}{marca}")

    beta = ajustar(centrar(s, TERRENO), s.classe.to_numpy())
    ic = ic_por_estrato(s, TERRENO, rng)
    print("\nEFEITO DO TERRENO DENTRO DO ESTRATO "
          "(chuva, estacao e propensao a ligar ja cancelados):")
    coefs = {}
    for f, b in zip(TERRENO, beta):
        lo, hi = ic[f]
        bate = np.sign(b) == SINAL_EXIGIDO[f]
        exclui = (lo > 0) or (hi < 0)
        coefs[f] = {"coef": round(float(b), 4), "ic95": [lo, hi],
                    "ic95_exclui_zero": bool(exclui),
                    "sinal_esperado": SINAL_EXIGIDO[f],
                    "sinal_bate": bool(bate),
                    "aprendido": bool(bate and exclui)}
        print(f"  {f:18s} {b:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"sinal_esperado={SINAL_EXIGIDO[f]:+d}  "
              f"{'APRENDIDO' if bate and exclui else 'nao sustentado'}")

    transf = auc_deixando_celula_de_fora(s, TERRENO)
    print(f"\ntransferencia entre celulas (uma celula fora por vez): "
          f"AUC media {transf['auc_media']}, faixa "
          f"[{transf['auc_minima']}, {transf['auc_maxima']}], "
          f"{transf['celulas_acima_do_acaso']}/{transf['celulas_avaliadas']} "
          f"acima do acaso")

    faixas = por_faixa_de_chuva(s, TERRENO, rng)
    print("\nO EFEITO DO TERRENO CRESCE COM A CHUVA DO DIA?")
    for f in faixas:
        alvo = f["coef"]["hand_m"]
        lo, hi = f["ic95"]["hand_m"]
        print(f"  faixa {f['faixa']}  chuva {f['chuva_mm'][0]:6.1f}-"
              f"{f['chuva_mm'][1]:6.1f} mm (mediana {f['chuva_mediana_mm']:5.1f})  "
              f"{f['estratos']:5d} estratos  hand_m {alvo:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    resultado = {
        "artefato": "nyc_professor_terreno_sob_gatilho",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method_version": METHOD_VERSION,
        "semente": SEMENTE,
        "desenho": {
            "estrato": "celula de chuva 0,1 grau x dia",
            "caso": "chamado hidrico",
            "controle": "chamado de outro assunto na mesma celula e no mesmo dia",
            "tipo": "target-group background: os dois ligaram, e a propensao a "
                    "telefonar cancela dentro do estrato",
            "estimador": "logistica centrada por estrato, sem intercepto",
            "ic": f"percentil de {N_BOOT} reamostragens de ESTRATOS inteiros",
            "criterios_fixados_antes": {
                "sinal_exigido": SINAL_EXIGIDO,
                "aprendido_se": "IC95 exclui zero E sinal bate com a fisica",
            },
        },
        "amostra": {
            "chamados": int(len(bruto)),
            "estratos_informativos": n_estratos,
            "pontos_em_estrato_informativo": int(len(s)),
            "positivos": int(s.classe.sum()),
            "negativos": int((1 - s.classe).sum()),
            "fracao_intra_estrato": intra,
        },
        "efeito_do_terreno": coefs,
        "transferencia_entre_celulas": transf,
        "por_faixa_de_chuva": faixas,
        "nao_e": ("nao e modelo servido, nao promove variavel e nao altera o "
                  "contrato: e a medida do que Nova York pode ensinar"),
        "efeito_principal_da_chuva": (
            "nao estimavel neste desenho por construcao -- constante no estrato. "
            "Quem o estima e o case-crossover: 1,51 por 10 mm em Nova York"),
    }
    (OUT / "terreno_sob_gatilho_nyc.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/chuva/gatilho_terreno_nyc.py\n", encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
