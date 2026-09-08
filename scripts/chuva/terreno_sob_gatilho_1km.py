"""A interacao terreno x chuva medida ENTRE LUGARES, e nao entre datas.

O QUE MUDA EM RELACAO A `gatilho_terreno_nyc.py`:

Aquele estudo usou o gatilho do ERA5-Land a 0,1 grau. Nova York inteira cabe em
19 celulas, e so 7,5% da variancia da chuva era entre celulas no mesmo dia.
Com a chuva praticamente constante dentro do dia, o unico jeito de ver a
interacao era comparar FAIXAS DE DOSE ENTRE DIAS -- e foi o que se fez, achando
o efeito do TWI crescendo 3,7x da faixa seca para a faixa forte.

Com o MRMS a 0,01 grau (~1 km) a chuva passa a variar DENTRO do dia. Entao:

    estrato = DIA

Dentro dele continuam constantes a estacao, o dia da semana, a tendencia e o
volume de chamados daquele dia. Mas agora a chuva NAO e constante: ela varia de
bairro para bairro. E ai a pergunta do mapa fica estimavel na escala do mapa:

    no mesmo dia, o lugar que recebeu mais chuva E tem terreno pior
    concentra mais chamado de agua?

CASO E CONTROLE continuam sendo os dois chamados -- um hidrico, outro de outro
assunto. `target-group background`: as duas pessoas ligaram, entao a propensao a
telefonar cancela dentro do estrato.

CRITERIOS FIXADOS ANTES DE ESTIMAR, os mesmos do estudo anterior:

    hand_m           negativo    twi_dinf         positivo
    slope_deg        negativo    elevation_rel_m  negativo
    mrms_qpe_24h_mm  positivo    (agora estimavel: varia dentro do dia)

    twi_dinf x chuva   positivo   mais convergencia rende mais quando chove mais
    hand_m  x chuva    negativo   estar alto acima do rio protege mais quando chove

Um efeito so conta como aprendido se o IC95 por reamostragem de ESTRATOS
excluir zero E o sinal bater com a fisica.

O QUE NAO FAZ: nao ajusta modelo servido, nao promove variavel, nao altera
contrato.

Uso:
    python scripts/chuva/terreno_sob_gatilho_1km.py
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

from chuva.contrato_chuva import METHOD_VERSION, parse_event_dates  # noqa: E402

TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.csv"
MRMS = REPO / "local_runs" / "chuva-mrms-nyc" / "mrms_por_ponto_nyc.csv"
OUT = REPO / "local_runs" / "nyc-terreno-sob-gatilho"

CHUVA = "mrms_qpe_24h_mm"
TERRENO = ["hand_m", "twi_dinf", "slope_deg", "elevation_rel_m"]
INTERACOES = ["twi_dinf", "hand_m"]
ESTRATO = ["dia"]

SINAL_EXIGIDO = {"hand_m": -1, "twi_dinf": +1, "slope_deg": -1,
                 "elevation_rel_m": -1, CHUVA: +1,
                 "twi_dinf_x_chuva": +1, "hand_m_x_chuva": -1}
SEMENTE = 20260902
N_BOOT = 300
MINIMO_POR_ESTRATO = 8


def carregar() -> pd.DataFrame:
    if not MRMS.exists():
        raise FileNotFoundError(
            f"{MRMS} ausente. Rode antes: "
            "python scripts/chuva/adquirir_mrms_nyc.py --online")
    d = pd.read_csv(TABELA, low_memory=False,
                    usecols=["ponto_id", "fonte", "classe", "data_evento"] + TERRENO)
    d = d[(d.fonte == "nyc") & d.classe.isin([0, 1])].copy()
    d["ts"] = parse_event_dates(d.data_evento)
    d = d.dropna(subset=TERRENO)
    d = d[d.ts.notna()].copy()
    d["dia"] = d.ts.dt.strftime("%Y-%m-%d")

    chuva = pd.read_csv(MRMS).drop_duplicates(["ponto_id", "dia"])
    d = d.merge(chuva[["ponto_id", "dia", CHUVA]], on=["ponto_id", "dia"],
                how="inner")
    return d[d[CHUVA].notna()].copy()


def estratos_utilizaveis(s: pd.DataFrame) -> pd.DataFrame:
    g = s.groupby(ESTRATO).classe.agg(["size", "nunique"])
    bons = g[(g["nunique"] > 1) & (g["size"] >= MINIMO_POR_ESTRATO)].index
    return s[s.dia.isin(bons)].copy()


def com_interacoes(s: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """A interacao entra como produto das variaveis JA padronizadas.

    Padronizar antes evita que a escala do produto (metros x milimetros) domine
    o ajuste, e deixa os coeficientes comparaveis entre si.
    """
    z = s.copy()
    feats = TERRENO + [CHUVA]
    for f in feats:
        dp = z[f].std()
        z[f] = (z[f] - z[f].mean()) / (dp if dp else 1.0)
    nomes = list(feats)
    for f in INTERACOES:
        nome = f"{f}_x_chuva"
        z[nome] = z[f] * z[CHUVA]
        nomes.append(nome)
    return z, nomes


def centrar(s: pd.DataFrame, feats: list[str]) -> np.ndarray:
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
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=np.inf, fit_intercept=False, max_iter=3000)
    m.fit(X, y)
    return m.coef_[0]


def ic_por_estrato(s: pd.DataFrame, feats: list[str], rng) -> dict:
    dias = s.dia.unique()
    por_dia = {d: g for d, g in s.groupby("dia")}
    saida = []
    for _ in range(N_BOOT):
        sorteio = rng.choice(dias, len(dias), replace=True)
        amostra = pd.concat([por_dia[d] for d in sorteio], ignore_index=False)
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


def main() -> int:
    rng = np.random.default_rng(SEMENTE)
    print("=== TERRENO SOB GATILHO, COM CHUVA A 1 KM ===")
    bruto = carregar()
    print(f"chamados com terreno e chuva MRMS: {len(bruto):,}")

    s = estratos_utilizaveis(bruto)
    n_dias = int(s.dia.nunique())
    print(f"dias informativos (as duas classes, >= {MINIMO_POR_ESTRATO} pontos): "
          f"{n_dias:,}")
    print(f"pontos neles: {len(s):,} | positivos {int(s.classe.sum()):,}")

    z, nomes = com_interacoes(s)
    intra = variacao_interna(z, nomes)
    print("\nfracao da variancia que sobra DENTRO do dia:")
    for f, v in intra.items():
        print(f"  {f:20s} {v}")
    if (intra.get(CHUVA) or 0) < 0.05:
        print("  AVISO: a chuva quase nao varia dentro do dia; a interacao "
              "nao e identificavel nesta escala")

    beta = ajustar(centrar(z, nomes), z.classe.to_numpy())
    ic = ic_por_estrato(z, nomes, rng)
    print("\nEFEITOS DENTRO DO DIA (estacao, tendencia e propensao a ligar "
          "cancelados):")
    coefs = {}
    for f, b in zip(nomes, beta):
        lo, hi = ic[f]
        esperado = SINAL_EXIGIDO.get(f)
        bate = esperado is None or np.sign(b) == esperado
        exclui = (lo > 0) or (hi < 0)
        coefs[f] = {"coef": round(float(b), 4), "ic95": [lo, hi],
                    "ic95_exclui_zero": bool(exclui),
                    "sinal_esperado": esperado, "sinal_bate": bool(bate),
                    "aprendido": bool(bate and exclui)}
        print(f"  {f:20s} {b:+.4f}  IC95 [{lo:+.4f}, {hi:+.4f}]  "
              f"{'APRENDIDO' if bate and exclui else 'nao sustentado'}")

    resultado = {
        "tipo": "terreno_sob_gatilho_1km",
        "gerado_por": "scripts/chuva/terreno_sob_gatilho_1km.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method_version": METHOD_VERSION,
        "semente": SEMENTE,
        "desenho": {
            "estrato": "dia",
            "por_que_o_dia_basta_agora": (
                "com a chuva a 1 km ela varia DENTRO do dia; com 0,1 grau nao "
                "variava, e por isso o estudo anterior precisou estratificar "
                "por celula x dia e comparar faixas de dose entre datas"),
            "caso": "chamado hidrico",
            "controle": "chamado de outro assunto no mesmo dia",
            "estimador": "logistica centrada por estrato, sem intercepto",
            "ic": f"percentil de {N_BOOT} reamostragens de dias inteiros",
            "criterios_fixados_antes": {"sinal_exigido": SINAL_EXIGIDO},
        },
        "amostra": {
            "chamados": int(len(bruto)),
            "dias_informativos": n_dias,
            "pontos": int(len(s)),
            "positivos": int(s.classe.sum()),
            "fracao_intra_estrato": intra,
        },
        "efeitos": coefs,
        "nao_e": (
            "nao e modelo servido e nao promove variavel: e a medida da "
            "interacao terreno x forcamento na escala do bairro. O MRMS cobre "
            "apenas os Estados Unidos, entao este resultado nao transfere "
            "sozinho para regiao brasileira"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "terreno_sob_gatilho_1km.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
