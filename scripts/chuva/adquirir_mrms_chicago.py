"""MRMS a 1 km sobre Chicago, pelo MESMO codigo que serviu Nova York.

POR QUE IMPORTAR EM VEZ DE COPIAR:

o objetivo desta cidade e responder se `twi_dinf x chuva`, medido em Nova York,
se repete em outro lugar. Se as duas cidades passassem por implementacoes
diferentes do mesmo produto, uma diferenca de resultado poderia vir do codigo em
vez do territorio -- e nao haveria como saber qual dos dois.

Entao `urls_do_dia`, `baixar_janela` e `amostrar` vem de
`adquirir_mrms_nyc.py`, sem reimplementacao. O que muda aqui e so a origem dos
pontos e o destino: os chamados de Chicago nao estao na tabela unica, e sim em
`local_runs/aq-chicago-01/`, porque a fonte ainda nao foi promovida.

O QUE NAO FAZ: nao promove fonte, nao vira variavel, nao entra em modelo.

Uso:
    python scripts/chuva/adquirir_mrms_chicago.py --online --limite 8
    python scripts/chuva/adquirir_mrms_chicago.py --online
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.adquirir_mrms_nyc import (  # noqa: E402
    INTERVALO_MINIMO_S, MARGEM, PRODUTO, RESOLUCAO_GRAUS, amostrar,
    baixar_janela,
)

CHAMADOS = REPO / "local_runs" / "aq-chicago-01"
OUT = REPO / "local_runs" / "chuva-mrms-chicago"
JANELAS = OUT / "janelas"
AGENTE = "REV-P/1.0 (pesquisa academica de suscetibilidade a inundacao)"


def pontos_de_chicago() -> pd.DataFrame:
    """Positivos e fundo da aquisicao, com a classe declarada por arquivo."""
    partes = []
    for arquivo, classe in (("chicago311_positivos.csv", 1),
                            ("chicago311_fundo.csv", 0)):
        caminho = CHAMADOS / arquivo
        if not caminho.exists():
            raise SystemExit(
                f"ausente: {caminho}. Rode antes: "
                "python scripts/externo/aq_chicago01_chamados.py --completo")
        d = pd.read_csv(caminho)
        d["classe"] = classe
        partes.append(d)
    d = pd.concat(partes, ignore_index=True)
    d = d.rename(columns={"sr_number": "ponto_id"})
    d["lat"] = pd.to_numeric(d.latitude, errors="coerce")
    d["lon"] = pd.to_numeric(d.longitude, errors="coerce")
    d["dia"] = d.created_date.astype(str).str[:10]
    d = d[d.lat.notna() & d.lon.notna() & d.dia.str.match(r"\d{4}-\d{2}-\d{2}")]
    return d[["ponto_id", "dia", "lat", "lon", "classe", "sr_type"]].copy()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--limite", type=int, default=None)
    ns = p.parse_args()

    pontos = pontos_de_chicago()
    dias = sorted(pontos.dia.unique())
    if ns.limite:
        dias = dias[:ns.limite]
    caixa = (pontos.lon.min() - MARGEM, pontos.lat.min() - MARGEM,
             pontos.lon.max() + MARGEM, pontos.lat.max() + MARGEM)
    print("=== MRMS 1 km SOBRE CHICAGO ===")
    print(f"pontos: {len(pontos):,} ({int(pontos.classe.sum()):,} positivos) "
          f"em {len(dias):,} dias")
    print(f"caixa: {[round(c, 3) for c in caixa]}  produto: {PRODUTO}")

    JANELAS.mkdir(parents=True, exist_ok=True)
    sessao = requests.Session()
    sessao.headers["User-Agent"] = AGENTE
    por_dia = {d: g for d, g in pontos.groupby("dia")}

    linhas, falhas, baixados, replay = [], [], 0, 0
    ultimo = 0.0
    for i, dia in enumerate(dias, start=1):
        cache = JANELAS / f"{dia}.npz"
        try:
            if cache.exists():
                z = np.load(cache)
                janela = {"z": z["z"], "transform": tuple(z["transform"]),
                          "produto": str(z["produto"])}
                replay += 1
            else:
                espera = INTERVALO_MINIMO_S - (time.monotonic() - ultimo)
                if espera > 0:
                    time.sleep(espera)
                if not ns.online:
                    raise FileNotFoundError("sem cache e sem --online")
                janela = baixar_janela(dia, caixa, sessao)
                ultimo = time.monotonic()
                np.savez_compressed(cache, z=janela["z"],
                                    transform=np.array(janela["transform"]),
                                    produto=janela["produto"])
                baixados += 1
            g = por_dia[dia]
            v = amostrar(janela, g.lat.to_numpy(), g.lon.to_numpy())
            linhas.append(pd.DataFrame({"ponto_id": g.ponto_id.to_numpy(),
                                        "dia": dia,
                                        "mrms_qpe_24h_mm": v}))
        except Exception as e:                                   # noqa: BLE001
            falhas.append({"dia": dia, "erro": type(e).__name__,
                           "detalhe": str(e)[:200]})
        if i % 100 == 0 or i == len(dias):
            print(f"  {i}/{len(dias)} dias · baixados={baixados} "
                  f"replay={replay} falhas={len(falhas)}", flush=True)

    quadro = pd.concat(linhas, ignore_index=True) if linhas else pd.DataFrame()
    OUT.mkdir(parents=True, exist_ok=True)
    if not quadro.empty:
        quadro.to_csv(OUT / "mrms_por_ponto_chicago.csv", index=False)

    com_valor = int(quadro.mrms_qpe_24h_mm.notna().sum()) if not quadro.empty else 0
    manifesto = {
        "tipo": "chuva_mrms_1km_chicago",
        "gerado_por": "scripts/chuva/adquirir_mrms_chicago.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "produto": PRODUTO,
        "resolucao_graus": RESOLUCAO_GRAUS,
        "caixa": [round(c, 4) for c in caixa],
        "dias_no_universo": len(dias),
        "dias_baixados": baixados,
        "dias_por_replay": replay,
        "dias_com_falha": len(falhas),
        "pontos_amostrados": int(len(quadro)) if not quadro.empty else 0,
        "pontos_com_valor": com_valor,
        "primeiras_falhas": falhas[:20],
        "mesmo_codigo_de": "scripts/chuva/adquirir_mrms_nyc.py",
        "por_que_mesmo_codigo": (
            "a pergunta e se o efeito medido em Nova York se repete aqui; "
            "implementacoes diferentes do mesmo produto deixariam a diferenca "
            "de resultado indistinguivel de diferenca de codigo"),
        "nao_e": (
            "nao e medida de pluviometro: e estimativa de radar calibrada por "
            "pluviometro. E nao entrou em modelo nenhum"),
    }
    (OUT / "manifesto.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/chuva/adquirir_mrms_chicago.py --online\n",
        encoding="utf-8")
    print(f"\ndias baixados {baixados} | replay {replay} | falhas {len(falhas)}")
    print(f"pontos com valor: {com_valor:,}")
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
