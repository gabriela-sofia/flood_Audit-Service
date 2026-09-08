"""MRMS a 1 km sobre Nova York: a chuva finalmente varia dentro da cidade.

O PROBLEMA QUE ESTE SCRIPT RESOLVE, medido antes de escreve-lo:

O produto de chuva do projeto e ERA5-Land a 0,1 grau. Nova York inteira cabe em
19 celulas, e apenas 7,5% da variancia do gatilho e entre celulas no mesmo dia.
Consequencia direta: a interacao terreno x chuva so pode ser medida comparando
FAIXAS DE DOSE ENTRE DIAS, e nunca LUGARES DENTRO DO MESMO DIA -- que e a
pergunta do mapa.

O MRMS do NOAA cobre os EUA a 0,01 grau (~1 km). Sobre a caixa de Nova York isso
sao ~2.160 celulas no lugar de 19: um fator de 114.

    produto   MultiSensor_QPE_24H_Pass2
    o que e   acumulado de 24 h estimado por radar E CORRIGIDO POR PLUVIOMETRO
    grade     0,01 grau, CONUS (-130..-60, 20..55)
    leitor    GDAL/rasterio, driver GRIB -- nenhuma dependencia nova

-------------------------------------------------------------------------
A JANELA, e por que o desalinhamento nao contamina a comparacao
-------------------------------------------------------------------------

O arquivo valido em 00:00Z do dia D+1 acumula 00Z de D ate 00Z de D+1. Nova York
e UTC-4/-5, entao essa janela termina as 19h ou 20h locais de D: ela NAO e o dia
local.

Isso ficaria grave se a pergunta fosse "quanto choveu naquele dia". Nao e. A
pergunta e "dentro deste dia, o lugar que recebeu mais chuva teve mais chamado
de agua?" -- e a janela e a MESMA para todos os pontos do dia. O desalinhamento
e constante dentro do estrato e cancela junto com tudo o mais que e constante
nele. O que nao cancela, e por isso o script existe, e a variacao ESPACIAL.

-------------------------------------------------------------------------
DISCO
-------------------------------------------------------------------------

Sao 2.402 dias a ~0,7 MB. Baixar e guardar seria 1,6 GB de CONUS inteiro para
usar uma janela de 40x54 pixels. O script recorta a janela, guarda so ela em
`.npz` (~17 KB) e descarta o GRIB. O replay depois nao toca a rede.

O QUE NAO FAZ: nao entra em modelo, nao promove variavel, nao altera contrato.

Uso:
    python scripts/chuva/adquirir_mrms_nyc.py --online --limite 5
    python scripts/chuva/adquirir_mrms_nyc.py --online
    python scripts/chuva/adquirir_mrms_nyc.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from chuva.contrato_chuva import parse_event_dates  # noqa: E402

TABELA = REPO / "local_runs" / "ds-05-tabela-unica" / "tabela_unica_admitidos_v2.csv"
OUT = REPO / "local_runs" / "chuva-mrms-nyc"
JANELAS = OUT / "janelas"

BASE = "https://mtarchive.geol.iastate.edu/{ano}/{mes}/{dia}/mrms/ncep/{produto}/"
ARQUIVO = "{produto}_00.00_{stamp}-000000.grib2.gz"

# O MRMS v12 (outubro de 2020) renomeou o produto corrigido por pluviometro. Os
# dois nomes sao o MESMO produto conceitual -- acumulado de 24 h com correcao
# por pluviometro -- e a serie do projeto atravessa a troca: 2020 comeca em
# `GaugeCorr` e termina em `MultiSensor`. Tentar so o nome novo devolvia 404 em
# todo o inicio da serie, e um 404 silencioso viraria "sem chuva".
PRODUTOS = ("MultiSensor_QPE_24H_Pass2", "GaugeCorr_QPE_24H")
PRODUTO = " ou ".join(PRODUTOS)
RESOLUCAO_GRAUS = 0.01

# Margem em graus ao redor da caixa dos pontos, para nao recortar na borda.
MARGEM = 0.05
INTERVALO_MINIMO_S = 0.5
# O MRMS marca ausencia com valores muito negativos (-999, -3). Ausencia nunca
# vira zero de precipitacao: e o mesmo fail-closed do contrato de chuva.
PISO_VALIDO = -0.001


def pontos_de_nyc() -> pd.DataFrame:
    d = pd.read_csv(TABELA, low_memory=False,
                    usecols=["ponto_id", "fonte", "classe", "lat", "lon",
                             "data_evento"])
    d = d[(d.fonte == "nyc") & d.classe.isin([0, 1])].copy()
    d["ts"] = parse_event_dates(d.data_evento)
    d = d[d.ts.notna() & d.lat.notna() & d.lon.notna()].copy()
    d["dia"] = d.ts.dt.strftime("%Y-%m-%d")
    return d[["ponto_id", "dia", "lat", "lon", "classe"]]


def urls_do_dia(dia: str) -> list[str]:
    """O arquivo valido em 00:00Z do dia SEGUINTE cobre 00Z..00Z do dia pedido.

    Devolve um candidato por nome de produto, na ordem de preferencia.
    """
    seguinte = datetime.strptime(dia, "%Y-%m-%d") + timedelta(days=1)
    stamp = seguinte.strftime("%Y%m%d")
    return [BASE.format(ano=seguinte.year, mes=f"{seguinte.month:02d}",
                        dia=f"{seguinte.day:02d}", produto=produto)
            + ARQUIVO.format(produto=produto, stamp=stamp)
            for produto in PRODUTOS]


def baixar_janela(dia: str, caixa: tuple, sessao) -> dict | None:
    """Baixa o GRIB do dia, recorta a caixa e devolve a janela; nao guarda o GRIB."""
    import rasterio
    from rasterio.windows import from_bounds

    lon_min, lat_min, lon_max, lat_max = caixa
    conteudo, usado, codigos = None, None, []
    for url in urls_do_dia(dia):
        resposta = sessao.get(url, timeout=180)
        codigos.append(f"{url.rsplit('/', 2)[-2]}:{resposta.status_code}")
        if resposta.status_code == 200:
            conteudo, usado = resposta.content, url.rsplit("/", 2)[-2]
            break
    if conteudo is None:
        raise RuntimeError("nenhum produto disponivel: " + ", ".join(codigos))
    bruto = gzip.decompress(conteudo)
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(bruto)
        caminho = Path(tmp.name)
    try:
        with rasterio.open(caminho) as fonte:
            janela = from_bounds(lon_min, lat_min, lon_max, lat_max,
                                 fonte.transform)
            z = fonte.read(1, window=janela).astype("float32")
            transform = fonte.window_transform(janela)
        return {"z": z, "transform": tuple(transform)[:6], "produto": usado}
    finally:
        caminho.unlink(missing_ok=True)


def amostrar(janela: dict, lats, lons):
    """Valor do pixel de cada ponto. Ausencia vira NaN, nunca zero."""
    from rasterio.transform import Affine, rowcol

    z = janela["z"]
    t = Affine(*janela["transform"])
    linhas, colunas = rowcol(t, list(lons), list(lats))
    linhas = np.asarray(linhas)
    colunas = np.asarray(colunas)
    dentro = ((linhas >= 0) & (linhas < z.shape[0])
              & (colunas >= 0) & (colunas < z.shape[1]))
    v = np.full(len(lats), np.nan, dtype="float32")
    v[dentro] = z[linhas[dentro], colunas[dentro]]
    v[v < PISO_VALIDO] = np.nan
    return v


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--limite", type=int, default=None)
    ns = p.parse_args()

    if not TABELA.exists():
        print(f"tabela unica ausente: {TABELA}")
        return 2

    pontos = pontos_de_nyc()
    caixa = (pontos.lon.min() - MARGEM, pontos.lat.min() - MARGEM,
             pontos.lon.max() + MARGEM, pontos.lat.max() + MARGEM)
    dias = sorted(pontos.dia.unique())
    universo = len(dias)
    if ns.limite:
        dias = dias[:ns.limite]
    print(f"pontos de Nova York: {len(pontos):,} | dias: {universo:,} "
          f"({len(dias):,} nesta corrida)")
    print(f"caixa: lon [{caixa[0]:.3f}, {caixa[2]:.3f}]  "
          f"lat [{caixa[1]:.3f}, {caixa[3]:.3f}]")

    JANELAS.mkdir(parents=True, exist_ok=True)
    sessao = None
    if ns.online:
        import requests
        sessao = requests.Session()

    linhas, falhas, baixados, replays = [], [], 0, 0
    por_dia = {d: g for d, g in pontos.groupby("dia")}
    ultimo = 0.0
    for i, dia in enumerate(dias, start=1):
        cache = JANELAS / f"{dia}.npz"
        try:
            if cache.exists():
                bruto = np.load(cache)
                janela = {"z": bruto["z"], "transform": tuple(bruto["transform"])}
                replays += 1
            elif not ns.online:
                raise FileNotFoundError("janela ausente e execucao offline")
            else:
                espera = INTERVALO_MINIMO_S - (time.monotonic() - ultimo)
                if espera > 0:
                    time.sleep(espera)
                janela = baixar_janela(dia, caixa, sessao)
                ultimo = time.monotonic()
                np.savez_compressed(cache, z=janela["z"],
                                    transform=np.array(janela["transform"]))
                baixados += 1
        except Exception as e:                                   # noqa: BLE001
            falhas.append({"dia": dia, "motivo": type(e).__name__,
                           "detalhe": str(e)[:160]})
            continue

        g = por_dia[dia]
        v = amostrar(janela, g.lat.to_numpy(), g.lon.to_numpy())
        linhas.append(pd.DataFrame({"ponto_id": g.ponto_id.to_numpy(),
                                    "dia": dia, "mrms_qpe_24h_mm": v}))
        if i % 100 == 0 or i == len(dias):
            print(f"  {i}/{len(dias)} dias · baixados={baixados} "
                  f"replay={replays} falhas={len(falhas)}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    quadro = pd.concat(linhas, ignore_index=True) if linhas else pd.DataFrame()
    destino = OUT / "mrms_por_ponto_nyc.csv"
    if not quadro.empty:
        if destino.exists():
            antes = pd.read_csv(destino)
            quadro = pd.concat([antes, quadro], ignore_index=True)
        quadro = quadro.drop_duplicates(["ponto_id", "dia"], keep="last")
        quadro.to_csv(destino, index=False)

    # Quanto da variacao do gatilho passa a ser ENTRE LUGARES no mesmo dia.
    intra = None
    if not quadro.empty and quadro.mrms_qpe_24h_mm.notna().any():
        bom = quadro.dropna(subset=["mrms_qpe_24h_mm"])
        total = float(bom.mrms_qpe_24h_mm.var())
        dentro = float((bom.mrms_qpe_24h_mm
                        - bom.groupby("dia").mrms_qpe_24h_mm.transform("mean")).var())
        intra = round(dentro / total, 4) if total else None

    manifesto = {
        "tipo": "chuva_mrms_1km_nyc",
        "gerado_por": "scripts/chuva/adquirir_mrms_nyc.py",
        "gerado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "produto": PRODUTO,
        "fonte": "NOAA MRMS via mtarchive.geol.iastate.edu",
        "resolucao_graus": RESOLUCAO_GRAUS,
        "janela_declarada": ("acumulado de 00:00Z do dia ate 00:00Z do dia "
                             "seguinte; nao e o dia local de Nova York"),
        "por_que_o_desalinhamento_nao_contamina": (
            "a janela e a mesma para todos os pontos do dia, entao ela e "
            "constante dentro do estrato e cancela junto com o resto; o que nao "
            "cancela, e o objetivo, e a variacao espacial"),
        "modo": "online_autorizado" if ns.online else "replay_offline",
        "dias_no_universo": universo,
        "dias_nesta_corrida": len(dias),
        "dias_baixados": baixados,
        "dias_por_replay": replays,
        "dias_com_falha": len(falhas),
        "pontos_com_valor": int(quadro.mrms_qpe_24h_mm.notna().sum())
                            if not quadro.empty else 0,
        "fracao_da_variancia_entre_lugares_no_mesmo_dia": intra,
        "sobre_a_fracao_acima": (
            "e descritiva da aquisicao e NAO serve de comparacao com o "
            "ERA5-Land: bases diferentes nao se comparam, e no agregado a "
            "fracao pune o produto certo, porque em dia seco o radar ve zero em "
            "toda parte. A comparacao correta, nas mesmas linhas e separando dia "
            "de chuva, esta em `mrms_vs_era5.json` "
            "(scripts/chuva/mrms_vs_era5_nyc.py)"),
        "primeiras_falhas": falhas[:20],
        "saida": str(destino.relative_to(REPO)) if not quadro.empty else None,
        "nao_e": (
            "nao e medida direta de lamina: e estimativa de radar corrigida por "
            "pluviometro. Cobre apenas os Estados Unidos, entao nao existe "
            "equivalente deste produto para as regioes brasileiras"),
        "nao_entrou_em_modelo": True,
    }
    (OUT / "manifesto.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/chuva/adquirir_mrms_nyc.py --online\n", encoding="utf-8")

    print(f"\ndias baixados {baixados} | replay {replays} | falhas {len(falhas)}")
    if intra is not None:
        print(f"variancia entre lugares no mesmo dia: {intra} "
              f"(ERA5-Land a 11 km: 0,075)")
    print(f"GRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
