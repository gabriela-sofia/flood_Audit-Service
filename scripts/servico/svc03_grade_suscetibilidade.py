"""
SVC-03 -- Grade de suscetibilidade por regiao (E5/M4).

O QUE O E5 PEDE: "levar o modelo as tres regioes e caracterizar a distancia
entre o dominio de treino e cada uma. Entregavel: mapa de suscetibilidade por
regiao, com maturidade declarada. Evidencia: nenhuma afirmacao de acerto onde
nao ha inventario local."

O QUE FALTAVA, e ja nao falta: uma grade. Ate aqui o projeto so tinha os pontos
ROTULADOS de cada regiao -- 278 em Recife, 1.680 em Curitiba, zero em
Petropolis. Mapa exige varrer o territorio, nao os pontos que alguem reportou.
A cadeia de terreno harmonizada ja cobre as tres regioes a 30 m, na mesma
convencao (D-infinity, canal de 0,1123 km2, WhiteboxTools 2.4.0), entao a grade
sai do que ja existe: nenhuma aquisicao nova.

-------------------------------------------------------------------------
A DECISAO QUE O E5 RESOLVEU
-------------------------------------------------------------------------

Petropolis estava em impasse: o esboco de telas declarava
`region_not_supported`, e `ext_criterios_de_acerto_v1.md` secao 6 dizia que
"para PREDIZER em Petropolis nao falta nada; o que falta e a validacao". O
proprio E5 desempata -- ele manda levar o modelo as TRES regioes, e a evidencia
que exige nao e acerto, e sim que nao se afirme acerto onde falta inventario.

Entao Petropolis entra, com maturidade `transferencia_sem_referencia_local`:
predicao por semelhanca de terreno, sem nenhum ponto rotulado para verificar.
A distincao em relacao a Curitiba e deliberada -- Curitiba tem inventario que o
projeto decidiu nao usar como criterio; Petropolis nao tem nem isso.

-------------------------------------------------------------------------
DESENHO, declarado antes de rodar
-------------------------------------------------------------------------

GRADE: subamostragem regular do raster de 30 m, um pixel a cada PASSO. Com
PASSO=4 a grade fica a 120 m. Nao e suavizacao nem reamostragem: e o mesmo
pixel de 30 m, tomado de PASSO em PASSO, para que cada celula da grade seja um
valor derivado e nao uma media inventada.

CELULA VALIDA: exige as quatro variaveis de terreno presentes e finitas. Celula
fora do raster derivado -- borda, corpo d'agua permanente -- fica de fora e e
contada, nao preenchida.

MODELO: o que o contrato ja mapeia para a regiao. Recife usa o pluvial de seis
variaveis; Curitiba e Petropolis usam os fluviais de terreno.

CHUVA, para Recife: a chuva nao e propriedade do lugar na escala desta grade --
Recife inteiro cabe em 4 celulas de 0,1 grau do produto de precipitacao
(`ext_chuva_estado_do_projeto_v1.md`). Entao ela entra como CENARIO declarado, e
nao como camada: tres niveis tirados da propria distribuicao dos eventos
observados em Recife (mediana, p90 e maximo). Isso e literalmente a definicao de
suscetibilidade do projeto -- predisposicao do terreno SOB UM DADO FORCAMENTO.
Como a chuva e constante na grade, ela desloca o escore inteiro e nao muda o
ordenamento: o mapa ordena terreno, e o cenario diz em que nivel ele opera.
Isso esta escrito na saida para nao ser lido como se a chuva variasse no espaco.

IC POR CELULA: mesmo mecanismo do contrato -- as replicas de bootstrap gravadas
pelo SVC-01 projetam um escore por celula, e o IC e o percentil. Calculado em
blocos para nao materializar uma matriz de celulas x replicas inteira.

DISTANCIA DE DOMINIO: por variavel, a diferenca padronizada da media da grade
contra a media do ajuste, e o percentual de celulas dentro da faixa 5-95% do
ajuste. E o entregavel explicito do E5, e o mesmo diagnostico de
`metodo_aplicacao_sem_rotulo_local_v1.md`, agora sobre o territorio e nao sobre
os pontos rotulados.

O MAPA E O CONTRATO, e isso vale nos dois sentidos:

  (a) o portao de dominio (G5) e aplicado CELULA A CELULA. Celula em que
      variaveis demais caem fora da faixa 5-95% do ajuste nao recebe escore --
      fica vazia no mapa e e contada a parte. Sem isso o mapa mostraria numero
      onde o servico recusaria responder, e um mapa que responde mais que o
      contrato e um mapa que mente;
  (b) ao final, N celulas sorteadas passam por `inferir()` uma a uma e o escore
      tem de bater com o da grade.

O quanto de cada regiao sobra depois de (a) e resultado, nao detalhe de
implementacao: e a medida de sobre quanto do territorio o modelo pode falar.

NAO FAZ: nao valida contra rotulo local, nao afirma acerto, nao promove
maturidade e nao publica raster em diretorio versionado -- tudo vai para
`local_runs/`.

Uso:
    python scripts/servico/svc03_grade_suscetibilidade.py
    python scripts/servico/svc03_grade_suscetibilidade.py --passo 8
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts" / "suscetibilidade"))
from ds03_esquema_alvo import VERSAO  # noqa: E402
from svc02_contrato_inferencia import (  # noqa: E402
    MAX_PCT_EXTRAPOLACAO, carregar_modelos, inferir, maturidade_da_regiao,
    modelo_da_regiao, regioes_com_referencia_local,
)

RUNS = REPO / "local_runs"
TERRENO = RUNS / "ter-01-cadeia-harmonizada"
UNICA = RUNS / "ds-05-tabela-unica" / f"tabela_unica_{VERSAO}.csv"
OUT = RUNS / "svc-03-grade"

REGIOES = {
    "recife": "recife_harmonizado",
    "curitiba": "curitiba_harmonizado",
    "petropolis": "petropolis_harmonizado",
}
RASTER_DE = {
    "elevation_m": "dem_filled.tif",
    "slope_deg": "slope_deg_wbt.tif",
    "hand_m": "hand_dinf.tif",
    "twi_dinf": "twi_dinf.tif",
}
PASSO = 4                    # 1 pixel a cada 4 -> grade de 120 m
BLOCO_IC = 20000             # celulas por bloco no calculo do IC
N_CONFERENCIA = 25           # celulas sorteadas para bater contra inferir()
SEMENTE = 20260820

# Cenarios de chuva para Recife, da distribuicao dos eventos observados na
# propria regiao (n=145 positivos na tabela unica).
CENARIOS_CHUVA = {
    "mediana_observada": {"rain_max_24h": 13.6, "rain_decay_index": 23.18},
    "p90_observado": {"rain_max_24h": 34.4, "rain_decay_index": 54.45},
    "maximo_observado": {"rain_max_24h": 100.2, "rain_decay_index": 115.22},
}


def ler_grade(dir_regiao: Path, passo: int) -> dict:
    """Le as quatro variaveis de terreno na mesma grade, subamostrada."""
    import rasterio
    from rasterio.transform import xy

    dados, perfil, transform, crs = {}, None, None, None
    for var, arquivo in RASTER_DE.items():
        p = dir_regiao / arquivo
        if not p.exists():
            raise FileNotFoundError(p)
        with rasterio.open(p) as src:
            a = src.read(1, masked=True).filled(np.nan)[::passo, ::passo]
            dados[var] = a.astype("float64")
            if perfil is None:
                perfil = src.profile.copy()
                transform = src.transform * src.transform.scale(passo, passo)
                crs = src.crs
    forma = next(iter(dados.values())).shape
    linhas, colunas = np.indices(forma)
    xs, ys = xy(transform, linhas.ravel(), colunas.ravel())
    return {"dados": dados, "forma": forma, "transform": transform,
            "crs": crs, "perfil": perfil,
            "x": np.array(xs), "y": np.array(ys)}


def para_lat_lon(x: np.ndarray, y: np.ndarray, crs) -> tuple[np.ndarray, np.ndarray]:
    from pyproj import Transformer

    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = t.transform(x, y)
    return np.asarray(lat), np.asarray(lon)


def dominio_por_celula(X: np.ndarray, modelo: dict) -> np.ndarray:
    """G5 aplicado celula a celula: True quando a celula e servivel.

    Mesma regra do contrato -- se mais que MAX_PCT_EXTRAPOLACAO das variaveis
    cai fora da faixa 5-95% do ajuste, o servico recusa. O mapa recusa junto.
    """
    feats = modelo["features"]
    faixa = modelo["faixa_dominio_5_95"]
    fora = np.zeros(len(X), dtype=float)
    for j, f in enumerate(feats):
        lo, hi = faixa[f]
        fora += ((X[:, j] < lo) | (X[:, j] > hi)).astype(float)
    return (100.0 * fora / len(feats)) <= MAX_PCT_EXTRAPOLACAO


def escores_em_bloco(Z: np.ndarray, modelo: dict) -> tuple[np.ndarray, np.ndarray,
                                                           np.ndarray]:
    """Escore e IC95 por celula, em blocos para nao estourar memoria."""
    beta = np.array([modelo["coef"][f] for f in modelo["features"]])
    eta = modelo["intercepto"] + Z @ beta
    p = 1.0 / (1.0 + np.exp(-eta))

    rep = np.array(modelo.get("replicas_bootstrap") or [])
    lo = np.full(len(p), np.nan)
    hi = np.full(len(p), np.nan)
    if rep.size:
        for i in range(0, len(p), BLOCO_IC):
            z = Z[i:i + BLOCO_IC]
            etas = rep[:, 0][None, :] + z @ rep[:, 1:].T
            ps = 1.0 / (1.0 + np.exp(-etas))
            lo[i:i + BLOCO_IC], hi[i:i + BLOCO_IC] = np.percentile(
                ps, [2.5, 97.5], axis=1)
    return p, lo, hi


def distancia_de_dominio(X: np.ndarray, modelo: dict) -> dict:
    """Entregavel explicito do E5: quao longe a regiao esta do que o modelo viu."""
    feats = modelo["features"]
    faixa = modelo["faixa_dominio_5_95"]
    d = {}
    for j, f in enumerate(feats):
        mu = modelo["padronizacao"][f]["media"]
        sd = modelo["padronizacao"][f]["desvio"]
        lo, hi = faixa[f]
        col = X[:, j]
        d[f] = {
            "media_da_grade": round(float(np.mean(col)), 3),
            "media_do_ajuste": round(float(mu), 3),
            "diferenca_padronizada": round(float((np.mean(col) - mu) / sd), 3),
            "faixa_do_ajuste_5_95": [lo, hi],
            "pct_dentro_da_faixa": round(
                100.0 * float(np.mean((col >= lo) & (col <= hi))), 1),
        }
    return d


def gravar_raster(caminho: Path, valores: np.ndarray, forma, transform, crs) -> None:
    import rasterio

    with rasterio.open(caminho, "w", driver="GTiff", height=forma[0],
                       width=forma[1], count=1, dtype="float32",
                       crs=crs, transform=transform, nodata=np.nan,
                       compress="deflate") as dst:
        dst.write(valores.reshape(forma).astype("float32"), 1)


def gravar_figura(caminho: Path, valores: np.ndarray, forma, titulo: str,
                  rodape: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 7 * forma[0] / max(forma[1], 1)))
    im = ax.imshow(valores.reshape(forma), cmap="viridis", vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_title(titulo, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, shrink=0.75, label="escore de suscetibilidade")
    fig.text(0.5, 0.02, rodape, ha="center", fontsize=7, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(caminho, dpi=130)
    plt.close(fig)


def processar(regiao: str, modelos: dict, com_ref: set, passo: int) -> dict:
    import pandas as pd

    dir_regiao = TERRENO / REGIOES[regiao]
    modelo = modelo_da_regiao(regiao, modelos)
    print(f"\n=== {regiao} ===")
    if modelo is None:
        print("  sem modelo mapeado: region_not_supported, nenhuma grade gerada")
        return {"regiao": regiao, "status": "region_not_supported",
                "maturidade": "nao_suportada", "grade_gerada": False,
                "motivo": "nenhum modelo servivel mapeado para a regiao"}
    if not dir_regiao.exists():
        print(f"  sem cadeia de terreno em {dir_regiao}: grade impossivel")
        return {"regiao": regiao, "status": "insufficient_data",
                "gate_que_nao_fechou": "variaveis_presentes",
                "grade_gerada": False,
                "motivo": "regiao sem cadeia de terreno derivada"}

    g = ler_grade(dir_regiao, passo)
    feats = modelo["features"]
    terreno = [f for f in feats if f in RASTER_DE]
    faltantes = [f for f in feats if f not in RASTER_DE]

    X_terr = np.column_stack([g["dados"][f].ravel() for f in terreno])
    valido = np.all(np.isfinite(X_terr), axis=1)
    n_total, n_valido = len(valido), int(valido.sum())
    print(f"  grade {g['forma']} a {30 * passo} m -> {n_total:,} celulas, "
          f"{n_valido:,} com as {len(terreno)} variaveis de terreno "
          f"({100 * n_valido / n_total:.1f}%)")

    lat, lon = para_lat_lon(g["x"][valido], g["y"][valido], g["crs"])
    maturidade = maturidade_da_regiao(regiao, modelo, com_ref)
    print(f"  modelo={modelo['nome']}  maturidade={maturidade}")
    if faltantes:
        print(f"  variaveis fora do raster, entram como cenario: {faltantes}")

    cenarios = (CENARIOS_CHUVA if faltantes else {"unico": {}})
    saidas, resumo_cenarios = {}, {}
    for nome_cen, valores_cen in cenarios.items():
        colunas = []
        for f in feats:
            if f in terreno:
                colunas.append(X_terr[valido, terreno.index(f)])
            else:
                colunas.append(np.full(n_valido, float(valores_cen[f])))
        X = np.column_stack(colunas)
        servivel = dominio_por_celula(X, modelo)
        mu = np.array([modelo["padronizacao"][f]["media"] for f in feats])
        sd = np.array([modelo["padronizacao"][f]["desvio"] for f in feats])
        p, lo, hi = escores_em_bloco((X - mu) / sd, modelo)
        p[~servivel] = np.nan
        lo[~servivel] = np.nan
        hi[~servivel] = np.nan
        saidas[nome_cen] = {"X": X, "p": p, "lo": lo, "hi": hi,
                            "servivel": servivel}
        n_serv = int(servivel.sum())
        resumo_cenarios[nome_cen] = {
            "cenario_de_chuva": valores_cen or None,
            "celulas_serviveis": n_serv,
            "celulas_recusadas_por_dominio": int((~servivel).sum()),
            "pct_servivel": round(100.0 * n_serv / len(servivel), 1),
            "escore_mediano": round(float(np.nanmedian(p)), 4) if n_serv else None,
            "escore_p90": round(float(np.nanpercentile(p, 90)), 4) if n_serv else None,
            "escore_minimo": round(float(np.nanmin(p)), 4) if n_serv else None,
            "escore_maximo": round(float(np.nanmax(p)), 4) if n_serv else None,
            "largura_mediana_do_ic": (round(float(np.nanmedian(hi - lo)), 4)
                                      if n_serv else None),
        }
        rc = resumo_cenarios[nome_cen]
        print(f"  cenario {nome_cen:20s} servivel={rc['pct_servivel']:5.1f}%  "
              f"escore mediano={rc['escore_mediano']}  p90={rc['escore_p90']}  "
              f"IC mediano={rc['largura_mediana_do_ic']}")

    principal = next(iter(saidas))
    dom = distancia_de_dominio(saidas[principal]["X"], modelo)
    print("  distancia de dominio:")
    for f, d in dom.items():
        print(f"    {f:18s} dif.padronizada={d['diferenca_padronizada']:+7.3f}  "
              f"dentro da faixa={d['pct_dentro_da_faixa']:5.1f}%")

    OUT.mkdir(parents=True, exist_ok=True)
    tab = pd.DataFrame({"lat": np.round(lat, 6), "lon": np.round(lon, 6)})
    for j, f in enumerate(terreno):
        tab[f] = np.round(X_terr[valido, j], 4)
    for nome_cen, s in saidas.items():
        suf = "" if nome_cen == "unico" else f"__{nome_cen}"
        tab[f"escore{suf}"] = np.round(s["p"], 4)
        tab[f"ic95_lo{suf}"] = np.round(s["lo"], 4)
        tab[f"ic95_hi{suf}"] = np.round(s["hi"], 4)
    tab.to_csv(OUT / f"grade_{regiao}.csv", index=False)

    plano = np.full(n_total, np.nan)
    plano[valido] = saidas[principal]["p"]
    gravar_raster(OUT / f"escore_{regiao}.tif", plano, g["forma"],
                  g["transform"], g["crs"])
    rodape = (f"{modelo['nome']} | maturidade {maturidade} | "
              f"{'cenario ' + principal if faltantes else 'sem chuva no modelo'} | "
              "escore de predisposicao do terreno, nao previsao de evento")
    gravar_figura(OUT / f"escore_{regiao}.png", plano, g["forma"],
                  f"Suscetibilidade -- {regiao} ({30 * passo} m)", rodape)

    # o mapa tem de ser a mesma coisa que o contrato responde
    rng = np.random.default_rng(SEMENTE)
    serviveis = np.flatnonzero(saidas[principal]["servivel"])
    idx = (rng.choice(serviveis, size=min(N_CONFERENCIA, len(serviveis)),
                      replace=False) if len(serviveis) else np.array([], dtype=int))
    X_p = saidas[principal]["X"]
    divergencias = 0
    for i in idx:
        req = {"regiao": regiao,
               "geometria": {"tipo": "ponto", "crs": "EPSG:4326",
                             "pontos": [{"lat": float(lat[i]), "lon": float(lon[i]),
                                         "camadas": {f: float(X_p[i, j])
                                                     for j, f in enumerate(feats)}}]}}
        r = inferir(req, modelos, {}, com_ref)
        if r["status"] != "ok" or abs(r["escore"] - saidas[principal]["p"][i]) > 1e-4:
            divergencias += 1
    print(f"  conferencia contra o contrato: {len(idx) - divergencias}/{len(idx)} "
          "celulas serviveis batem")

    return {
        "regiao": regiao, "status": "ok", "grade_gerada": True,
        "modelo": modelo["nome"], "maturidade": maturidade,
        "resolucao_m": 30 * passo,
        "celulas_da_grade": n_total, "celulas_validas": n_valido,
        "pct_validas": round(100.0 * n_valido / n_total, 1),
        "celulas_serviveis": int(saidas[principal]["servivel"].sum()),
        "pct_servivel_do_valido": round(
            100.0 * float(saidas[principal]["servivel"].mean()), 1),
        "variaveis_do_modelo": feats,
        "variaveis_do_raster": terreno,
        "variaveis_por_cenario": faltantes,
        "cenarios": resumo_cenarios,
        "distancia_de_dominio": dom,
        "conferencia_contra_o_contrato": {
            "celulas": int(len(idx)), "divergencias": int(divergencias)},
        "limitacoes": modelo.get("limitacoes_declaradas", []),
        "nao_e": ("nao e afirmacao de acerto: nenhuma destas regioes tem "
                  "inventario local usado como criterio de aprovacao"),
    }


def main() -> int:
    passo = PASSO
    if "--passo" in sys.argv:
        passo = int(sys.argv[sys.argv.index("--passo") + 1])
    t0 = time.time()
    print("===SVC-03: GRADE DE SUSCETIBILIDADE POR REGIAO (E5/M4)===")

    modelos = carregar_modelos()
    if not modelos:
        print("ABORTADO: modelos servidos ausentes. Gere com: "
              "python scripts/servico/svc01_construir_modelos_servidos.py")
        return 1
    com_ref = regioes_com_referencia_local()
    print(f"passo={passo} ({30 * passo} m)  regioes com referencia local: "
          f"{sorted(com_ref & set(REGIOES))}")

    OUT.mkdir(parents=True, exist_ok=True)
    resultados = [processar(r, modelos, com_ref, passo) for r in REGIOES]

    (OUT / "resultado.json").write_text(json.dumps({
        "passo": passo, "resolucao_m": 30 * passo, "semente": SEMENTE,
        "cenarios_de_chuva": CENARIOS_CHUVA,
        "por_que_chuva_e_cenario": (
            "a chuva nao varia na escala desta grade -- Recife inteiro cabe em 4 "
            "celulas do produto de precipitacao. Entrando como cenario, ela "
            "desloca o escore e nao muda o ordenamento: o mapa ordena terreno"),
        "regioes": resultados,
        "segundos": round(time.time() - t0, 1),
        "nao_e": ("nao e mapa operacional nem previsao de evento: e escore de "
                  "predisposicao do terreno com maturidade declarada por regiao"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "commands.txt").write_text(
        "python scripts/servico/svc01_construir_modelos_servidos.py\n"
        "python scripts/servico/svc03_grade_suscetibilidade.py\n"
        "python -m pytest tests/test_svc03_grade.py -q\n",
        encoding="utf-8")

    print("\n--- RESUMO ---")
    for r in resultados:
        if r.get("grade_gerada"):
            print(f"  {r['regiao']:12s} {r['celulas_validas']:>9,} celulas, "
                  f"{r['pct_servivel_do_valido']:5.1f}% serviveis  "
                  f"maturidade={r['maturidade']}")
        else:
            print(f"  {r['regiao']:12s} sem grade: {r['status']}")
    print(f"\nGRAVADO={OUT}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
