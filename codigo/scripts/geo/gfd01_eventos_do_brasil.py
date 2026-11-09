"""Global Flood Database: o negativo observado que existe em escala brasileira.

-------------------------------------------------------------------------
POR QUE ESTA FONTE, E O QUE NELA E OBSERVACAO
-------------------------------------------------------------------------

`GLOBAL_FLOOD_DB/MODIS_EVENTS/V1` tem 913 eventos, e as bandas dele carregam a
estrutura que torna um negativo OBSERVAVEL -- a mesma do CEMS, por outra via:

    flooded          a extensao inundada            -> positivo
    clear_views      observacoes limpas do MODIS    -> E O FOOTPRINT
    jrc_perm_water   agua permanente                -> fora das duas classes

A EXCLUSAO DE `jrc_perm_water` NAO E A DO PROJETO, e a diferenca foi medida.
Ver a secao homonima no fim deste cabecalho antes de confiar nela.

`flooded == 0` E `clear_views > 0` e literalmente "foi olhado e nao tinha
agua". Sem `clear_views` seria ausencia de registro, que e fundo.

E ha 19 eventos com `dfo_country = Brazil`, contra UMA unica ativacao CEMS
brasileira. Para um projeto cuja lacuna e o Brasil, essa razao e o argumento.

-------------------------------------------------------------------------
A EXCLUSAO DE AGUA PERMANENTE DESTA FONTE E MAIS FRACA QUE A DO PROJETO
-------------------------------------------------------------------------

Este modulo tira `jrc_perm_water` das duas classes e isso NAO basta. Medido em
12/09/2026, depois de derivar a cadeia canonica sob os pontos:

    tile        classe      n      occurrence mediana   >= 50
    S22W053     INUNDACAO   4.647  78                   99,5%
    S23W053     INUNDACAO   3.629  80                   99,9%
    S11W051     INUNDACAO   2.456   0                    1,8%
    S12W051     INUNDACAO   2.234   0                    2,6%
    S29W057     INUNDACAO   2.140   3                    7,0%
    S30W057     INUNDACAO   1.850   6                   15,4%

O `occurrence` do JRC e a fracao das observacoes 1984-2021 com agua. Nos dois
tiles do alto Parana o positivo do GFD tem mediana 79 e apenas 5,7% acima de
90: nao e agua aberta -- seria 95-100 --, e a VARZEA, terreno molhado em tres
de cada quatro observacoes.

A contaminacao e LOCAL e nao propriedade da fonte: quatro dos seis tiles
medidos tem positivo limpo, e o negativo observado esta limpo nos seis
(mediana 0). Os dois contaminados sao tambem os dois mais densos, e a causa e
a mesma: a varzea e onde o MODIS mais registra cheia, porque ela esta molhada
quase sempre.

O que isso custa se passar: pedir ao terreno que preveja a varzea ensina "a
varzea e a varzea". Da AUC alta e nao significa nada -- a mesma armadilha do
oraculo de lugar medida em `RAIZ_DO_FRACASSO_PLUVIAL`.

-------------------------------------------------------------------------
A FAIXA, E POR QUE ELA E DECLARADA ANTES
-------------------------------------------------------------------------

Em oito eventos brasileiros o negativo bruto soma 6,97 milhoes de km2 contra
52 mil de positivo -- razao de 200 para 1. Quase tudo e planalto seco a
centenas de quilometros do evento, e negativo trivial nao ensina: qualquer
modelo acerta que o topo de chapada nao alagou.

Restringindo a uma faixa em torno do inundado, a razao cai para 2,9-16,4 --
comparavel ao 5,94 do EMSR720, que e negativo escolhido por analista dentro de
uma AOI de risco.

A FAIXA PADRAO E 2 km e ela e PARAMETRO DECLARADO, nao ajuste. Mexer nela
depois de ver o resultado seria escolher o desenho que favorece; o manifesto
grava o valor usado e a razao que ele produziu.

-------------------------------------------------------------------------
A ESCALA E 250 m, E ISSO NAO SE DISFARCA
-------------------------------------------------------------------------

O MODIS entrega 250 m. A cadeia de terreno do projeto e 30 m. Amostrar terreno
de 30 m num ponto rotulado por mascara de 250 m reintroduz erro de posicao do
rotulo -- que e exatamente o defeito diagnosticado no pluvial, onde o chamado
geocodificado por endereco derruba a separacao.

Entao a amostragem e no PASSO DE 250 m, alinhada a propria grade do produto, e
o manifesto declara `resolucao_nativa_m = 250`. Quem consumir decide o que faz
com isso; o que nao pode e o numero viajar sem a etiqueta.

E os pontos saem do INTERIOR das classes: um ponto a menos de 250 m da borda
entre inundado e nao inundado esta dentro da incerteza da propria mascara.

-------------------------------------------------------------------------
A AMOSTRA E BALANCEADA POR TETO, E ISSO NAO E A PREVALENCIA
-------------------------------------------------------------------------

`stratifiedSample` recebe `numPoints` por classe, entao quase todo evento sai
com 4.000 e 4.000 -- e esse equilibrio e decisao de amostragem, nao propriedade
do territorio. A prevalencia real na faixa varia de 2,3% a 22,0% entre os
eventos brasileiros.

O `aud_desenho02` ja registrou o que acontece quando esse numero se perde: um
escore de logistica herda a prevalencia do conjunto de ajuste, e quem ler o
valor absoluto le uma escolha de quem amostrou.

Por isso `--todos` mede e grava `areas_na_faixa` -- positivo e negativo em km2
por evento --, para que o nivel absoluto possa ser recuperado por
reponderacao. Sem esse campo a amostra serviria so para ordenar.

-------------------------------------------------------------------------
O QUE ESTE MODULO NAO FAZ
-------------------------------------------------------------------------

Nao deriva terreno -- os pontos saem sem as quatro variaveis, e sem elas o
`ds05` os recusa por `cadeia_global_nao_comparavel`. Isso e deliberado: o
terreno tem de vir da cadeia canonica, e derivar 250 mil km2 a 30 m e uma
decisao de custo que nao se toma dentro de um extrator.

Nao promove rotulo, nao treina e nao entra no `ds01` sozinho.

Uso:
    python scripts/geo/gfd01_eventos_do_brasil.py --listar
    python scripts/geo/gfd01_eventos_do_brasil.py --evento 3291 --baixar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "servico"))

from proveniencia import agora_utc, sha256_json  # noqa: E402

COLECAO = "GLOBAL_FLOOD_DB/MODIS_EVENTS/V1"
SAIDA = REPO / "local_runs" / "gfd-brasil"
FAIXA_M = 2000.0
PASSO_M = 250.0
MARGEM_DA_BORDA_M = 250.0
LICENCA = ("Global Flood Database v1 (Tellman et al. 2021), "
           "CC BY 4.0 -- uso com atribuicao")


def _colecao(pais: str = "Brazil"):
    import ee

    ee.Initialize()
    return (ee.ImageCollection(COLECAO)
            .filter(ee.Filter.eq("dfo_country", pais)))


def listar(pais: str = "Brazil") -> dict:
    col = _colecao(pais)
    info = col.getInfo()
    eventos = []
    for f in info["features"]:
        p = f["properties"]
        eventos.append({
            "id_ee": f["id"], "id": p.get("id"),
            "pais": p.get("dfo_country"),
            "causa": p.get("dfo_main_cause"),
            "mortos": p.get("dfo_dead"),
            "deslocados": p.get("dfo_displaced"),
            "began": p.get("dfo_began"), "ended": p.get("dfo_ended"),
        })
    return {"tipo": "catalogo_gfd", "quando_utc": agora_utc(),
            "colecao": COLECAO, "pais": pais, "n": len(eventos),
            "eventos": eventos, "licenca": LICENCA}


# OS EVENTOS QUE TOCAM O BRASIL, e nao so os rotulados Brasil.
#
# O `br02` mediu em 2026-09: dos 913 eventos do GFD, 87 tem footprint sobre o
# territorio brasileiro e so 19 carregam `dfo_country = Brazil`. O DFO rotula
# o evento pelo pais do registro de desastre, e nao pela agua: uma cheia do
# Parana registrada na Argentina inunda -- e OBSERVA seco -- dentro do Brasil
# tambem. Este modulo so amostrava os 19, e o negativo observado brasileiro
# desses outros 68 eventos nunca virou ponto. Aqui ele vira, RECORTADO ao
# territorio brasileiro: o que e de fora continua de fora.
EVENTOS_QUE_TOCAM = (REPO / "local_runs" / "br-02-negativo-nacional"
                     / "eventos_que_tocam_brasil.csv")
LIMITE_PAIS = "USDOS/LSIB_SIMPLE/2017"


def _territorio_brasileiro():
    import ee

    return (ee.FeatureCollection(LIMITE_PAIS)
            .filter(ee.Filter.eq("country_na", "Brazil")).geometry())


def classes_do_rotulo(img, esc: float, faixa_m: float = FAIXA_M):
    """(positivo interior, negativo util na faixa) -- a mascara de rotulo do projeto para o GFD.

    Extraida de `amostrar` sem mudar nada (04/10/2026), para que a selecao de
    eventos novos meca disponibilidade de classe com a MESMA regra.
    """
    perm = img.select("jrc_perm_water").eq(1)
    pos = img.select("flooded").eq(1).And(perm.Not())
    obs = img.select("clear_views").gt(0).And(perm.Not())
    neg = obs.And(pos.Not())

    # A DISTANCIA FAZ DUAS COISAS, e por isso ela e calculada uma vez so.
    # Restringe o negativo a faixa util, e afasta os dois lados da borda --
    # ponto a menos de 250 m da fronteira esta dentro da incerteza da mascara.
    d_pos = pos.fastDistanceTransform(512).sqrt().multiply(esc)
    d_neg = neg.fastDistanceTransform(512).sqrt().multiply(esc)
    pos_interior = pos.And(d_neg.gt(MARGEM_DA_BORDA_M))
    neg_util = neg.And(d_pos.lt(faixa_m)).And(d_pos.gt(MARGEM_DA_BORDA_M))
    return pos_interior, neg_util


def amostrar(id_evento: int, faixa_m: float = FAIXA_M,
             passo_m: float = PASSO_M, limite: int = 8000,
             recortar_ao_brasil: bool = False) -> dict:
    """Pontos do interior das classes, dentro da faixa, no passo de 250 m.

    `recortar_ao_brasil` busca o evento na colecao INTEIRA e amostra so a
    parte dele dentro do Brasil. Sem ele o caminho e o de sempre, byte a
    byte, para os 19 eventos ja amostrados nao mudarem de ponto.
    """
    import ee

    ee.Initialize()
    col = ee.ImageCollection(COLECAO) if recortar_ao_brasil else _colecao()
    img = ee.Image(col.filter(ee.Filter.eq("id", id_evento)).first())
    geo = img.geometry()
    if recortar_ao_brasil:
        geo = geo.intersection(_territorio_brasileiro(), maxError=1000)
    esc = img.select("flooded").projection().nominalScale().getInfo()

    # `jrc_perm_water` E MAIS ESTRITO QUE A MASCARA DO PROJETO.
    #
    # Medido em 12/09/2026 com a cadeia derivada: dos 8.276 pontos INUNDACAO
    # dos dois tiles do alto Parana que sobreviveram a esta exclusao, 99,7%
    # caem sobre `agua_permanente_jrc` -- o corte `occurrence >= 50` calibrado
    # contra o `JRCWaterHand` do Sen1Floods11. A mediana de `occurrence` deles
    # e 79 e so 5,7% passa de 90: nao e reservatorio, e varzea.
    #
    # As duas definicoes sao defensaveis, e por isso este modulo NAO muda a
    # dele: quem reconfere a 30 m e declara e o `gfd02`, que grava a
    # `occurrence` continua por ponto para que o corte seja reversivel.
    pos_interior, neg_util = classes_do_rotulo(img, esc, faixa_m)

    rotulo = (ee.Image(0).where(neg_util, 1).where(pos_interior, 2)
              .rename("classe_gfd").selfMask())
    amostra = rotulo.addBands(ee.Image.pixelLonLat()).stratifiedSample(
        numPoints=limite // 2, classBand="classe_gfd", region=geo,
        scale=max(passo_m, esc), projection=img.select("flooded").projection(),
        seed=20260911, geometries=False, tileScale=4)

    # `getDownloadURL` E NAO `getInfo`. O segundo aborta com "Collection query
    # aborted after accumulating over 5000 elements", e baixar 5.000 pontos por
    # evento limitaria a amostra por um detalhe de transporte -- nao por
    # desenho. O CSV vem inteiro.
    import csv as _csv
    import io as _io
    import urllib.request as _url

    endereco = amostra.getDownloadURL(
        filetype="CSV", selectors=["classe_gfd", "latitude", "longitude"])
    with _url.urlopen(endereco, timeout=900) as r:
        texto = r.read().decode("utf-8", errors="replace")
    linhas = list(_csv.DictReader(_io.StringIO(texto)))

    pontos = []
    for i, p in enumerate(linhas):
        if not p.get("latitude"):
            continue
        pontos.append({
            "ponto_id": f"GFD{id_evento}_{i:06d}",
            "classe_obs": ("INUNDACAO" if int(float(p["classe_gfd"])) == 2
                           else "NAO_INUNDADO_OBSERVADO"),
            "lat": float(p["latitude"]), "lon": float(p["longitude"]),
            "evento_gfd": id_evento,
            "grupo_cv": f"GFD{id_evento}",
            "resolucao_m": esc, "fonte": "global_flood_database",
            "licenca": LICENCA,
        })
    return {"evento": id_evento, "pontos": pontos, "escala_nativa_m": esc,
            "faixa_m": faixa_m, "passo_m": max(passo_m, esc)}


def amostrar_os_que_tocam(faixa_m: float = FAIXA_M, refazer: bool = False) -> int:
    """Os eventos da lista do `br02`, recortados ao Brasil, um CSV por evento.

    Evento ja amostrado -- os 19 rotulados Brasil incluidos -- e pulado: o
    CSV dele continua sendo o do caminho original.
    """
    import csv

    import pandas as pd

    if not EVENTOS_QUE_TOCAM.exists():
        raise SystemExit(f"sem a lista de eventos: rode o br02 ({EVENTOS_QUE_TOCAM})")
    from concurrent.futures import ThreadPoolExecutor

    lista = pd.read_csv(EVENTOS_QUE_TOCAM)
    feitos, falhas, pulados = [], [], []
    a_fazer = []
    for ev, pais in zip(lista["id"], lista["pais"]):
        if (SAIDA / f"pontos_gfd{int(ev)}.csv").exists() and not refazer:
            pulados.append(int(ev))
        else:
            a_fazer.append((int(ev), pais))

    # Cada evento e um pedido independente ao Earth Engine, e o tempo e
    # quase todo espera do servidor (3-4 min por evento medidos em 30/09):
    # em serie, 81 eventos custavam ~4 h. O resultado de cada um nao depende
    # da ordem -- a semente e fixa por evento.
    def _um(par):
        ev, pais = par
        try:
            return ev, pais, amostrar(ev, faixa_m, recortar_ao_brasil=True), None
        except Exception as e:                                   # noqa: BLE001
            return ev, pais, None, e

    with ThreadPoolExecutor(max_workers=4) as pool:
        resultados = list(pool.map(_um, a_fazer))

    for i, (ev, pais, r, erro) in enumerate(resultados, start=1):
        alvo = SAIDA / f"pontos_gfd{ev}.csv"
        if erro is not None:
            falhas.append({"evento": ev, "pais": pais, "erro": type(erro).__name__,
                           "detalhe": str(erro)[:160]})
            print(f"  {i}/{len(a_fazer)} evento {ev} ({pais}): FALHA {erro}",
                  file=sys.stderr, flush=True)
            continue
        c: dict = {}
        for x in r["pontos"]:
            c[x["classe_obs"]] = c.get(x["classe_obs"], 0) + 1
        if not r["pontos"]:
            # Tocar o Brasil nao garante as duas classes dentro dele: evento
            # que so observou seco aqui, ou so agua, fica registrado como tal.
            falhas.append({"evento": ev, "pais": pais,
                           "por_que": "sem ponto na faixa dentro do Brasil"})
        else:
            with alvo.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(r["pontos"][0]))
                w.writeheader()
                w.writerows(r["pontos"])
            feitos.append({"evento": ev, "pais_do_registro": pais,
                           "pontos": len(r["pontos"]), "por_classe": c})
        print(f"  {i}/{len(a_fazer)} evento {ev} ({pais}): {c or 'sem ponto'}",
              file=sys.stderr, flush=True)
    resumo = {
        "tipo": "amostragem_gfd_eventos_que_tocam_o_brasil",
        "quando_utc": agora_utc(), "colecao": COLECAO, "licenca": LICENCA,
        "limite_do_pais": LIMITE_PAIS, "faixa_m": faixa_m, "passo_m": PASSO_M,
        "eventos_na_lista": int(len(lista)), "ja_amostrados": pulados,
        "amostrados_agora": feitos, "falhas": falhas,
        "pontos_novos": sum(x["pontos"] for x in feitos),
        "nao_e": ("nao e evento brasileiro por declaracao do DFO: e a parte "
                  "BRASILEIRA de eventos registrados em qualquer pais. Nao deriva "
                  "terreno, nao promove rotulo e nao entra no ds01 sozinho"),
    }
    resumo["hash_resumo"] = sha256_json(
        {k: v for k, v in resumo.items() if k != "quando_utc"})
    (SAIDA / "lote_que_tocam_o_brasil.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in resumo.items()
                      if k not in ("amostrados_agora", "ja_amostrados")},
                     indent=2, ensure_ascii=False))
    print("===END===")
    return 0


def main() -> int:
    import csv

    p = argparse.ArgumentParser()
    p.add_argument("--listar", action="store_true")
    p.add_argument("--evento", type=int)
    p.add_argument("--todos", action="store_true",
                   help="amostra todos os eventos do pais")
    p.add_argument("--que-tocam-o-brasil", action="store_true",
                   help=("amostra, recortado ao Brasil, cada evento do GFD cujo "
                         "footprint toca o territorio (lista do br02)"))
    p.add_argument("--faixa-m", type=float, default=FAIXA_M)
    p.add_argument("--baixar", action="store_true")
    p.add_argument("--refazer", action="store_true")
    a = p.parse_args()

    SAIDA.mkdir(parents=True, exist_ok=True)
    if a.que_tocam_o_brasil:
        return amostrar_os_que_tocam(a.faixa_m, a.refazer)
    if a.listar:
        r = listar()
        r["hash_resumo"] = sha256_json(
            {k: v for k, v in r.items() if k != "quando_utc"})
        (SAIDA / "catalogo.json").write_text(
            json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"eventos no Brasil: {r['n']}")
        for e in r["eventos"]:
            print(f"  {e['id']:>6}  {str(e['causa'])[:26]:<28} "
                  f"mortos={e['mortos']}  deslocados={e['deslocados']}")
        print(f"\nGRAVADO={SAIDA / 'catalogo.json'}")
        return 0

    if a.todos:
        cat = listar()
        ids = [e["id"] for e in cat["eventos"] if e.get("id")]
        feitos, falhas = [], []
        for i, ev in enumerate(ids, start=1):
            alvo = SAIDA / f"pontos_gfd{ev}.csv"
            if alvo.exists() and not a.refazer:
                feitos.append({"evento": ev, "pulado": True})
                continue
            try:
                r = amostrar(ev, a.faixa_m)
                if not r["pontos"]:
                    falhas.append({"evento": ev, "por_que": "sem ponto na faixa"})
                    continue
                with alvo.open("w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(r["pontos"][0]))
                    w.writeheader()
                    w.writerows(r["pontos"])
                c = {}
                for x in r["pontos"]:
                    c[x["classe_obs"]] = c.get(x["classe_obs"], 0) + 1
                feitos.append({"evento": ev, "pontos": len(r["pontos"]),
                               "por_classe": c})
            except Exception as e:                               # noqa: BLE001
                falhas.append({"evento": ev, "erro": type(e).__name__,
                               "detalhe": str(e)[:160]})
            print(f"  {i}/{len(ids)} evento {ev}", file=sys.stderr, flush=True)
        resumo = {"tipo": "amostragem_gfd_em_lote", "quando_utc": agora_utc(),
                  "colecao": COLECAO, "licenca": LICENCA,
                  "faixa_m": a.faixa_m, "passo_m": PASSO_M,
                  "margem_da_borda_m": MARGEM_DA_BORDA_M,
                  "eventos": feitos, "falhas": falhas,
                  "pontos_totais": sum(x.get("pontos", 0) for x in feitos)}
        resumo["hash_resumo"] = sha256_json(
            {k: v for k, v in resumo.items() if k != "quando_utc"})
        (SAIDA / "lote.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: v for k, v in resumo.items() if k != "eventos"},
                         indent=2, ensure_ascii=False))
        return 0

    if not a.evento:
        raise SystemExit("informe --listar, --evento ou --todos")
    if not a.baixar:
        print(f"DRY RUN: amostraria o evento {a.evento} na faixa de "
              f"{a.faixa_m:.0f} m, passo {PASSO_M:.0f} m. Use --baixar.")
        return 0

    r = amostrar(a.evento, a.faixa_m)
    pontos = r["pontos"]
    if not pontos:
        raise SystemExit(f"evento {a.evento} nao produziu ponto na faixa")
    csv_path = SAIDA / f"pontos_gfd{a.evento}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pontos[0]))
        w.writeheader()
        w.writerows(pontos)

    por_classe = {}
    for x in pontos:
        por_classe[x["classe_obs"]] = por_classe.get(x["classe_obs"], 0) + 1
    manifesto = {
        "tipo": "pontos_observados_do_global_flood_database",
        "quando_utc": agora_utc(), "colecao": COLECAO,
        "evento": a.evento, "licenca": LICENCA,
        "pontos": len(pontos), "por_classe": por_classe,
        "escala_nativa_m": r["escala_nativa_m"],
        "passo_da_amostragem_m": r["passo_m"],
        "faixa_em_torno_do_inundado_m": r["faixa_m"],
        "margem_da_borda_m": MARGEM_DA_BORDA_M,
        "definicao": {
            "positivo": "flooded == 1, fora da agua permanente",
            "negativo": ("clear_views > 0 E flooded == 0, fora da agua "
                         "permanente, dentro da faixa"),
            "footprint": "clear_views > 0 -- o que o MODIS conseguiu observar",
        },
        "por_que_a_faixa": (
            "sem ela o negativo e 200 vezes o positivo e quase todo planalto "
            "seco longe do evento; a faixa traz a razao para a ordem de "
            "grandeza do negativo escolhido por analista no CEMS"),
        "por_que_a_margem": (
            "ponto a menos de 250 m da fronteira entre as classes esta dentro "
            "da incerteza da propria mascara"),
        "escala_declarada": (
            "250 m, contra os 30 m da cadeia de terreno. Amostrar terreno fino "
            "em rotulo grosso reintroduz erro de posicao -- o defeito "
            "diagnosticado no pluvial"),
        "sem_terreno": (
            "os pontos saem SEM as quatro variaveis; sem elas o ds05 os recusa "
            "por cadeia_global_nao_comparavel. E deliberado: o terreno tem de "
            "vir da cadeia canonica, e derivar a area destes eventos a 30 m e "
            "decisao de custo que nao se toma dentro de um extrator"),
        "nao_e": ("nao promove rotulo, nao treina e nao entra no ds01 "
                  "sozinho: e a amostra observada de um evento"),
        "arquivo": str(csv_path),
    }
    manifesto["hash_resumo"] = sha256_json(
        {k: v for k, v in manifesto.items() if k != "quando_utc"})
    (SAIDA / f"resumo_gfd{a.evento}.json").write_text(
        json.dumps(manifesto, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifesto, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
