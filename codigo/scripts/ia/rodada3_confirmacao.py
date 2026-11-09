"""Rodada 3, parte 4: confirmacao externa retrospectiva dos pesos Q (Q_conexao - Q_base, fluvial) em ativacoes CEMS
historicas nao examinadas, com protocolo congelado ANTES de qualquer download.

Linhagem `confirmacao_rodada3_v1` (17/10/2026). Etapas (cada uma grava, confere a anterior e nao reescreve):

  protocolo   hipoteses, pesos (hash), padronizacao, codigo, entradas, populacao examinada (ledger), regras de
              selecao de produto, elegibilidade, admissao, grafo, metricas, agregacao e regra; data de corte
  adquirir    ate 2 produtos DEL por candidato (prioridade fixa por metadado), cabecalho antes do download,
              teto por pacote e total, retomada; ordem: fila inicial e depois alternativas da coordenacao
  geometria   le SO areaOfInterestA, imageFootprintA e _source (data da cena) -- camadas neutras; grafo MT01
              (evento, L8, onda MAIN_BAS 30 d, chuva 500 km/15 d) contra TUDO o que o ledger marca como examinado,
              reservado ou planejado; sensibilidade por exutorio. Sela a selecao antes de abrir observedEvent
  areal       le observedEventA dos selecionados: observado = AOI interseccao footprint, positivo = 5-Flood,
              negativo = observado - positivo - massa; processo por obj_desc; admissao por AOI
  terreno     FABDEM + cadeia canonica (derivar_aoi_nova) + JRC (pesado; rodar via executar_serializado)
  pontos      grade 150 m, JRC fora, piso local; atributos reais (cobertura GHSL/SoilGrids do ano, conexao,
              HydroSHEDS) pela mesma `mt03_atributos_novos.completar` usada no MT03/MT04
  avaliar     escores Q/A0 congelados sem retreino; AUC dentro do evento; media por componente; bootstrap

Uso:
    python scripts/ia/rodada3_confirmacao.py protocolo|adquirir|geometria|areal|terreno|pontos|avaliar
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "metodo", "servico", "geo", "terreno", "sar", "suscetibilidade"):
    sys.path.insert(0, str(REPO / "scripts" / s))
from proveniencia import agora_utc, sha256_arquivo, sha256_json  # noqa: E402
import rodada3_ledger as led  # noqa: E402

C3 = led.R3 / "confirmacao"
PACOTES, AREAL, MDT, TERRENO = (C3 / n for n in ("pacotes", "areal", "mdt", "terreno"))
PESOS_Q = REPO / "local_runs/mecanismos-transferiveis/v5/sem_amostra_v1/candidatas/pesos_e_padronizacao.json"
MODELOS_Q = PESOS_Q.parent / "modelos.pkl"
FILA = led.COORD / "fontes" / "fila_16_candidatos.json"
CAB = led.COORD / "fontes" / "cabecalhos_16_pacotes.json"
HOST = "cems-mapping-website.s3.eu-west-1.amazonaws.com"
TETO_PACOTE = 64 * 1024 ** 2
TETO_TOTAL = 1024 ** 3
PRODUTOS_POR_CANDIDATO = 2
ALVO_COMPONENTES = 16
MARGEM = 0.01
REPS, SEMENTE_BOOT = 2000, 7
MIN_POS, MIN_NEG, MIN_N = 10, 10, 20
CEMS_PROC = {"riverine flood": "fluvial", "flash flood": "enxurrada", "coastal flood": "costeira"}
CODIGO_CADEIA = ["scripts/ia/rodada3_confirmacao.py", "scripts/ia/rodada3_ledger.py",
                 "scripts/metodo/mt03_cems_desenvolvimento.py", "scripts/metodo/mt03_atributos_novos.py",
                 "scripts/metodo/mt05_transferencia_sem_amostra.py", "scripts/metodo/mt01_independencia.py",
                 "scripts/metodo/mt01_populacao.py", "scripts/metodo/mt01_relatorio.py",
                 "scripts/geo/cems01_pacote_areal.py", "scripts/geo/cems03_amostrar_pontos.py",
                 "scripts/geo/cob01_cobertura_e_solo.py", "scripts/terreno/baixar_fabdem.py",
                 "scripts/terreno/derivar_aoi_nova.py"]


def ler(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def gravar(p, o):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(o, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(p)


# =========================================================================== protocolo

def candidatos_em_ordem() -> list[dict]:
    f = ler(FILA)
    out = [{"code": c["codigo"], "ordem": i, "papel": "fila_inicial", "pacotes": c["pacotes_vetoriais"],
            "inicio_catalogo": c["inicio_catalogo"], "pais": c["pais"], "nome": c["nome"], "alerta": c.get("alerta")}
           for i, c in enumerate(sorted(f["candidatos"], key=lambda c: c["prioridade"]))]
    hist = {a: ler(led.V4 / "triagem_historica" / "metadados" / f"{a}.json")
            for a in [x["code"] for x in f["alternativas"]] if (led.V4 / "triagem_historica" / "metadados" / f"{a}.json").exists()}
    for x in f["alternativas"]:
        if x.get("estado") != "oportunidade_pendente_admissao":
            continue
        pac = x.get("pacotes_vetoriais") or (hist.get(x["code"]) or {}).get("pacotes") or [
            f"https://{HOST}/static/activations/{x['code']}/{n}" for n in x.get("produtos_catalogo", [])]
        out.append({"code": x["code"], "ordem": len(out), "papel": "alternativa", "pacotes": pac,
                    "inicio_catalogo": x.get("inicio"), "pais": x.get("pais"), "nome": x.get("nome_catalogo"),
                    "alerta": x.get("motivo", "")[:200]})
    vistos, unicos = set(), []
    for c in out:
        if c["code"] not in vistos:
            vistos.add(c["code"])
            unicos.append(c)
    return unicos


def escolher_produtos(pacotes: list[str]) -> list[dict]:
    """DEL primario (sem MONIT), maior versao por AOI; AOIs em ordem crescente; legado DELINEATION_MAP se nao houver."""
    por = {}
    for u in pacotes:
        n = Path(urlparse(u).path).name
        if "_MONIT" in n.upper():
            continue
        m = re.search(r"_AOI(\d+)_DEL_PRODUCT_", n) or re.search(r"_(\d{2})[A-Z0-9]+_\d*DELINEATION_MAP_", n, re.I)
        if not m:
            continue
        v = re.search(r"_v(\d+)_", n)
        k, val = int(m[1]), (int(v[1]) if v else 0, u)
        if k not in por or val > por[k]:
            por[k] = val
    return [{"aoi": a, "url": u} for a, (_, u) in sorted(por.items())][:PRODUTOS_POR_CANDIDATO]


def protocolo() -> dict:
    arq = C3 / "protocolo.json"
    ledger = led.LED / "ledger_eventos.csv"
    cands = candidatos_em_ordem()
    for c in cands:
        c["produtos"] = escolher_produtos(c["pacotes"])
    dim = led.R3 / "dimensionamento.json"
    p = {
        "tipo": "protocolo_confirmacao_externa_rodada3", "linhagem": "confirmacao_rodada3_v1",
        "pergunta_primaria": "Q_conexao - Q_base no recorte fluvial (positivos com obj_desc Riverine flood vs todos os "
                             "negativos observados) em componentes independentes novos",
        "por_que_esta": "contribuicao fluvial da conexao nos pesos consultaveis e a candidata com maior sinal no "
                        "desenvolvimento MT05 (+0,0265, dp 0,0184) e ainda sem confirmacao externa; HAND fica como referencia",
        "secundarias": {"Q_conexao_menos_HAND_fluvial": ["Q_conexao", "A0_regra_hand", "fluvial"],
                        "Q_base_menos_HAND_todos": ["Q_base", "A0_regra_hand", "todos"],
                        "Q_conexao_menos_Q_base_todos": ["Q_conexao", "Q_base", "todos"]},
        "multiplicidade": "primaria IC95; secundarias IC 98,33% (Bonferroni 3), leitura descritiva",
        "regra": {"superioridade": f"delta medio >= {MARGEM} E IC95 inferior > 0 (bootstrap de componentes {REPS}, semente {SEMENTE_BOOT})",
                  "estrita": f"IC inferior > {MARGEM}, reportada a parte", "minimo_protocolo": 8,
                  "abaixo_do_minimo": "reporta estimativa e IC alcancados com o rotulo abaixo_do_minimo; nao vira sustentado"},
        "metrica": {"auc": "dentro do evento (codigo CEMS), AOIs do mesmo codigo juntas", "agregacao":
                    "media de eventos por componente", "evento_valido": f">= {MIN_N} pontos, >= {MIN_POS} positivos, >= {MIN_NEG} negativos"},
        "pesos": {"arquivo": str(PESOS_Q.relative_to(REPO)), "sha256": sha256_arquivo(PESOS_Q),
                  "modelos_pkl": sha256_arquivo(MODELOS_Q), "entradas": "mt05.caracteristicas (hand, ln_sca, declive, "
                  "construido_v2, areia_v2/100, dist_drenagem, dist_canal10, declive_p50)",
                  "ausente": "mediana do padronizador (mesma regra do treino)", "retreino": False},
        "codigo": {f: sha256_arquivo(REPO / f) for f in CODIGO_CADEIA if (REPO / f).exists()},
        "populacao_examinada": {"ledger": sha256_arquivo(ledger), "regra": "examinado, reservado ou planejado por outra frente sai"},
        "grafo": {"regra": "mt01_independencia.grafo primario (evento, L8, onda 30 d, chuva 500 km/15 d) sobre populacao MT01 + "
                           "pontos MT03/MT04 + AOIs de enxurrada + eventos examinados por centroide do catalogo + AOIs dos candidatos",
                  "sensibilidade": "desenho exutorio (MAIN_BAS em qualquer data)", "candidato_ligado_a_examinado": "recusado",
                  "candidatos_ligados_entre_si": "um componente; nunca decompor"},
        "selecao_de_produtos": {"regra": "DEL primario sem MONIT, maior versao por AOI, AOIs crescentes, ate 2 por candidato; "
                                         "legado DELINEATION_MAP se nao houver DEL_PRODUCT", "tetos": {"pacote": TETO_PACOTE, "total": TETO_TOTAL},
                                "recusa_e_substituicao": "candidato recusado (ledger, grafo, esquema, admissao) e substituido pelo proximo "
                                                         "da lista congelada; nunca por desempenho; produto recusado nao e trocado por outra AOI do mesmo candidato"},
        "admissao": {"suporte": "AOI interseccao imageFootprintA (footprint obrigatorio)", "legenda": "observedEventA, event_type 5-Flood",
                     "positivo": "poligono de cheia dentro do observado", "negativo": "observado - cheia - movimento de massa (nunca fora do poligono)",
                     "agua_permanente": "JRC ocorrencia >= 50 fora das duas classes", "data": "cena pos-evento do _source (dbf; xml so em ultimo caso)",
                     "processo": "obj_desc do poligono de cheia mais proximo; fluvial exige Riverine flood",
                     "degelo_lago_barragem_costeira": "admitidos so pela legenda/obj_desc; costeira e sem obj_desc ficam fora do recorte fluvial; "
                                                      "lago/agua permanente sai pelo JRC; nenhum caso e forcado a fluvial"},
        "candidatos": cands, "alvo_componentes": ALVO_COMPONENTES,
        "dimensionamento": {"arquivo": str(dim.relative_to(REPO)), "sha256": sha256_arquivo(dim)} if dim.exists() else None,
        "tempo": {"corte_de_conhecimento_dos_pesos": "MT05 sem_amostra_v1 (pesos de 2026-10)", "regime":
                  "avaliacao externa retrospectiva, previamente cega aos resultados destas unidades; nao e previsao a data do evento",
                  "cobertura": "GHSL na epoca <= ano do evento; SoilGrids estatico (2020) declarado como referencia posterior a 2019"},
        "protegidas": "nao abre OBS03, MT02, GEOID/SAR nem reserva MT; candidatos reservados saem antes do download",
        "promocao": False,
    }
    p["hash_resumo"] = sha256_json({k: v for k, v in p.items()})
    if arq.exists():
        old = ler(arq)
        if old["hash_resumo"] != p["hash_resumo"]:
            raise SystemExit("protocolo mudou depois de congelado; versionar em vez de sobrescrever")
        return old
    gravar(arq, {**p, "quando_utc": agora_utc()})
    gravar(C3 / "registro_inicio.json", {"quando_utc": agora_utc(), "protocolo": sha256_arquivo(arq),
                                         "rotulos_novos_lidos": False, "pacotes_baixados": 0})
    return p


def conferir() -> dict:
    p = ler(C3 / "protocolo.json")
    h = sha256_json({k: v for k, v in p.items() if k not in ("quando_utc", "hash_resumo")})
    if h != p["hash_resumo"]:
        raise SystemExit("protocolo alterado")
    if sha256_arquivo(PESOS_Q) != p["pesos"]["sha256"]:
        raise SystemExit("pesos Q alterados")
    for f, hh in p["codigo"].items():
        if f != "scripts/ia/rodada3_confirmacao.py" and sha256_arquivo(REPO / f) != hh:
            raise SystemExit(f"codigo da cadeia alterado: {f}")
    conferir_revisao_principal(p)
    return p


REVISOES = led.R3.parent / "v2" / "proveniencia" / "revisoes_efetivas.json"


def conferir_revisao_principal(p: dict) -> str:
    """Rodada 3B: o script principal so passa no hash do protocolo ou numa revisao registrada por emenda/adendo
    (`revisoes_efetivas.json`) cujo snapshot existe e tem o mesmo hash. Antes, o hash principal era ignorado."""
    atual = sha256_arquivo(Path(__file__))
    if atual == p["codigo"]["scripts/ia/rodada3_confirmacao.py"]:
        return "protocolo"
    for r in (ler(REVISOES)["revisoes"] if REVISOES.exists() else []):
        if r["sha256"] == atual and (REPO / r["snapshot"]).exists() and sha256_arquivo(REPO / r["snapshot"]) == atual:
            return r["revisao"]
    raise SystemExit("script principal fora das revisoes registradas; registrar a revisao com snapshot antes de executar")


# =========================================================================== adquirir

def adquirir() -> dict:
    import requests
    p = conferir()
    ev = pd.read_csv(led.LED / "ledger_eventos.csv").set_index("code")
    feitos = ler(C3 / "aquisicao.json")["pacotes"] if (C3 / "aquisicao.json").exists() else []
    ja = {x["url"]: x for x in feitos}
    total = sum(x.get("bytes", 0) for x in feitos if x["estado"] == "adquirido")
    rows = []
    for c in p["candidatos"]:
        x = ev.loc[c["code"]] if c["code"] in ev.index else None
        if x is not None and (x.examinado or x.reservado or x.planejado_outra_frente):
            rows.append({"code": c["code"], "estado": "recusado_ledger", "motivo": f"{x.estado} :: {x.frentes_com_exame}"})
            continue
        if not c["produtos"]:
            rows.append({"code": c["code"], "estado": "recusado_sem_delineacao_primaria"})
            continue
        for pr in c["produtos"]:
            u = pr["url"]
            if u in ja and ja[u]["estado"] == "adquirido":
                rows.append(ja[u])
                continue
            if urlparse(u).hostname != HOST:
                rows.append({"code": c["code"], "aoi": pr["aoi"], "url": u, "estado": "recusado_origem"})
                continue
            try:
                h = requests.head(u, timeout=60)
                h.raise_for_status()
                n = int(h.headers["Content-Length"])
                if n > TETO_PACOTE or total + n > TETO_TOTAL:
                    rows.append({"code": c["code"], "aoi": pr["aoi"], "url": u, "estado": "acima_do_teto", "bytes": n})
                    continue
                destino = PACOTES / c["code"] / Path(urlparse(u).path).name
                destino.parent.mkdir(parents=True, exist_ok=True)
                if not destino.exists():
                    tmp = destino.with_suffix(".parcial")
                    with requests.get(u, timeout=(60, 300), stream=True) as r, tmp.open("wb") as f:
                        r.raise_for_status()
                        for b in r.iter_content(1 << 20):
                            f.write(b)
                    if tmp.stat().st_size != n:
                        raise RuntimeError(f"{tmp.stat().st_size} de {n} bytes")
                    tmp.replace(destino)
                total += n
                rows.append({"code": c["code"], "aoi": pr["aoi"], "url": u, "estado": "adquirido", "bytes": n,
                             "arquivo": str(destino.relative_to(REPO)), "sha256": sha256_arquivo(destino),
                             "etag": h.headers.get("ETag"), "quando_utc": agora_utc()})
                print("pacote", c["code"], pr["aoi"], n, flush=True)
            except Exception as e:  # noqa: BLE001 -- falha fica registrada; nao vira ausencia
                rows.append({"code": c["code"], "aoi": pr["aoi"], "url": u, "estado": "falha", "erro": str(e)[:300]})
    gravar(C3 / "aquisicao.json", {"tipo": "aquisicao_confirmacao_rodada3", "quando_utc": agora_utc(),
                                   "protocolo": sha256_arquivo(C3 / "protocolo.json"), "pacotes": rows, "bytes": total,
                                   "rotulos_lidos": False, "licenca": "Copernicus EMS; atribuicao a Uniao Europeia"})
    s = pd.DataFrame(rows)
    return {"estados": s.estado.value_counts().to_dict(), "bytes": total, "candidatos_com_pacote":
            int(s[s.estado == "adquirido"].code.nunique())}


# =========================================================================== geometria neutra + grafo

def _extrair(arq: Path, destino: Path) -> None:
    import mt04_enxurrada as en
    with zipfile.ZipFile(arq) as z:
        en.extrair_seguro(z, destino)
    for interno in list(destino.rglob("*.zip")):
        with zipfile.ZipFile(interno) as z:
            en.extrair_seguro(z, interno.with_suffix(""))


def _camada(pasta: Path, chave: re.Pattern, epsg: int):
    """Le TODAS as feicoes de shapefiles cujo nome casa `chave`; devolve (geometrias, propriedades, arquivos)."""
    import fiona
    from pyproj import CRS, Transformer
    from shapely.geometry import shape
    from shapely.ops import transform
    from shapely.validation import make_valid
    gs, props, arqs = [], [], []
    with fiona.Env(SHAPE_RESTORE_SHX="YES"):
        for shp in sorted(pasta.rglob("*.shp")):
            if not chave.search(shp.name):
                continue
            arqs.append(shp.name)
            with fiona.open(shp) as src:
                t = Transformer.from_crs(CRS.from_user_input(src.crs), CRS.from_epsg(epsg), always_xy=True).transform
                for f in src:
                    g = shape(f["geometry"])
                    if g.is_empty:
                        continue
                    g = transform(t, g)
                    gs.append(g if g.is_valid else make_valid(g))
                    props.append(dict(f["properties"]))
    return gs, props, arqs


AOI_RE = re.compile(r"area_?of_?interest", re.I)
FP_RE = re.compile(r"image_?footprint", re.I)
EVENTO_RE = re.compile(r"observed_?event", re.I)


def _data_cena(pasta: Path) -> str | None:
    """Cena pos-evento mais recente do `_source` (dbf). Emenda 1: produtos RTP01 nomeiam `*_source_r1_v2.dbf`, que o
    padrao `*_source_v*.dbf` do cems03 nao casava; a regra (Post-event, dd/mm/aaaa, a mais recente) e a mesma."""
    import fiona
    achados = []
    for dbf in sorted(pasta.rglob("*.dbf")):
        if not re.search(r"_source(_r\d+)?_v\d+\.dbf$", dbf.name, re.I):
            continue
        with fiona.open(dbf) as src:
            for f in src:
                pr = dict(f["properties"])
                if "post" not in str(pr.get("eventphase") or "").lower():
                    continue
                m = re.search(r"(\d{2})/(\d{2})/(\d{4})", str(pr.get("src_date") or ""))
                if m:
                    achados.append(f"{m[3]}-{m[2]}-{m[1]}")
    return max(achados) if achados else None


def geometria() -> dict:
    """Le so AOI, footprint e data da cena (neutros) e decide elegibilidade pelo grafo; sela antes dos rotulos."""
    from shapely import to_wkt, union_all
    import mt01_independencia as ind
    import mt01_populacao as mp
    import mt03_triagem_cems as tri
    p = conferir()
    if (C3 / "selecao.json").exists():
        return ler(C3 / "selecao.json")
    aq = [x for x in ler(C3 / "aquisicao.json")["pacotes"] if x["estado"] == "adquirido"]
    cat = {a["code"]: a for a in ler(led.V4 / "triagem_historica" / "catalogo.json")["ativacoes"]}
    aois, neutros = [], []
    for x in aq:
        with tempfile.TemporaryDirectory(prefix="r3_geo_") as tmp:
            _extrair(REPO / x["arquivo"], Path(tmp))
            gaoi, _, fa = _camada(Path(tmp), AOI_RE, 4326)
            gfp, _, ff = _camada(Path(tmp), FP_RE, 4326)
            data = _data_cena(Path(tmp))
        r = {"code": x["code"], "aoi": x["aoi"], "arquivo": x["arquivo"], "camadas_aoi": fa, "camadas_footprint": ff,
             "data_cena_pos": data, "tem_aoi": bool(gaoi), "tem_footprint": bool(gfp)}
        neutros.append(r)
        if not gaoi:
            continue
        g = union_all(gaoi)
        inicio = (cat.get(x["code"]) or {}).get("activationTime", "")[:10] or None
        for lon, lat in tri._pontos_da_aoi(to_wkt(g)):
            aois.append({"evento_id": "CEMS_R3:" + x["code"], "unidade_espacial": f"{x['code']}_AOI{x['aoi']:02d}",
                         "janela_inicio": inicio, "janela_fim": data or inicio, "lon": lon, "lat": lat,
                         "fonte": "cems_r3", "classe": 0})
    novos = pd.DataFrame(aois)
    b, hb = mp.localizar_bacias(novos.lon.to_numpy(), novos.lat.to_numpy())
    novos = pd.concat([novos.reset_index(drop=True), b[["MAIN_BAS", "HYBAS_L8"]]], axis=1)
    exam = populacao_examinada(cat)
    juntos = pd.concat([exam, novos], ignore_index=True)
    for c in ("HYBAS_L8", "MAIN_BAS"):
        juntos[c] = juntos[c].fillna(0).astype("int64")
    linhas = []
    for desenho in ("primario", "exutorio"):
        comp, lig = ind.grafo(juntos, desenho)
        exam_ids = set(exam.evento_id)
        por_comp = pd.Series(comp)
        for e in sorted(novos.evento_id.unique()):
            r = comp[e]
            membros = por_comp[por_comp == r].index
            ligados = sorted(set(membros) & exam_ids)
            linhas.append({"desenho": desenho, "code": e.split(":")[1], "componente": r,
                           "ligado_a_examinado": bool(ligados), "exemplos_ligacao": ligados[:6],
                           "candidatos_no_componente": sorted(m.split(":")[1] for m in membros if m.startswith("CEMS_R3:")),
                           "sem_bacia": bool((novos.loc[novos.evento_id == e, "MAIN_BAS"] <= 0).all()), "ligacoes": lig})
    g = pd.DataFrame(linhas)
    prim = g[g.desenho == "primario"].set_index("code")
    selec = []
    for c in p["candidatos"]:
        code = c["code"]
        neu = [r for r in neutros if r["code"] == code]
        if code not in prim.index:
            est = next((x["estado"] for x in ler(C3 / "aquisicao.json")["pacotes"] if x["code"] == code), "sem_pacote")
            selec.append({"code": code, "estado": est if est.startswith("recusado") else "sem_geometria_neutra"})
            continue
        r = prim.loc[code]
        if r.ligado_a_examinado:
            selec.append({"code": code, "estado": "recusado_grafo", "ligacoes": r.exemplos_ligacao})
        elif not any(x["tem_footprint"] for x in neu):
            selec.append({"code": code, "estado": "recusado_sem_footprint"})
        elif not any(x["data_cena_pos"] for x in neu):
            selec.append({"code": code, "estado": "recusado_sem_data_observada"})
        else:
            selec.append({"code": code, "estado": "selecionado_para_rotulo", "componente": str(r.componente),
                          "candidatos_no_componente": r.candidatos_no_componente,
                          "aois": [{"aoi": x["aoi"], "arquivo": x["arquivo"], "data_cena_pos": x["data_cena_pos"]}
                                   for x in neu if x["tem_aoi"] and x["tem_footprint"]]})
    g.to_csv(C3 / "grafo_candidatos.csv", index=False)
    out = {"tipo": "selecao_por_geometria_neutra", "quando_utc": agora_utc(), "protocolo": sha256_arquivo(C3 / "protocolo.json"),
           "hydrobasins": hb, "neutros": neutros, "selecao": selec, "rotulos_lidos": False,
           "populacao_examinada": {"eventos": int(exam.evento_id.nunique()), "fontes": exam.fonte.value_counts().to_dict()},
           "grafo": sha256_arquivo(C3 / "grafo_candidatos.csv")}
    gravar(C3 / "selecao.json", out)
    return {"selecionados": sum(s["estado"] == "selecionado_para_rotulo" for s in selec),
            "estados": pd.Series([s["estado"] for s in selec]).value_counts().to_dict()}


def populacao_examinada(cat: dict) -> pd.DataFrame:
    """No do grafo para tudo o que ja foi examinado, reservado ou planejado (nunca para os candidatos)."""
    import mt01_populacao as mp
    pop = pd.read_parquet(mp.SAIDA / "populacao.parquet", columns=["evento_id", "fonte", "classe", "janela_inicio", "janela_fim",
                                                                   "lat", "lon", "unidade_espacial", "HYBAS_L8", "MAIN_BAS", "papel"])
    pop = pop[pop.fonte.isin(mp.FONTES_CONTRASTE) & pop.papel.isin(["positivo_episodico", "negativo_observado"])].drop(columns="papel")
    partes = [pop]
    meta3 = led.V3 / "triagem_cems" / "metadados"
    for arq in (led.V3 / "cems_desenvolvimento" / "pontos.parquet", led.V4 / "confirmacao" / "pontos_avaliacao.parquet"):
        d = pd.read_parquet(arq, columns=["code", "lon", "lat", "nome", "HYBAS_L8", "MAIN_BAS"] if "HYBAS_L8" in
                            pd.read_parquet(arq).columns else ["code", "lon", "lat", "nome"])
        d = d.groupby("nome").head(40)
        jan = {}
        for c in d.code.unique():
            a = ler(meta3 / f"{c}.json")["ativacao"] if (meta3 / f"{c}.json").exists() else {}
            jan[c] = (a.get("eventTime", "")[:10] or None, a.get("eventTime", "")[:10] or None)
        d = d.assign(evento_id="EXAM:" + d.code, fonte="cems_examinado", classe=0, unidade_espacial=d.nome,
                     janela_inicio=d.code.map(lambda c: jan[c][0]), janela_fim=d.code.map(lambda c: jan[c][1]))
        if "HYBAS_L8" not in d:
            b, _ = mp.localizar_bacias(d.lon.to_numpy(), d.lat.to_numpy())
            d = pd.concat([d.reset_index(drop=True), b[["MAIN_BAS", "HYBAS_L8"]]], axis=1)
        partes.append(d.drop(columns=["code", "nome"]))
    # todo codigo examinado/reservado/planejado pelo ledger que ainda nao tem pontos: centroide do catalogo
    ev = pd.read_csv(led.LED / "ledger_eventos.csv")
    alvo = ev[ev.examinado | ev.reservado | ev.planejado_outra_frente].code
    com_pontos = {e.split(":")[-1] for d in partes[1:] for e in d.evento_id.unique()}
    rows = []
    for c in alvo:
        if c in com_pontos or c not in cat or not cat[c].get("centroid"):
            continue
        m = re.match(r"POINT \(([-\d.]+) ([-\d.]+)\)", cat[c]["centroid"])
        if not m:
            continue
        t = (cat[c].get("activationTime") or "")[:10] or None
        rows.append({"evento_id": "EXAM:" + c, "fonte": "cems_examinado_centroide", "classe": 0, "unidade_espacial": "centroide:" + c,
                     "janela_inicio": t, "janela_fim": t, "lon": float(m[1]), "lat": float(m[2])})
    if rows:
        d = pd.DataFrame(rows)
        b, _ = mp.localizar_bacias(d.lon.to_numpy(), d.lat.to_numpy())
        partes.append(pd.concat([d.reset_index(drop=True), b[["MAIN_BAS", "HYBAS_L8"]]], axis=1))
    return pd.concat(partes, ignore_index=True)


# =========================================================================== areal (le rotulos)

def areal() -> dict:
    import cems01_pacote_areal as c1
    from shapely import union_all
    sel = ler(C3 / "selecao.json")
    if sel["protocolo"] != sha256_arquivo(C3 / "protocolo.json"):
        raise SystemExit("selecao de outro protocolo")
    AREAL.mkdir(parents=True, exist_ok=True)
    linhas = []
    for s in sel["selecao"]:
        if s["estado"] != "selecionado_para_rotulo":
            continue
        for a in s["aois"]:
            nome = f"{s['code']}_AOI{a['aoi']:02d}"
            arq = AREAL / f"{nome}.parquet"
            if arq.exists():
                linhas.append(ler(arq.with_suffix(".json")))
                continue
            with tempfile.TemporaryDirectory(prefix="r3_areal_") as tmp:
                _extrair(REPO / a["arquivo"], Path(tmp))
                g4, _, _ = _camada(Path(tmp), AOI_RE, 4326)
                c = union_all(g4).centroid
                epsg = c1.crs_metrico_de(c.x, c.y)
                gaoi, _, _ = _camada(Path(tmp), AOI_RE, epsg)
                gfp, _, _ = _camada(Path(tmp), FP_RE, epsg)
                gev, pev, fev = _camada(Path(tmp), EVENTO_RE, epsg)
            tipos = [str(p.get("event_type") or p.get("obj_type") or p.get("notation") or "") for p in pev]
            cheia = [(g, str(p.get("obj_desc") or "")) for g, p, t in zip(gev, pev, tipos) if "flood" in t.lower() or t.startswith("5")]
            massa = [g for g, t in zip(gev, tipos) if t.startswith("6") or "mass" in t.lower()]
            obs = union_all(gaoi).intersection(union_all(gfp))
            pos = obs.intersection(union_all([g for g, _ in cheia])) if cheia else None
            neg = obs.difference(pos) if pos is not None else obs
            if massa:
                neg = neg.difference(union_all(massa))
            import mt03_cems_desenvolvimento as cd
            geo = [{"papel": "observado", "geometry": obs}, {"papel": "negativo", "geometry": neg}]
            geo += [{"papel": "positivo", "geometry": pos}] if pos is not None and not pos.is_empty else []
            geo += [{"papel": f"cheia|{d}", "geometry": g} for g, d in cheia]
            cd._gravar_geo(arq, geo, epsg)
            km = lambda g: 0.0 if g is None else g.area / 1e6  # noqa: E731
            r = {"nome": nome, "code": s["code"], "aoi": f"AOI{a['aoi']:02d}", "componente": s["componente"], "epsg": epsg,
                 "data_cena_pos": a["data_cena_pos"], "camadas_evento": fev, "legenda": sorted(set(tipos)),
                 "obj_desc": pd.Series([d for _, d in cheia]).value_counts().to_dict(),
                 "area_observada_km2": km(obs), "area_positiva_km2": km(pos), "area_negativa_km2": km(neg),
                 "pacote_sha256": sha256_arquivo(REPO / a["arquivo"])}
            arq.with_suffix(".json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            linhas.append(r)
            print(nome, round(r["area_observada_km2"], 1), round(r["area_positiva_km2"], 1), r["obj_desc"], flush=True)
    t = pd.DataFrame(linhas)
    t.to_csv(C3 / "areal.csv", index=False)
    gravar(C3 / "registro_rotulos.json", {"quando_utc": agora_utc(), "rotulos_novos_lidos": True,
                                          "selecao": sha256_arquivo(C3 / "selecao.json"), "aois": len(t)})
    return {"aois": len(t), "com_positivo": int((t.area_positiva_km2 > 0).sum()),
            "com_negativo": int((t.area_negativa_km2 > 0).sum())}


# =========================================================================== terreno, pontos

def _configurar_cd():
    import mt03_cems_desenvolvimento as cd
    cd.CD, cd.PACOTES, cd.AREAL, cd.MDT, cd.TERRENO = C3, PACOTES, AREAL, MDT, TERRENO
    return cd


def _origem_fabdem_responde() -> bool:
    import requests
    try:
        r = requests.get("https://data.bris.ac.uk/datasets/s5hqmjcdj8yo2ibzi9b4ew3sn/N40E010-N50E020_FABDEM_V1-2.zip",
                         headers={"Range": "bytes=0-99"}, timeout=60)
        return r.status_code == 206
    except Exception:  # noqa: BLE001
        return False


def terreno(maximo: int | None = None) -> dict:
    """Cadeia MT03. Emenda 2: se a origem do FABDEM recusa conexao, os MESMOS bytes vem do espelho GEE, habilitado so
    depois de `rodada3_fabdem_gee.conferir` dar identidade pixel a pixel com uma janela baixada da origem."""
    import baixar_fabdem as fab
    import rodada3_fabdem_gee as fg
    cd = _configurar_cd()
    conf = C3 / "fabdem_gee_conferencia.json"
    if not _origem_fabdem_responde():
        if not (conf.exists() and ler(conf)["identico"]):
            raise SystemExit("origem FABDEM fora e transporte GEE nao conferido")
        fab.baixar = lambda plano, saida=MDT: fg.baixar(plano, saida)
        print("transporte: espelho GEE do FABDEM V1-2 (conferido identico)", flush=True)
    return cd.terreno(maximo)


def pontos() -> dict:
    import mt03_atributos_novos as an
    import mt01_experimento as e1
    cd = _configurar_cd()
    t = pd.read_csv(C3 / "areal.csv")
    t["unidade_nova"] = "R3C_" + t.componente.astype(str)
    t.to_csv(C3 / "areal.csv", index=False)
    r = cd.pontos()
    d = pd.read_parquet(C3 / "pontos.parquet")
    d["evento_id"] = "CEMS_R3:" + d.code
    d["ponto_id"] = d.ponto_id.str.replace("MT03DEV_", "R3CONF_", regex=False)
    d["unidade_terreno"] = "r3conf/" + d.nome
    # emenda 4: a data da observacao e exigida POR AOI (regra de admissao); AOI sem data propria sai, mesmo que outra
    # AOI do mesmo codigo tenha data
    sem_data = sorted(t.loc[t.data_cena_pos.isna(), "nome"])
    gravar(C3 / "admissao_aoi.json", {"quando_utc": agora_utc(), "recusadas_sem_data_observada": sem_data,
                                      "admitidas": sorted(set(t.nome) - set(sem_data))})
    d = d[~d.nome.isin(sem_data)]
    te = e1.amostrar(d, e1.CONFIG["teto_teste_por_evento_classe"], e1.CONFIG["semente"])
    partes = []
    for code, g in te.groupby("code"):
        ano = int(min(str(x)[:4] for x in t.loc[(t.code == code) & t.data_cena_pos.notna(), "data_cena_pos"]))
        arq = C3 / "atributos" / f"{code}.parquet"
        if arq.exists():
            partes.append(pd.read_parquet(arq))
            continue
        g = g.drop(columns=["elevation_local_m", "ln_sca"], errors="ignore").copy()
        try:
            x = an.completar(g, TERRENO, ano, code, C3 / "lotes_cobertura", grupo="unidade_espacial")
        except ValueError as e:  # emenda 5: falha da cadeia congelada (cobertura geografica), nunca por rotulo
            falhas = ler(C3 / "atributos_falhas.json") if (C3 / "atributos_falhas.json").exists() else {}
            falhas[code] = {"erro": str(e)[:300], "quando_utc": agora_utc()}
            gravar(C3 / "atributos_falhas.json", falhas)
            print("FALHA atributos", code, str(e)[:120], flush=True)
            continue
        arq.parent.mkdir(parents=True, exist_ok=True)
        x.to_parquet(arq, index=False)
        partes.append(x)
        print("atributos", code, len(x), flush=True)
    d = pd.concat(partes, ignore_index=True)
    d.to_parquet(C3 / "pontos_avaliacao.parquet", index=False)
    return {**r, "pontos_avaliacao": len(d), "eventos": int(d.evento_id.nunique()),
            "cobertura_ausente": {c: float(d[c].isna().mean()) for c in ("construido_v2", "areia_v2")}}


# =========================================================================== equivalencia (so dados de treino)

def equivalencia(n_por_fonte: int = 75, semente: int = 20261017) -> dict:
    """Cobertura recalculada pela cadeia de importacao x valor do treino, e escore do JSON exportado x objeto de treino.

    Usa so a populacao de desenvolvimento (MT03-A, a mesma que ajustou Q); nenhum destino novo e lido.
    """
    import mt03_atributos_novos as an
    import mt03_experimento as e3
    import mt05_transferencia_sem_amostra as m5
    d = e3.carregar()
    x = m5.caracteristicas(d)
    esc = escores_q_completos(d, x)
    rng = np.random.default_rng(semente)
    amostra = []
    for _, g in d[d.construido_v2.notna() & d.janela_inicio.notna()].groupby("fonte"):
        amostra.append(g.iloc[np.sort(rng.choice(len(g), min(n_por_fonte, len(g)), replace=False))])
    a = pd.concat(amostra)
    a = a.assign(ano=pd.to_datetime(a.janela_inicio, utc=True, errors="coerce").dt.year)
    partes = []
    for ano, g in a.dropna(subset=["ano"]).groupby("ano"):
        r = an.cobertura(g[["ponto_id", "lat", "lon"]].copy(), int(ano), C3 / "equivalencia_lotes", f"eq{int(ano)}")
        partes.append(r.merge(g[["ponto_id", "construido_v2", "areia_v2", "fonte"]], on="ponto_id"))
    c = pd.concat(partes)
    dc = (c.construido_frac - c.construido_v2).abs()
    da = (c.areia_pct_0_30 - c.areia_v2).abs()
    r = {"tipo": "equivalencia_atributos_e_escore", "quando_utc": agora_utc(),
         "escore_json_vs_objeto": esc, "pontos_cobertura": len(c), "por_fonte": c.fonte.value_counts().to_dict(),
         "construido": {"max_abs": float(dc.max()), "frac_iguais_1e6": float((dc < 1e-6).mean()), "nan_novo": int(c.construido_frac.isna().sum())},
         "areia": {"max_abs": float(da.max()), "frac_iguais_1e6": float((da < 1e-6).mean()), "nan_novo": int(c.areia_pct_0_30.isna().sum()),
                   "nan_treino": int(c.areia_v2.isna().sum())},
         "nao_e": "distancias de drenagem/canal e contexto usam a mesma funcao (mt02_atributos/mt01_populacao); nao recalculadas aqui"}
    gravar(C3 / "equivalencia.json", r)
    return r


def escores_q_completos(d: pd.DataFrame, x: pd.DataFrame) -> dict:
    import pickle
    import rodada2_confirmacao as r2
    with MODELOS_Q.open("rb") as f:
        mods = pickle.load(f)
    return {n: float(np.nanmax(np.abs(np.asarray(mods[n].escore(x), float) - r2.escore_q(n, x)))) for n in ("Q_base", "Q_conexao")}


# =========================================================================== avaliar

def escores(d: pd.DataFrame) -> pd.DataFrame:
    import pickle
    import mt05_transferencia_sem_amostra as m5
    import rodada2_confirmacao as r2
    x = m5.caracteristicas(d)
    out = pd.DataFrame({"A0_regra_hand": -d.hand_m.to_numpy()}, index=d.index)
    for nome in ("Q_base", "Q_conexao"):
        out[nome] = r2.escore_q(nome, x)
    with MODELOS_Q.open("rb") as f:
        mods = pickle.load(f)
    mods = mods.get("modelos", mods)
    for nome in ("Q_base", "Q_conexao"):
        ref = np.asarray(mods[nome].escore(x), float)
        if np.nanmax(np.abs(ref - out[nome].to_numpy())) > 1e-9:
            raise SystemExit(f"escore exportado de {nome} diverge do objeto de treino")
    return out


def medidas(d: pd.DataFrame, cols) -> pd.DataFrame:
    from sklearn.metrics import roc_auc_score
    import mt01_experimento as e1
    rows = []
    for (un, ev), g in d.groupby(["unidade", "evento_id"]):
        for rec, s in {"todos": g, "fluvial": g[(g.classe == 0) | (g.processo_ponto == "fluvial")]}.items():
            if len(s) < MIN_N or s.classe.sum() < MIN_POS or (s.classe == 0).sum() < MIN_NEG:
                continue
            for m in cols:
                auc = roc_auc_score(s.classe, s[m])
                if abs(auc - e1.auc(s.classe.to_numpy(), s[m].to_numpy())) > 1e-9:
                    raise SystemExit("AUC independente divergente")
                rows.append({"unidade": un, "evento_id": ev, "recorte": rec, "modelo": m, "auc": auc,
                             "n": len(s), "positivos": int(s.classe.sum())})
    return pd.DataFrame(rows)


def contraste(m: pd.DataFrame, rec: str, a: str, b: str, nivel: float = 95.0) -> dict:
    s = m[m.recorte == rec]
    if s.empty:
        return {"estado": "not_testable", "motivo": "nenhum evento valido no recorte"}
    u = s.groupby(["unidade", "evento_id", "modelo"]).auc.mean().groupby(level=[0, 2]).mean().unstack()
    dd = (u[a] - u[b]).dropna()
    r = {"unidades": int(len(dd)), "delta_medio": float(dd.mean()), "delta_mediano": float(dd.median()),
         "unidades_a_favor": int((dd > 0).sum()), "por_unidade": dd.round(5).to_dict()}
    if len(dd) >= 2:
        rng = np.random.default_rng(SEMENTE_BOOT)
        v = dd.to_numpy()
        bs = v[rng.integers(0, len(v), (REPS, len(v)))].mean(1)
        lo, hi = np.percentile(bs, [(100 - nivel) / 2, 100 - (100 - nivel) / 2])
        r["ic"] = [float(lo), float(hi)]
        r["nivel_ic"] = nivel
        regra = r["delta_medio"] >= MARGEM and lo > 0
        r["veredito"] = (("sustentado" if regra else "nao_sustentado" if hi < MARGEM else "indecidivel")
                         if len(dd) >= 8 else "abaixo_do_minimo_do_protocolo")
        r["estrita_ic_inferior_acima_da_margem"] = bool(lo > MARGEM)
    else:
        r["veredito"] = "abaixo_do_minimo_do_protocolo"
    return r


def sensibilidade_exutorio(d: pd.DataFrame, cols) -> dict:
    """Desenho exutorio (MAIN_BAS em qualquer data): so codigos livres de examinado, componentes desse desenho."""
    g = pd.read_csv(C3 / "grafo_candidatos.csv")
    ex = g[(g.desenho == "exutorio") & ~g.ligado_a_examinado].set_index("code").componente
    s = d[d.code.isin(ex.index)].copy()
    if s.empty:
        return {"estado": "not_testable", "motivo": "nenhum codigo avaliado fica livre no desenho exutorio"}
    s["unidade"] = "EXU_" + s.code.map(ex).astype(str)
    m = medidas(s, cols)
    return {"codigos": sorted(s.code.unique()), "primaria": contraste(m, "fluvial", "Q_conexao", "Q_base"),
            "Q_base_menos_HAND_todos": contraste(m, "todos", "Q_base", "A0_regra_hand", 98.33)}


def avaliar() -> dict:
    p = conferir()
    d = pd.read_parquet(C3 / "pontos_avaliacao.parquet")
    d = pd.concat([d.reset_index(drop=True), escores(d).reset_index(drop=True)], axis=1)
    d.to_parquet(C3 / "predicoes.parquet", index=False)
    cols = ["A0_regra_hand", "Q_base", "Q_conexao"]
    m = medidas(d, cols)
    m.to_parquet(C3 / "medidas.parquet", index=False)
    r = {"tipo": "confirmacao_externa_rodada3", "quando_utc": agora_utc(), "protocolo": p["hash_resumo"],
         "predicoes": sha256_arquivo(C3 / "predicoes.parquet"), "medidas": sha256_arquivo(C3 / "medidas.parquet"),
         "pontos": len(d), "eventos": int(d.evento_id.nunique()), "componentes": int(d.unidade.nunique()),
         "primaria": contraste(m, "fluvial", "Q_conexao", "Q_base"),
         "secundarias": {k: contraste(m, v[2], v[0], v[1], 98.33) for k, v in p["secundarias"].items()},
         "sensibilidade_exutorio": sensibilidade_exutorio(d, cols),
         "auc_media_por_modelo": m.groupby(["recorte", "modelo"]).auc.mean().unstack().to_dict("index"),
         "promovido": False, "retreino": False}
    gravar(C3 / "resultado.json", r)
    return r


# =========================================================================== tabela da busca e contagens

def tabela() -> dict:
    p = ler(C3 / "protocolo.json")
    ev = pd.read_csv(led.LED / "ledger_eventos.csv").set_index("code")
    aq = pd.DataFrame(ler(C3 / "aquisicao.json")["pacotes"]) if (C3 / "aquisicao.json").exists() else pd.DataFrame()
    sel = {s["code"]: s for s in ler(C3 / "selecao.json")["selecao"]} if (C3 / "selecao.json").exists() else {}
    neu = pd.DataFrame(ler(C3 / "selecao.json")["neutros"]) if (C3 / "selecao.json").exists() else pd.DataFrame()
    ar = pd.read_csv(C3 / "areal.csv") if (C3 / "areal.csv").exists() else pd.DataFrame()
    pts = pd.read_parquet(C3 / "pontos_avaliacao.parquet") if (C3 / "pontos_avaliacao.parquet").exists() else pd.DataFrame()
    med = pd.read_parquet(C3 / "medidas.parquet") if (C3 / "medidas.parquet").exists() else pd.DataFrame()
    falhas_attr = ler(C3 / "atributos_falhas.json") if (C3 / "atributos_falhas.json").exists() else {}
    sem_data_aoi = set(ler(C3 / "admissao_aoi.json")["recusadas_sem_data_observada"]) if (C3 / "admissao_aoi.json").exists() else set()
    linhas = []
    for c in p["candidatos"]:
        code = c["code"]
        x = ev.loc[code] if code in ev.index else None
        a = aq[aq.code == code] if len(aq) else aq
        s = sel.get(code, {})
        r_ar = ar[ar.code == code] if len(ar) else ar
        r_pt = pts[pts.code == code] if len(pts) else pts
        r_me = med[med.evento_id == "CEMS_R3:" + code] if len(med) else med
        if len(r_me) and (r_me.recorte == "fluvial").any():
            adm = "admitido_primaria_e_secundarias"
        elif len(r_me):
            adm = "admitido_so_recorte_todos"
        elif len(r_pt):
            adm = "pontos_sem_evento_valido"
        else:
            adm = None
        motivo = None
        if x is not None and (x.examinado or x.reservado or x.planejado_outra_frente):
            motivo = f"ledger: {x.estado} ({x.frentes_com_exame or 'reserva/plano'})"
        elif s.get("estado", "").startswith("recusado") or s.get("estado") == "sem_geometria_neutra":
            motivo = s["estado"] + (f" {s.get('ligacoes', [])[:3]}" if s.get("ligacoes") else "")
        elif len(a) and not (a.estado == "adquirido").any():
            motivo = "; ".join(sorted(set(a.estado)))
        elif code in falhas_attr:
            motivo = "falha da cadeia congelada (HydroSHEDS: regiao 'as' sem cobertura no Timor; emenda 5)"
        elif len(r_ar) and set(r_ar.nome) <= sem_data_aoi:
            motivo = "admissao: AOI sem cena pos-evento propria (emenda 4)"
        elif adm in ("pontos_sem_evento_valido",) or (len(r_ar) and adm is None):
            motivo = "admissao: sem evento com >=10 positivos e >=10 negativos observados"
        linhas.append({
            "ordem": c["ordem"], "papel": c["papel"], "code": code, "pais": c["pais"], "nome": c["nome"],
            "produtos": "; ".join(Path(urlparse(pp["url"]).path).name for pp in c["produtos"]),
            "uso_anterior": None if x is None else x.estado, "frentes_com_exame": None if x is None else x.frentes_com_exame,
            "reserva": None if x is None else bool(x.reservado),
            "pacotes_adquiridos": int((a.estado == "adquirido").sum()) if len(a) else 0,
            "data_cena_pos": ";".join(sorted({str(v) for v in neu[neu.code == code].data_cena_pos.dropna()})) if len(neu) else None,
            "componente": s.get("componente"), "candidatos_no_componente": ";".join(s.get("candidatos_no_componente", [])),
            "aois_areal": len(r_ar), "obj_desc": "; ".join(f"{k}:{v}" for o in (r_ar.obj_desc if len(r_ar) else [])
                                                          for k, v in (ast.literal_eval(o) if isinstance(o, str) else o).items()),
            "positivos_fluviais": int(((r_pt.classe == 1) & (r_pt.processo_ponto == "fluvial")).sum()) if len(r_pt) else 0,
            "positivos": int(r_pt.classe.sum()) if len(r_pt) else 0, "negativos": int((r_pt.classe == 0).sum()) if len(r_pt) else 0,
            "admissao": adm, "motivo_recusa": motivo, "alerta_coordenacao": c.get("alerta")})
    t = pd.DataFrame(linhas)
    t.to_csv(C3 / "tabela_busca.csv", index=False)
    cont = {"codigos_na_lista_congelada": len(t), "codigos_com_pacote": int((t.pacotes_adquiridos > 0).sum()),
            "codigos_selecionados_pelo_grafo": int(sum(s.get("estado") == "selecionado_para_rotulo" for s in sel.values())),
            "aois_lidas": int(len(ar)), "eventos_avaliados": int(med.evento_id.nunique()) if len(med) else 0,
            "componentes_avaliados": int(med.unidade.nunique()) if len(med) else 0,
            "componentes_no_recorte_fluvial": int(med[med.recorte == "fluvial"].unidade.nunique()) if len(med) else 0,
            "realmente_novos": "componentes avaliados sem ligacao a examinado no grafo primario"}
    gravar(C3 / "contagens.json", cont)
    return cont


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["protocolo", "adquirir", "geometria", "areal", "terreno", "pontos", "equivalencia", "avaliar", "tabela"])
    ap.add_argument("--maximo", type=int)
    a = ap.parse_args()
    f = {"protocolo": protocolo, "adquirir": adquirir, "geometria": geometria, "areal": areal,
         "terreno": lambda: terreno(a.maximo), "pontos": pontos, "equivalencia": equivalencia, "avaliar": avaliar, "tabela": tabela}[a.acao]
    print(json.dumps(f(), ensure_ascii=False, indent=1, default=str)[:8000])
