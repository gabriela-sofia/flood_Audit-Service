"""Rodada 3, parte 1: ledger de eventos examinados com identidade canonica e evidencia por caminho e campo.

Linhagem `confirmacao_rodada3_v1` (17/10/2026). Substitui a `elegibilidade()` da rodada 2, que ignorava qualquer
diretorio com `confirmacao` no caminho e por isso declarou como nao examinadas 15 ativacoes ja abertas pelo MT04.

Cada linha do ledger diz: codigo canonico (EMSRnnn), alias encontrado, categoria de evidencia, frente, caminho e
campo. A classificacao do evento e a evidencia MAIS FORTE entre as linhas, nesta ordem:

  rotulo_lido            pontos/atributos/geometria areal/raster de rotulo lidos por alguma frente
  uso_chip_sar           chip rotulado citado em protocolo/resultado/admissao SAR (treino, avaliacao, retencao)
  exame_registrado       abertura, registro de exame, pacote baixado ou admissao (rotulo pode ter sido lido)
  artefato_binario       zip/tif/shp/parquet com o codigo no caminho fora de catalogo (presumido examinado)
  reserva_protegida      reserva de outra frente; nunca abrir
  planejado_outra_frente selecionado em plano de outra frente sem recibo de aquisicao
  catalogo               so inventario/metadado/lista (nao e exame)
  mencao_textual         codigo citado em texto sem campo de selecao (nao e exame)

Fontes estruturadas (com campo) vem primeiro; a varredura generica de caminhos/conteudo so complementa e nunca
rebaixa uma evidencia estruturada. Ausencia de linha nao certifica ausencia de exame: o limite da varredura fica
registrado (tamanhos, extensoes, exclusoes).

Uso:
    python scripts/ia/rodada3_ledger.py construir
    python scripts/ia/rodada3_ledger.py elegibilidade
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
for s in ("ia", "metodo", "servico", "sar"):
    sys.path.insert(0, str(REPO / "scripts" / s))
from proveniencia import agora_utc, sha256_arquivo  # noqa: E402

RUNS = REPO / "local_runs"
R3 = RUNS / "confirmacao-rodada3" / "v1"
LED = R3 / "ledger"
V3 = RUNS / "mecanismos-transferiveis" / "v3"
V4 = RUNS / "mecanismos-transferiveis" / "v4"
COORD = RUNS / "coordenacao-rodada3" / "v1"

CODIGO = re.compile(r"EMSR-?(\d{3})", re.I)
CHIP = re.compile(r"EMSR-?(\d{3})-\d+-\d+", re.I)
ORDEM = ["rotulo_lido", "uso_chip_sar", "exame_registrado", "artefato_binario", "reserva_protegida",
         "planejado_outra_frente", "catalogo", "mencao_textual"]
EXAMINADO = {"rotulo_lido", "uso_chip_sar", "exame_registrado", "artefato_binario"}
BINARIOS = {".zip", ".tif", ".tiff", ".shp", ".dbf", ".shx", ".gpkg", ".geojson", ".parquet", ".npz", ".npy",
            ".kml", ".kmz", ".pt", ".tar"}
TEXTO = {".json", ".jsonl", ".csv", ".txt", ".md", ".gz"}
LIMITE_TEXTO = 30 * 1024 ** 2
CATALOGO_PALAVRAS = re.compile(r"catalog|catalogo|shard_index|metadados|cems-metadados|lista_cems|mmflood|triagem|"
                               r"fila_|ocorrencias|pretriagem|cabecalhos|tile_catalog|inventario", re.I)
EXCLUIR_RAIZES = {"confirmacao-rodada3", "coordenacao-rodada3"}  # saida propria e insumos tratados como estruturados


def canon(txt: str) -> list[str]:
    return sorted({f"EMSR{m}" for m in CODIGO.findall(str(txt))})


def frente(rel: str) -> str:
    r = rel.replace("\\", "/")
    regras = [("mecanismos-transferiveis/v4/confirmacao", "MT04_confirmacao"),
              ("mecanismos-transferiveis/v4/enxurrada", "MT04_enxurrada"),
              ("mecanismos-transferiveis/v4/urbansarfloods", "MT04_urbano"),
              ("mecanismos-transferiveis/v4/triagem_historica", "catalogo_historico"),
              ("mecanismos-transferiveis/v3", "MT03"), ("mecanismos-transferiveis", "MT_outros"),
              ("sar-", "SAR"), ("aprendizado-continuo", "SAR"), ("evolucao-sar", "SAR"), ("pre-pos-geoid", "SAR"),
              ("pretreino-sar", "SAR"), ("geoid-", "SAR_catalogo_GEOID"), ("ponte-sar", "SAR"),
              ("reservas-", "reservas"), ("observacoes-dirigidas", "OBS"),
              ("aprendizado-conjunto", "base_fisica"), ("referencia-fisica-forcamento", "base_fisica_rodada2"),
              ("fechamento-aprendizado", "base_fisica"), ("pilha-s1", "base_fisica"), ("ter-0", "base_fisica"),
              ("cems-", "base_fisica_cems"), ("mdt-fabdem", "base_fisica")]
    for chave, nome in regras:
        if chave in r:
            return nome
    return "outros:" + r.split("/")[1] if "/" in r else "outros"


def linha(code, alias, categoria, fr, caminho, campo):
    return {"code": code, "alias": str(alias)[:120], "categoria": categoria, "frente": fr,
            "caminho": str(Path(caminho).relative_to(REPO)) if Path(caminho).is_absolute() else str(caminho),
            "campo": campo}


def ler(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ fontes estruturadas

def estruturadas() -> list[dict]:
    L = []
    # 1. base fisica e registros congelados
    import conjunta_base as cb
    p = cb.BASE / "pontos.parquet"
    for e in pd.read_parquet(p, columns=["evento_id"]).evento_id.unique():
        for c in canon(e):
            L.append(linha(c, e, "rotulo_lido", "base_fisica", p, "evento_id"))
    pop = V3.parent / "v1" / "populacao.parquet"
    if pop.exists():
        for e in pd.read_parquet(pop, columns=["evento_id"]).evento_id.dropna().unique():
            for c in canon(e):
                L.append(linha(c, e, "rotulo_lido", "MT01_populacao", pop, "evento_id"))
    k = RUNS / "reservas-examinadas" / "conhecidos_2026_10_03.json"
    for fonte, evs in ler(k)["por_fonte"].items():
        for e in evs:
            for c in canon(e):
                L.append(linha(c, e, "exame_registrado", "registro_conhecidos", k, f"por_fonte.{fonte}"))
    j = RUNS / "reservas-examinadas" / "examinados_apos_congelamento.jsonl"
    for n, t in enumerate(j.read_text(encoding="utf-8").splitlines()):
        d = json.loads(t)
        for e in d.get("eventos", []):
            for c in canon(e):
                L.append(linha(c, e, "exame_registrado", "registro_exames:" + d.get("avaliacao", "?"), j,
                               f"linha{n}.eventos"))
    # 2. MT03 particao (desenvolvimento lido; reserva aberta depois pelo MT04)
    part = V3 / "triagem_cems" / "particao.json"
    for u in ler(part)["unidades"]:
        for c in u["ativacoes"]:
            cat = "rotulo_lido" if u["papel"] == "desenvolvimento" else "exame_registrado"
            L.append(linha(c, u["unidade_nova"], cat, "MT03", part, f"unidades[papel={u['papel']}].ativacoes"))
    for arq, campo in ((V3 / "cems_desenvolvimento" / "pacotes.json", "pacotes.code"),
                       (V4 / "confirmacao" / "cems" / "pacotes.json", "pacotes.code")):
        if arq.exists():
            for x in ler(arq)["pacotes"]:
                if x.get("estado") == "baixado":
                    L.append(linha(x["code"], x.get("produto"), "exame_registrado", frente(str(arq.relative_to(REPO))),
                                   arq, campo + "[estado=baixado]"))
    for arq in (V3 / "cems_desenvolvimento" / "pontos.parquet", V4 / "confirmacao" / "pontos_avaliacao.parquet",
                V4 / "confirmacao" / "cems" / "pontos.parquet", V4 / "confirmacao" / "predicoes.parquet"):
        if arq.exists():
            for c in pd.read_parquet(arq, columns=["code"]).code.unique():
                L.append(linha(c, c, "rotulo_lido", frente(str(arq.relative_to(REPO))), arq, "code"))
    # 3. MT04 confirmacao: abertura e atributos
    ab = V4 / "confirmacao" / "abertura.json"
    for c in ler(ab)["ativacoes"]:
        L.append(linha(c, c, "exame_registrado", "MT04_confirmacao", ab, "ativacoes"))
    for f in sorted((V4 / "confirmacao" / "atributos").glob("EMSR*.parquet")):
        L.append(linha(f.stem, f.name, "rotulo_lido", "MT04_confirmacao", f, "arquivo de atributos (classe por ponto)"))
    # 4. MT04 enxurrada: aquisicoes, qualificacoes (leram obj_desc), admissao areal
    for arq in sorted((V4 / "enxurrada").rglob("aquisicao.json")):
        for x in ler(arq).get("pacotes", []):
            if x.get("estado") == "adquirido":
                for c in canon(x.get("code", "")):
                    L.append(linha(c, x.get("arquivo"), "exame_registrado", "MT04_enxurrada", arq, "pacotes[estado=adquirido]"))
    for arq in sorted((V4 / "enxurrada").rglob("qualificacao.json")):
        for x in ler(arq).get("ativacoes", []):
            lidas = any(cam.get("descricoes") for cam in x.get("camadas", []))
            for c in canon(x.get("code", "")):
                L.append(linha(c, x.get("arquivo"), "rotulo_lido" if lidas else "exame_registrado", "MT04_enxurrada",
                               arq, "ativacoes.camadas.descricoes" if lidas else "ativacoes"))
    for arq in sorted((V4 / "enxurrada").rglob("resultado.json")):
        d = ler(arq)
        for x in d.get("aois", []) if isinstance(d.get("aois"), list) else []:
            for c in canon(x.get("code", x.get("nome", ""))):
                L.append(linha(c, x.get("nome"), "rotulo_lido", "MT04_enxurrada", arq, "aois (geometria areal)"))
    # 5. SAR: planos (pares selecionados), admissoes, aquisicoes, rotulos materializados
    for arq in list((RUNS / "sar-observacoes-datadas").rglob("plano.json")) + list((RUNS / "sar-temporal-v2").rglob("plano.json")):
        try:
            d = ler(arq)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for x in d.get("pares", []) or []:
            for c in canon(x.get("evento", x.get("chave", ""))):
                L.append(linha(c, x.get("chave"), "planejado_outra_frente", "SAR", arq, "pares[].evento"))
    for raiz in ("sar-observacoes-datadas", "sar-temporal-v2"):
        for arq in (RUNS / raiz).rglob("manifesto.json"):
            partes = arq.parts
            if "admissao" in partes or "aquisicoes" in partes:
                try:
                    d = ler(arq)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                for x in d.get("pares", []) or d.get("itens", []) or []:
                    if not isinstance(x, dict):
                        continue
                    cat = "rotulo_lido" if ("admissao" in partes and x.get("pixels_conjuntos_por_classe")) else "exame_registrado"
                    for c in canon(x.get("evento", x.get("chave", ""))):
                        L.append(linha(c, x.get("chave"), cat, "SAR", arq,
                                       "pares[].pixels_conjuntos_por_classe" if cat == "rotulo_lido" else "pares[].evento"))
    # 6. reservas e contratos de outras frentes
    import mt03_triagem_cems as tri
    src = REPO / "scripts/metodo/mt03_triagem_cems.py"
    for nome, conj in (("OBS03_NUNCA_LER", tri.NUNCA_LER), ("MT02_confirmacao", tri.RESERVA_MT02),
                       ("FRENTE_SAR_GEOID", tri.FRENTE_SAR)):
        for c in conj:
            L.append(linha(c, c, "reserva_protegida", "reservas:" + nome, src, nome))
    res = RUNS / "mecanismos-transferiveis" / "confirmacao" / "reserva.json"
    if res.exists():
        for c in canon(res.read_text(encoding="utf-8")):
            L.append(linha(c, c, "reserva_protegida", "reservas:MT_confirmacao", res, "texto do contrato"))
    tri_csv = V3 / "triagem_cems" / "triagem.csv"
    t = pd.read_csv(tri_csv)
    for c in t.loc[t.motivo_exclusao == "reserva_sar_geoid", "code"]:
        L.append(linha(c, c, "reserva_protegida", "reservas:GEOID_SAR", tri_csv, "motivo_exclusao=reserva_sar_geoid"))
    # 7. catalogos e inventarios (nao sao exame)
    cat = V4 / "triagem_historica" / "catalogo.json"
    for a in ler(cat)["ativacoes"]:
        L.append(linha(a["code"], a.get("name"), "catalogo", "catalogo_historico", cat, "ativacoes.code"))
    for arq, chave in ((REPO / "local_runs/referencia-fisica-forcamento/v1/confirmacao/lista_cems_20261017.json", "itens"),):
        if arq.exists():
            for a in ler(arq)[chave]:
                L.append(linha(a["code"], a.get("name"), "catalogo", "lista_recente", arq, "itens.code"))
    mm = COORD / "fontes" / "mmflood_activations.json"
    for c in ler(mm):
        L.append(linha(c, "mmflood", "catalogo", "catalogo_MMFlood", mm, "chave"))
    return L


# ------------------------------------------------------------------ varredura generica

def _texto(p: Path) -> str | None:
    try:
        if p.suffix == ".gz":
            with gzip.open(p, "rt", encoding="utf-8", errors="replace") as f:
                return f.read()
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def generica() -> tuple[list[dict], dict]:
    L, limites = [], {"arquivos_vistos": 0, "texto_lido": 0, "texto_acima_do_limite": 0, "parquet_esquemas": 0,
                      "parquet_lidos": 0, "excluidos": sorted(EXCLUIR_RAIZES), "limite_texto_bytes": LIMITE_TEXTO}
    colunas_id = {"evento_id", "code", "codigo", "evento", "event", "activation", "chave", "unidade_espacial", "nome",
                  "aoi", "tile_id", "tile", "event_id", "emsr"}
    import pyarrow.parquet as pq
    for root, dirs, files in os.walk(RUNS):
        rel_root = Path(root).relative_to(RUNS)
        if rel_root.parts and rel_root.parts[0] in EXCLUIR_RAIZES:
            dirs[:] = []
            continue
        for n in dirs:
            for c in canon(n):
                cam = Path(root) / n
                rel = str(cam.relative_to(REPO))
                if CHIP.search(n) or re.search(r"EMSR-?\d{3}-\d+", n):
                    cat = "rotulo_lido" if any(x in rel.lower() for x in ("label", "dados")) else "artefato_binario"
                    L.append(linha(c, n, cat, frente(rel), cam, "nome de diretorio de chip rotulado"))
        for n in files:
            limites["arquivos_vistos"] += 1
            cam = Path(root) / n
            rel = str(cam.relative_to(REPO))
            suf = cam.suffix.lower()
            no_nome = canon(n) or (canon(str(rel_root)) if suf in BINARIOS else [])
            catalogo = bool(CATALOGO_PALAVRAS.search(rel))
            for c in no_nome:
                if suf in BINARIOS:
                    cat = ("catalogo" if catalogo else
                           "rotulo_lido" if "label" in rel.lower() else "artefato_binario")
                else:
                    cat = "catalogo" if catalogo else "mencao_textual"
                L.append(linha(c, n, cat, frente(rel), cam, "nome do arquivo/caminho"))
            if suf in TEXTO:
                try:
                    tam = cam.stat().st_size
                except OSError:
                    continue
                if tam > LIMITE_TEXTO:
                    limites["texto_acima_do_limite"] += 1
                    continue
                t = _texto(cam)
                if not t or "EMSR" not in t.upper():
                    continue
                limites["texto_lido"] += 1
                chips = {f"EMSR{m}" for m in CHIP.findall(t)}
                for c in canon(t):
                    if catalogo:
                        cat = "catalogo"
                    elif c in chips and n.lower() != "plano.json":
                        cat = "uso_chip_sar"
                    else:
                        cat = "mencao_textual"
                    L.append(linha(c, n, cat, frente(rel), cam, "conteudo (chip rotulado)" if cat == "uso_chip_sar"
                                   else "conteudo"))
            elif suf == ".parquet":
                try:
                    esq = pq.read_schema(cam)
                except Exception:  # noqa: BLE001 -- parquet ilegivel fica fora e conta no limite
                    continue
                limites["parquet_esquemas"] += 1
                cols = [x for x in esq.names if x.lower() in colunas_id]
                rotulo = any(x.lower() in ("classe", "label", "y", "rotulo", "positivo") for x in esq.names)
                if not cols:
                    continue
                try:
                    tab = pq.read_table(cam, columns=cols)
                except Exception:  # noqa: BLE001
                    continue
                limites["parquet_lidos"] += 1
                vals = set()
                for col in cols:
                    s = tab.column(col)
                    if str(s.type) in ("string", "large_string", "dictionary<values=string, indices=int32, ordered=0>"):
                        vals |= {str(v) for v in pd.unique(s.to_pandas().dropna())}
                for c in canon(" ".join(vals)):
                    cat = "catalogo" if catalogo else "rotulo_lido" if rotulo else "artefato_binario"
                    L.append(linha(c, n, cat, frente(rel), cam, f"colunas {cols}" + (" + classe" if rotulo else "")))
    return L, limites


# ------------------------------------------------------------------ consolidacao e testes

def consolidar(L: pd.DataFrame) -> pd.DataFrame:
    L = L.assign(ordem=L.categoria.map({c: i for i, c in enumerate(ORDEM)}))
    linhas = []
    for code, g in L.groupby("code"):
        g = g.sort_values(["ordem", "caminho"])
        top = g.iloc[0]
        frentes_exame = sorted(set(g[g.categoria.isin(EXAMINADO)].frente))
        linhas.append({"code": code, "estado": top.categoria, "examinado": top.categoria in EXAMINADO,
                       "reservado": bool((g.categoria == "reserva_protegida").any()),
                       "planejado_outra_frente": bool((g.categoria == "planejado_outra_frente").any()),
                       "frentes_com_exame": ";".join(frentes_exame), "evidencias": len(g),
                       **{f"n_{c}": int((g.categoria == c).sum()) for c in ORDEM},
                       "caminho_principal": top.caminho, "campo_principal": top.campo})
    return pd.DataFrame(linhas)


def testes(ev: pd.DataFrame) -> dict:
    e = ev.set_index("code")
    abertos_mt04 = ler(V4 / "confirmacao" / "abertura.json")["ativacoes"]
    rodada2 = json.loads((RUNS / "referencia-fisica-forcamento/v1/confirmacao/elegibilidade.json").read_text(encoding="utf-8"))
    falsos_novos = rodada2["nao_examinadas_e_nao_reservadas"]
    so_catalogo = [c for c in e.index if e.loc[c, "estado"] == "catalogo"]
    t = {
        "mt04_abertos_todos_examinados": all(bool(e.loc[c, "examinado"]) for c in abertos_mt04),
        "falsos_novos_rodada2_saem_da_lista": {c: bool(e.loc[c, "examinado"]) for c in falsos_novos},
        "emsr471_detectado_pela_frente_sar": bool(e.loc["EMSR471", "examinado"]) and "SAR" in e.loc["EMSR471", "frentes_com_exame"],
        "catalogo_sem_classe_nao_vira_exame": bool(len(so_catalogo)) and not any(e.loc[so_catalogo, "examinado"]),
        "reservas_protegidas_marcadas": all(bool(e.loc[c, "reservado"]) for c in
                                            ("EMSR868", "EMSR871", "EMSR926", "EMSR927", "EMSR933", "EMSR692", "EMSR858", "EMSR866")),
    }
    t["todos_passam"] = all(v if isinstance(v, bool) else all(v.values()) for v in t.values())
    return t


def construir() -> dict:
    LED.mkdir(parents=True, exist_ok=True)
    est = estruturadas()
    gen, limites = generica()
    L = pd.DataFrame(est + gen).drop_duplicates()
    L.to_parquet(LED / "ledger_linhas.parquet", index=False)
    ev = consolidar(L)
    ev.to_csv(LED / "ledger_eventos.csv", index=False)
    t = testes(ev)
    r = {"tipo": "ledger_eventos_examinados_v2", "quando_utc": agora_utc(),
         "linhas": len(L), "linhas_estruturadas": len(est), "linhas_varredura": len(gen),
         "codigos": int(len(ev)), "por_estado": ev.estado.value_counts().to_dict(),
         "examinados": int(ev.examinado.sum()), "reservados": int(ev.reservado.sum()),
         "limites_da_varredura": limites, "testes_de_regressao": t,
         "ordem_de_evidencia": ORDEM,
         "saidas": {"ledger_linhas.parquet": sha256_arquivo(LED / "ledger_linhas.parquet"),
                    "ledger_eventos.csv": sha256_arquivo(LED / "ledger_eventos.csv")},
         "codigo": sha256_arquivo(Path(__file__)),
         "nao_e": "ausencia de linha nao certifica ausencia de exame; mencao textual e catalogo nao sao exame"}
    (LED / "ledger.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    if not t["todos_passam"]:
        raise SystemExit(f"regressao falhou: {t}")
    return r


def elegibilidade() -> dict:
    """Elegibilidade v2: inundacoes do catalogo historico (1.067) e da lista recente contra o ledger."""
    ev = pd.read_csv(LED / "ledger_eventos.csv").set_index("code")
    cat = ler(V4 / "triagem_historica" / "catalogo.json")["ativacoes"]
    linhas = []
    for a in cat:
        slug = (a.get("category") or {}).get("slug", "")
        if slug not in ("flood", "storm"):
            continue
        c = a["code"]
        x = ev.loc[c] if c in ev.index else None
        estado = ("reservado" if x is not None and x.reservado else
                  "examinado" if x is not None and x.examinado else
                  "planejado_outra_frente" if x is not None and x.planejado_outra_frente else
                  "nao_examinado_por_ledger")
        linhas.append({"code": c, "categoria": slug, "nome": a.get("name"), "ativacao": a.get("activationTime"),
                       "centroide": a.get("centroid"), "estado": estado,
                       "frentes_com_exame": None if x is None else x.frentes_com_exame,
                       "evidencia": None if x is None else f"{x.caminho_principal} :: {x.campo_principal}"})
    t = pd.DataFrame(linhas)
    t.to_csv(LED / "elegibilidade_v2.csv", index=False)
    r = {"tipo": "elegibilidade_v2", "quando_utc": agora_utc(), "universo": "catalogo historico local (1.067 ativacoes)",
         "inundacao_tempestade": len(t), "por_estado": t.estado.value_counts().to_dict(),
         "ledger": sha256_arquivo(LED / "ledger_eventos.csv"),
         "catalogo": sha256_arquivo(V4 / "triagem_historica" / "catalogo.json"),
         "substitui": "local_runs/referencia-fisica-forcamento/v1/confirmacao/elegibilidade.json (preservado; estado historico)",
         "nao_e": "nao examinado pelo ledger nao e independente: falta o grafo da AOI completa"}
    (LED / "elegibilidade_v2.json").write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    return r


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("acao", choices=["construir", "elegibilidade"])
    a = ap.parse_args()
    r = construir() if a.acao == "construir" else elegibilidade()
    print(json.dumps(r, ensure_ascii=False, indent=1, default=str)[:6000])
