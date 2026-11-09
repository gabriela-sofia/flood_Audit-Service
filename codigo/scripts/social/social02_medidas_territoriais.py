"""Medidas sociais separadas: base setorial, exposição delimitada e observabilidade.

Reutiliza fontes IBGE locais com polígonos reais. Não estima perdas, não
atribui médias territoriais a pessoas e não altera o escore físico.
"""
from __future__ import annotations


def _declarar_metadados_de_saida(dados):
    """Metadados de procedencia na copia serializada; nao modifica o objeto de calculo."""
    if not isinstance(dados, dict):
        return dados
    return {"tipo": "registro_social02_medidas_territoriais",
            "gerado_por": 'scripts/social/social02_medidas_territoriais.py',
            "nao_e": "Registro do produtor social02_medidas_territoriais: declarar a procedencia nao revalida resultados nem autoriza promocao; conservar os estados e limites especificos do artefato.",
            **dados}


import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FONTES = REPO / "local_runs/camada-social-fontes/ibge"
SAIDA = REPO / "local_runs/social-medidas-territoriais/2026-10-11"
VARS = {"v0001": "populacao", "v0002": "domicilios_total", "v0007": "domicilios_ocupados",
        "v0005": "media_moradores_dpo", "v0006": "percentual_dpo_imputados", "V06006": "renda_mediana_responsavel_com_rendimento"}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def numero(v):
    if v is None or str(v).strip() in ("", ".", "...", "X", "-"): return None
    try: n = float(str(v).replace(",", "."))
    except ValueError: raise ValueError(f"valor numérico inesperado: {v}")
    if not math.isfinite(n) or n < 0: raise ValueError("medida não finita ou negativa")
    return n


def gravar(p, d):
    p = Path(p)
    txt = json.dumps(_declarar_metadados_de_saida(d), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if p.exists() and p.read_text(encoding="utf-8") != txt:
        raise ValueError(f"preservar artefato existente; usar nova pasta: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt, encoding="utf-8")


def verificar_fontes(pasta):
    pasta = Path(pasta)
    m = json.loads((pasta / "manifesto_aquisicao_fontes_sociais.json").read_text(encoding="utf-8"))
    regs = {}
    for r in m["fontes"]:
        p = pasta / r["arquivo"]
        if not p.is_file() or sha(p) != r["sha256"]: raise ValueError(f"fonte ausente/alterada: {p.name}")
        regs[p.name] = {"sha256": r["sha256"], "url": r["url"]}
    return regs


def dicionario(pasta):
    import openpyxl
    out = {}
    for p in sorted(Path(pasta).glob("dicionario*.xlsx")):
        w = openpyxl.load_workbook(p, read_only=True)
        for s in w.worksheets:
            for r in s.values:
                if len(r) >= 3 and str(r[1]).lower() in {c.lower() for c in VARS}:
                    out[str(r[1]).lower()] = {"definicao": str(r[2]), "arquivo": p.name, "aba": s.title}
        w.close()
    if set(out) != {c.lower() for c in VARS}: raise ValueError("variáveis sem definição oficial")
    return out


def ler_agregado(p, municipio):
    with zipfile.ZipFile(p) as z:
        csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if len(csvs) != 1: raise ValueError("esperado único CSV")
        with z.open(csvs[0]) as f:
            rd = csv.DictReader(io.TextIOWrapper(f, encoding="latin1"), delimiter=";")
            out = {}
            for r in rd:
                if r["CD_SETOR"].startswith(municipio):
                    if r["CD_SETOR"] in out: raise ValueError("setor duplicado no agregado")
                    out[r["CD_SETOR"]] = r
    return out


def limites_exposicao(unidades, geometria, tipo_recorte, ano_base=2022):
    """Limites de localização para totais setoriais, NÃO intervalo de confiança.

    Sem localização intra-setor, interseção parcial não permite contar pessoas
    por proporção de área. Intervalo [inteiramente contidos, todos os tocados].
    O chamador deve fornecer geometria e unidades no MESMO CRS projetado.
    """
    if tipo_recorte not in ("agua_observada", "cenario_suscetibilidade", "recorte_de_estudo"):
        raise ValueError("declarar tipo do recorte; susceptibilidade não é água observada")
    if geometria.is_empty or not geometria.is_valid: raise ValueError("geometria inválida")
    ids = [u["id"] for u in unidades]
    if len(set(ids)) != len(ids): raise ValueError("unidades duplicadas")
    out = {k: {"minimo": 0, "maximo": 0, "unidades_sem_total": 0} for k in ("populacao", "domicilios_ocupados")}
    touched = []
    for u in unidades:
        g = u["geometria"]
        if g.is_empty or not g.is_valid or g.area <= 0: raise ValueError("setor inválido")
        inter = g.intersection(geometria)
        if inter.area <= 0: continue  # contato de borda não conta
        inteiro = geometria.covers(g)
        touched.append({"id": u["id"], "inteiro": inteiro, "fracao_area": inter.area / g.area})
        for k in out:
            v = numero(u.get(k))
            if v is None:
                out[k]["unidades_sem_total"] += 1
            else:
                if not v.is_integer(): raise ValueError("contagem censitária deve ser inteira")
                out[k]["maximo"] += int(v)
                if inteiro: out[k]["minimo"] += int(v)
    for k in out:
        if out[k]["unidades_sem_total"]:
            out[k]["maximo_somente_unidades_com_dado"] = out[k]["maximo"]
            out[k]["maximo"] = None  # não é limite global se faltam totais
    return {"tipo_recorte": tipo_recorte, "ano_base": ano_base, "limites_localizacao": out,
            "unidades_intersectadas": touched, "estimativa_por_area": None,
            "limite": "totais de 2022; sem identificação de indivíduos, sem impacto/perda, sem precisão intra-setor ou atualização temporal automática"}


def cobertura_por_oportunidade(linhas):
    """Comparação descritiva por estrato social dentro de evento/sensor/regime.

    Exige áreas: área em footprint/visada e área detectável após controles de
    qualidade. Nuvem/oclusão não é pouca água. Relaciona suporte espacial, sem
    inferir pessoas observadas nem chance conhecida de detecção por indivíduo.
    """
    vistos, grupos = set(), {}
    for r in linhas:
        for k in ("unidade_id", "evento_canonico", "cena_id", "sensor", "regime", "estrato_social", "evidencia_oportunidade"):
            if not r.get(k): raise ValueError(f"metadado de oportunidade ausente: {k}")
        chave = (r["unidade_id"], r["evento_canonico"], r["cena_id"], r["sensor"])
        if chave in vistos: raise ValueError("oportunidade duplicada")
        vistos.add(chave)
        areas = [numero(r.get(k)) for k in ("area_visada_m2", "area_detectavel_m2", "area_referencia_admitida_m2")]
        if any(a is None for a in areas): raise ValueError("área ausente não é zero")
        visada, detectavel, referencia = areas
        if not 0 <= referencia <= detectavel <= visada: raise ValueError("referência <= detectável <= visada é obrigatório")
        if referencia and r.get("instrumento_admitido") != "sim": raise ValueError("referência sem instrumento admitido")
        g = grupos.setdefault((r["evento_canonico"], r["sensor"], r["regime"], r["estrato_social"]),
                              {"area_visada_m2": 0, "area_detectavel_m2": 0, "area_referencia_admitida_m2": 0, "unidades": set()})
        for k, v in zip(("area_visada_m2", "area_detectavel_m2", "area_referencia_admitida_m2"), areas): g[k] += v
        g["unidades"].add(r["unidade_id"])
    medidas = []
    for chave, g in sorted(grupos.items()):
        v, d, a = [g[k] for k in ("area_visada_m2", "area_detectavel_m2", "area_referencia_admitida_m2")]
        medidas.append(dict(zip(("evento", "sensor", "regime", "estrato_social"), chave),
                            area_visada_m2=v, area_detectavel_m2=d, area_referencia_admitida_m2=a,
                            fracao_detectavel=d/v if v else None, cobertura_cientifica_condicional=a/d if d else None,
                            unidades=len(g["unidades"])))
    return {"estado": "descritivo" if linhas else "not_testable", "medidas": medidas,
            "suporte": "área-visada somada por cena, não área territorial única; agrupamento por evento",
            "desigualdade_confirmada": False,
            "limite": "comparações exigem protocolo, eventos independentes e controle de regime/visada; associação não atribui exclusão institucional ou causalidade"}


def executar(pasta=FONTES, saida=SAIDA, municipio="4106902"):
    import geopandas as gpd
    if not re.fullmatch(r"\d{7}", municipio): raise ValueError("código municipal inválido")
    pasta, saida = Path(pasta), Path(saida)
    fontes = verificar_fontes(pasta)
    defs = dicionario(pasta)
    bas = ler_agregado(pasta / "Agregados_por_setores_basico_BR_20260520.zip", municipio)
    ren = ler_agregado(pasta / "Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip", municipio)
    malha = gpd.read_file(pasta / "PR_setores_CD2022.zip", where=f"CD_MUN = '{municipio}'", engine="pyogrio")
    if malha.empty: raise ValueError("município não coberto pela malha local PR")
    if malha.crs is None or not malha.geometry.is_valid.all() or malha.CD_SETOR.duplicated().any():
        raise ValueError("malha sem CRS, inválida ou duplicada")
    faltantes = set(malha.CD_SETOR) - set(bas)
    if faltantes: raise ValueError("setores sem agregado básico; reconciliar antes")
    # Sem percentis ou média de eixos. Mantém grandezas e denominadores originais.
    rows = []
    for cid in malha.CD_SETOR:
        b, r = bas[cid], ren.get(cid, {})
        row = {"CD_SETOR": cid, "ano": 2022}
        for c, nome in VARS.items(): row[nome] = numero((r if c.startswith("V06") else b).get(c))
        row["estado_renda"] = "disponivel" if row[VARS["V06006"]] is not None else "ausente_ou_suprimida"
        rows.append(row)
    import pandas as pd
    tabela = pd.DataFrame(rows)
    for c in ("populacao", "domicilios_total", "domicilios_ocupados"):
        if ((tabela[c].dropna() % 1) != 0).any(): raise ValueError("contagens não inteiras")
    out = malha[["CD_SETOR", "NM_BAIRRO", "geometry"]].merge(tabela, on="CD_SETOR", validate="one_to_one")
    saida.mkdir(parents=True, exist_ok=True)
    # Vetor por setor; valores não são atribuídos a cada pessoa/pixel.
    geo = json.loads(out.to_crs(4326).to_json(drop_id=True, na="null"))
    gravar(saida / "setores_contexto.geojson", geo)
    classes = {c: {"setores_com_dado": int(out[c].notna().sum()), "setores_sem_dado": int(out[c].isna().sum())}
               for c in VARS.values()}
    m = {"tipo": "social02_contexto_setorial", "municipio": municipio, "ano_referencia": 2022,
         "versao_publicacao": "fontes locais maio/2026, hashes conferidos", "setores": len(out), "crs_origem": str(malha.crs),
         "fontes": fontes, "dicionario": defs, "cobertura_variaveis": classes,
         "totais_no_recorte_com_dado": {c: int(out[c].sum()) for c in ("populacao", "domicilios_total", "domicilios_ocupados")},
         "geometria": "polígonos setoriais reais; nenhum bbox substitui o setor",
         "arquivo_sha256": sha(saida / "setores_contexto.geojson"), "codigo_sha256": sha(__file__),
         "exposicao": {"estado": "not_testable", "falta": "recorte de água ou cenário físico elegível com suporte/tempo documentados"},
         "vulnerabilidade": {"estado": "componentes_descritivos", "indice_validado": False,
                              "limite": "renda mediana do responsável COM rendimentos não é renda per capita ou índice de pobreza; ocupação domiciliar não é infraestrutura"},
         "cobertura_cientifica": {"estado": "not_testable", "falta": "ledger de oportunidade de detecção e referência admitida nas mesmas unidades"},
         "avaliacao_prioridade_publica": "nao_executada", "avaliacao_compreensao": "nao_executada",
         "usado_no_modelo_fisico": False, "promovido": False}
    gravar(saida / "manifesto.json", m)
    return {k: m[k] for k in ("setores", "totais_no_recorte_com_dado", "cobertura_variaveis", "exposicao", "vulnerabilidade", "cobertura_cientifica")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fontes", type=Path, default=FONTES)
    ap.add_argument("--saida", type=Path, default=SAIDA)
    ap.add_argument("--municipio", default="4106902")
    a = ap.parse_args()
    print(json.dumps(executar(a.fontes, a.saida, a.municipio), ensure_ascii=False))
