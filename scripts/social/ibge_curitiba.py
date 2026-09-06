"""Integra IBGE/Censo 2022 ao contrato social do REV-P para Curitiba.

Entrada esperada: zips oficiais baixados para
`local_runs/camada-social-fontes/ibge/`.

Saida: setores de Curitiba convertidos em unidades sociais territoriais que a
camada `social_v0_review_only` consegue cruzar com AOI. O piloto e real no dado
IBGE, mas ainda nao e validacao social completa: usa somente basico, renda do
responsavel e atributos da malha de setores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "social"))

from camada_social import (  # noqa: E402
    BBox,
    DOMINIOS_SOCIAIS,
    UnidadeSocial,
    avaliar_aoi_social,
)


FONTES = {
    "malha_pr": {
        "arquivo": "PR_setores_CD2022.zip",
        "url": "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_intramunicipais/censo_2022/setores/shp/UF/PR_setores_CD2022.zip",
        "descricao": "IBGE malha definitiva de setores censitarios 2022 - PR",
    },
    "basico_br": {
        "arquivo": "Agregados_por_setores_basico_BR_20260520.zip",
        "url": "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/Agregados_por_Setor_csv/Agregados_por_setores_basico_BR_20260520.zip",
        "descricao": "IBGE Censo 2022 agregados basicos por setor - Brasil",
    },
    "renda_br": {
        "arquivo": "Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip",
        "url": "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios_Rendimento_do_Responsavel/Agregados_por_setores_renda_responsavel_BR_20260508_csv.zip",
        "descricao": "IBGE Censo 2022 rendimento do responsavel por setor - Brasil",
    },
    "dicionario_basico": {
        "arquivo": "dicionario_de_dados_agregados_por_setores_censitarios_20260520.xlsx",
        "url": "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios/dicionario_de_dados_agregados_por_setores_censitarios_20260520.xlsx",
        "descricao": "Dicionario de dados dos agregados por setores censitarios",
    },
    "dicionario_renda": {
        "arquivo": "dicionario_de_dados_renda_responsavel_20260508.xlsx",
        "url": "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Agregados_por_Setores_Censitarios_Rendimento_do_Responsavel/dicionario_de_dados_renda_responsavel_20260508.xlsx",
        "descricao": "Dicionario de dados de rendimento do responsavel",
    },
}

MUNICIPIO_CURITIBA = "4106902"
DEFAULT_FONTES = REPO / "local_runs" / "camada-social-fontes" / "ibge"
DEFAULT_OUT = REPO / "local_runs" / "camada-social-territorial" / "ibge-curitiba"


def parse_num(valor: Any) -> float | None:
    if valor is None:
        return None
    texto = str(valor).strip()
    if not texto or texto == ".":
        return None
    texto = texto.replace(".", "").replace(",", ".") if "," in texto else texto
    try:
        v = float(texto)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fonte_manifesto(fontes_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for fonte_id, meta in FONTES.items():
        path = fontes_dir / meta["arquivo"]
        rows.append({
            "fonte_id": fonte_id,
            "arquivo": meta["arquivo"],
            "url": meta["url"],
            "descricao": meta["descricao"],
            "existe": path.exists(),
            "bytes": path.stat().st_size if path.exists() else None,
            "sha256": sha256(path) if path.exists() else None,
        })
    return rows


def exigir_fontes(fontes_dir: Path) -> None:
    faltantes = [r["arquivo"] for r in fonte_manifesto(fontes_dir)
                 if not r["existe"]]
    if faltantes:
        raise FileNotFoundError(
            "fontes oficiais ausentes em "
            f"{fontes_dir}: {', '.join(faltantes)}")


def _csv_zip_rows(zip_path: Path) -> tuple[list[str], Any]:
    z = zipfile.ZipFile(zip_path)
    nomes = [n for n in z.namelist() if n.lower().endswith(".csv")]
    if len(nomes) != 1:
        raise ValueError(f"{zip_path.name}: esperado um CSV, encontrado {nomes}")
    f = z.open(nomes[0])
    text = io.TextIOWrapper(f, encoding="latin1", newline="")
    reader = csv.DictReader(text, delimiter=";")
    return reader.fieldnames or [], reader


def ler_agregado_filtrado(zip_path: Path, municipio: str) -> list[dict[str, str]]:
    _, reader = _csv_zip_rows(zip_path)
    rows = []
    for row in reader:
        cd_setor = str(row.get("CD_SETOR") or "")
        if cd_setor.startswith(municipio):
            rows.append(row)
    return rows


def ler_setores_malha(zip_path: Path, municipio: str) -> list[dict[str, Any]]:
    try:
        import fiona
        from shapely.geometry import shape
    except ImportError as e:
        raise RuntimeError("fiona e shapely sao necessarios para ler a malha") from e

    rows = []
    with fiona.open(f"zip://{zip_path}", encoding="latin1") as src:
        crs = src.crs.to_string() if hasattr(src.crs, "to_string") else str(src.crs)
        for feat in src:
            props = dict(feat["properties"])
            if props.get("CD_MUN") != municipio:
                continue
            geom = shape(feat["geometry"])
            minx, miny, maxx, maxy = geom.bounds
            props["bbox"] = {
                "lat_min": float(miny),
                "lon_min": float(minx),
                "lat_max": float(maxy),
                "lon_max": float(maxx),
            }
            props["crs"] = crs
            props["geometry_type"] = feat["geometry"]["type"]
            rows.append(props)
    return rows


def percentile_scores(values: dict[str, float],
                      invert: bool = False) -> dict[str, float]:
    pares = sorted((v, k) for k, v in values.items()
                   if v is not None and math.isfinite(v))
    n = len(pares)
    if n == 0:
        return {}
    if n == 1:
        return {pares[0][1]: 0.5}
    scores: dict[str, float] = {}
    for rank, (_, key) in enumerate(pares):
        p = rank / (n - 1)
        scores[key] = round(1.0 - p if invert else p, 4)
    return scores


def flag_fcu(row: dict[str, Any]) -> bool:
    valor = row.get("CD_FCU")
    return valor not in (None, "", ".", "None")


def construir_unidades_sociais(
    malha: list[dict[str, Any]],
    basico: list[dict[str, str]],
    renda: list[dict[str, str]],
) -> tuple[list[UnidadeSocial], list[dict[str, Any]], dict[str, Any]]:
    basico_por_setor = {r["CD_SETOR"]: r for r in basico}
    renda_por_setor = {r["CD_SETOR"]: r for r in renda}

    densidade = {}
    renda_mediana = {}
    moradores = {}
    imputacao = {}
    for setor in malha:
        cd = setor["CD_SETOR"]
        b = basico_por_setor.get(cd, {})
        r = renda_por_setor.get(cd, {})
        area = parse_num(setor.get("AREA_KM2"))
        pop = parse_num(b.get("v0001"))
        mediana = parse_num(r.get("V06006"))
        media_moradores = parse_num(b.get("v0005"))
        pct_imputado = parse_num(b.get("v0006"))
        if area and area > 0 and pop is not None:
            densidade[cd] = pop / area
        if mediana is not None:
            renda_mediana[cd] = mediana
        if media_moradores is not None:
            moradores[cd] = media_moradores
        if pct_imputado is not None:
            imputacao[cd] = pct_imputado

    score_densidade = percentile_scores(densidade)
    score_renda = percentile_scores(renda_mediana, invert=True)
    score_moradores = percentile_scores(moradores)
    score_imputacao = percentile_scores(imputacao)

    unidades = []
    tabela = []
    for setor in malha:
        cd = setor["CD_SETOR"]
        indicadores = {}
        if cd in score_densidade:
            indicadores["exposicao_populacional"] = score_densidade[cd]
        if cd in score_renda:
            indicadores["condicao_socioeconomica"] = score_renda[cd]
        if cd in score_moradores:
            indicadores["moradia_infraestrutura"] = score_moradores[cd]
        if flag_fcu(setor):
            indicadores["moradia_infraestrutura"] = max(
                indicadores.get("moradia_infraestrutura", 0.0), 1.0)
        if cd in score_imputacao:
            indicadores["lacuna_institucional"] = score_imputacao[cd]

        unidade = UnidadeSocial(
            unidade_id=f"ibge_setor_{cd}",
            nome=f"{setor.get('NM_MUN')} / {setor.get('NM_BAIRRO') or setor.get('NM_SUBDIST') or cd}",
            geometria=BBox.from_dict(setor["bbox"]),
            indicadores=indicadores,
            fonte="ibge_censo_2022_setores_curitiba",
            observacao=(
                "indicadores relativos dentro de Curitiba; "
                "bbox da geometria setorial usado no piloto v1"),
        )
        unidades.append(unidade)
        tabela.append({
            "CD_SETOR": cd,
            "NM_BAIRRO": setor.get("NM_BAIRRO"),
            "AREA_KM2": setor.get("AREA_KM2"),
            "populacao_total_v0001": parse_num(basico_por_setor.get(cd, {}).get("v0001")),
            "densidade_pop_km2": round(densidade[cd], 4) if cd in densidade else None,
            "renda_mediana_responsavel_v06006": renda_mediana.get(cd),
            "media_moradores_v0005": moradores.get(cd),
            "pct_domicilios_imputados_v0006": imputacao.get(cd),
            "em_tipologia_territorial_ibge": flag_fcu(setor),
            "tipologia_territorial_fonte": (
                "IBGE Censo 2022 - Favelas e Comunidades Urbanas"
                if flag_fcu(setor) else None),
            "dominios_disponiveis": ";".join(sorted(indicadores)),
            **{f"score_{d}": indicadores.get(d) for d in DOMINIOS_SOCIAIS},
        })

    cobertura = Counter()
    for row in tabela:
        for dominio in DOMINIOS_SOCIAIS:
            if row.get(f"score_{dominio}") is not None:
                cobertura[dominio] += 1
    resumo = {
        "setores_malha": len(malha),
        "setores_basico": len(basico),
        "setores_renda": len(renda),
        "setores_com_unidade_social": len(unidades),
        "setores_em_fcu": sum(1 for s in malha if flag_fcu(s)),
        "dominios_sociais": list(DOMINIOS_SOCIAIS),
        "cobertura_por_dominio": dict(cobertura),
        "metodo_indicadores": {
            "exposicao_populacional": "percentil municipal da densidade populacional",
            "condicao_socioeconomica": "percentil municipal invertido da renda mediana do responsavel",
            "moradia_infraestrutura": "maximo entre percentil municipal da media de moradores e indicador agregado de tipologia territorial oficial do IBGE",
            "lacuna_institucional": "percentil municipal de domicilios imputados no Censo; nao mede chamado de enchente",
        },
    }
    return unidades, tabela, resumo


def unidade_to_json(unidade: UnidadeSocial) -> dict[str, Any]:
    return {
        "unidade_id": unidade.unidade_id,
        "nome": unidade.nome,
        "geometria": unidade.geometria.as_dict(),
        "indicadores": unidade.indicadores,
        "fonte": unidade.fonte,
        "observacao": unidade.observacao,
    }


def escrever_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def escrever_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def escolher_aoi_por_score(tabela: list[dict[str, Any]],
                           dominio: str,
                           maior: bool) -> dict[str, Any]:
    candidatos = [r for r in tabela if r.get(f"score_{dominio}") is not None]
    if not candidatos:
        raise ValueError(f"sem candidatos para {dominio}")
    alvo = max(candidatos, key=lambda r: r[f"score_{dominio}"]) if maior else min(
        candidatos, key=lambda r: r[f"score_{dominio}"])
    return alvo


def experimento_piloto_real(unidades: list[UnidadeSocial],
                            tabela: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alto = escolher_aoi_por_score(tabela, "condicao_socioeconomica", maior=True)
    baixo = escolher_aoi_por_score(tabela, "condicao_socioeconomica", maior=False)
    casos = []
    for nome, row in (("curitiba_ibge_alta_vulnerabilidade", alto),
                      ("curitiba_ibge_baixa_vulnerabilidade", baixo)):
        unidade = next(u for u in unidades
                       if u.unidade_id == f"ibge_setor_{row['CD_SETOR']}")
        r = avaliar_aoi_social(
            unidade.geometria.as_dict(),
            unidades,
            {"status": "ok", "escore": 0.82,
             "maturidade": "transferencia_caracterizada"},
            cobertura_minima=0.05,
        )
        r["caso"] = nome
        r["setor_base"] = row["CD_SETOR"]
        r["bairro"] = row.get("NM_BAIRRO")
        casos.append(r)
    return casos


def relatorio(resultado: dict[str, Any]) -> str:
    resumo = resultado["resumo"]
    piloto = resultado["piloto_real"]
    linhas = [
        "# Piloto social real - IBGE Curitiba",
        "",
        "Este artefato integra dados oficiais do IBGE/Censo 2022 para criar a",
        "primeira camada social territorial real do REV-P em Curitiba. O piloto",
        "usa indicadores relativos dentro do municipio e nao altera nenhum escore",
        "fisico-hidrologico.",
        "",
        "## Fontes",
        "",
    ]
    for f in resultado["manifesto_fontes"]:
        linhas.append(
            f"- `{f['fonte_id']}`: {f['arquivo']} ({f['bytes']} bytes, sha256 `{f['sha256']}`)"
        )
    linhas += [
        "",
        "## Resultado",
        "",
        f"- setores na malha de Curitiba: {resumo['setores_malha']}",
        f"- setores com agregados basicos: {resumo['setores_basico']}",
        f"- setores com renda do responsavel: {resumo['setores_renda']}",
        f"- unidades sociais geradas: {resumo['setores_com_unidade_social']}",
        f"- setores com indicador agregado de tipologia territorial oficial do IBGE: {resumo['setores_em_fcu']}",
        "",
        "## Cobertura por dominio",
        "",
    ]
    for d, n in resumo["cobertura_por_dominio"].items():
        linhas.append(f"- {d}: {n} setores")
    linhas += [
        "",
        "## Piloto de nao-colapso com dado real",
        "",
        "| caso | setor | fisica | social | prioridade |",
        "|---|---|---|---|---|",
    ]
    for c in piloto:
        fis = c["eixos"]["suscetibilidade_fisica"]
        soc = c["eixos"]["vulnerabilidade_exposicao"]
        linhas.append(
            f"| {c['caso']} | {c['setor_base']} | {fis['classe_relativa']} | "
            f"{soc['classe_relativa']} | {c['prioridade_publica']['classe']} |"
        )
    linhas += [
        "",
        "## Limites",
        "",
        "- O recorte ainda usa bbox de setor, nao intersecao poligonal completa.",
        "- Renda, densidade e media de moradores nao esgotam vulnerabilidade social.",
        "- O indicador de imputacao censitaria nao mede subregistro de chamado de enchente.",
        "- Curitiba continua sendo laboratorio metodologico, nao centro final do projeto.",
        "",
    ]
    return "\n".join(linhas)


def executar(fontes_dir: Path, output_dir: Path,
             municipio: str = MUNICIPIO_CURITIBA) -> dict[str, Any]:
    exigir_fontes(fontes_dir)
    manifesto = fonte_manifesto(fontes_dir)
    malha = ler_setores_malha(fontes_dir / FONTES["malha_pr"]["arquivo"],
                              municipio)
    basico = ler_agregado_filtrado(fontes_dir / FONTES["basico_br"]["arquivo"],
                                   municipio)
    renda = ler_agregado_filtrado(fontes_dir / FONTES["renda_br"]["arquivo"],
                                  municipio)
    unidades, tabela, resumo = construir_unidades_sociais(malha, basico, renda)
    piloto = experimento_piloto_real(unidades, tabela)

    resultado = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "municipio": municipio,
        "contrato": "social_v0_review_only",
        "metodo": "IBGE Censo 2022 por setor; indicadores relativos municipais",
        "manifesto_fontes": manifesto,
        "resumo": resumo,
        "unidades_sociais": [unidade_to_json(u) for u in unidades],
        "piloto_real": piloto,
        "invariantes": {
            "nao_altera_escore_fisico": True,
            "nao_treina_modelo": True,
            "nao_cria_rotulo": True,
            "ausencia_social_nao_e_baixa_vulnerabilidade": True,
            "vulnerabilidade_nao_e_causa_hidrologica": True,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    escrever_json(output_dir / "unidades_sociais_curitiba_ibge_v1.json",
                  {"unidades_sociais": resultado["unidades_sociais"]})
    escrever_json(output_dir / "manifesto_ibge_curitiba_v1.json", resultado)
    escrever_csv(output_dir / "setores_curitiba_indicadores_sociais_v1.csv",
                 tabela)
    escrever_json(output_dir / "piloto_nao_colapso_curitiba_ibge_v1.json",
                  piloto)
    (output_dir / "relatorio_ibge_curitiba_v1.md").write_text(
        relatorio(resultado), encoding="utf-8")
    (output_dir / "commands.txt").write_text(
        "python scripts/social/ibge_curitiba.py\n"
        "python -m pytest tests/test_camada_social.py tests/test_ibge_curitiba.py -q\n",
        encoding="utf-8")
    return resultado


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integra dados oficiais do IBGE para piloto social de Curitiba.")
    parser.add_argument("--fontes-dir", default=str(DEFAULT_FONTES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--municipio", default=MUNICIPIO_CURITIBA)
    return parser.parse_args()


def main() -> int:
    ns = parse_args()
    resultado = executar(Path(ns.fontes_dir), Path(ns.output_dir), ns.municipio)
    resumo = resultado["resumo"]
    print("===IBGE CURITIBA SOCIAL===")
    print(f"municipio={resultado['municipio']}")
    print(f"setores={resumo['setores_com_unidade_social']}")
    print(f"fcu={resumo['setores_em_fcu']}")
    print(f"dominios={resumo['cobertura_por_dominio']}")
    print(f"GRAVADO={Path(ns.output_dir)}")
    print("===END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
