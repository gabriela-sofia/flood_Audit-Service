"""Geocodifica endereços de reclamações SIAC 156 via Nominatim (OpenStreetMap), live, sem
credencial paga -- mesmo método/mesma régua de confiança já usada e documentada em
`docs/FONTES.md` §1 (SEDEC
Recife), reaplicado aqui pra manter a comparação entre regiões consistente.

Query = "{Logradouro}, {Bairro}, Curitiba, PR, Brasil" (uma tentativa; se falhar, tenta sem o
Logradouro, só bairro -- fallback de bairro-centroide, não fabricação de número).

Camadas de confiança (idênticas ao relatório do Recife):
  - strong: resultado nível rua (osm class 'highway' ou address contém 'road') dentro da bbox
  - medium: sem nível de rua, mas achou algo nível bairro/suburb/place dentro da bbox
  - failed: nenhum resultado dentro da bbox -- NENHUMA coordenada é fabricada

Rate limit: 1.1s entre chamadas (política de uso justo do Nominatim público -- max 1 req/s),
User-Agent identificado com contato, conforme https://operations.osmfoundation.org/policies/nominatim/.

Cache local em JSON (`--cache`) pra nunca repetir uma consulta já respondida -- mínimo de tráfego
necessário, e permite retomar uma rodada interrompida sem re-consultar o que já foi resolvido.

Uso:
    python geocodificar_nominatim.py --candidates candidatos.csv --out geocoded.csv \
        --cache cache_nominatim.json --bbox -49.45,-25.65,-49.10,-25.30
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "REV-P-TCC-Enchentes-Curitiba/1.0 (pesquisa academica; contato: gabrielasofia.bcc@gmail.com)"
RATE_LIMIT_S = 1.1


@dataclass
class GeocodeResult:
    query: str
    nivel_confianca: str  # strong | medium | failed
    lat: float | None
    lon: float | None
    nominatim_nome_exibicao: str | None
    osm_class: str | None
    osm_type: str | None
    osm_id: str | None
    reason: str | None


def _atomic_write_json(path: Path | str, data: dict) -> None:
    """Escreve via arquivo temporário + rename -- evita corromper o cache se o processo for
    interrompido no meio da escrita (aconteceu com timeout de shell durante a rodada real)."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float]) -> bool:
    lon_min, lat_min, lon_max, lat_max = bbox
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max


def _call_nominatim(query: str, opener: urllib.request.OpenerDirector) -> list[dict]:
    params = {
        "q": query,
        "format": "jsonv2",
        "addressdetails": 1,
        "countrycodes": "br",
        "limit": 5,
    }
    url = NOMINATIM_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with opener.open(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def classify_result(results: list[dict], bbox: tuple[float, float, float, float]) -> GeocodeResult | None:
    """Aplica a régua strong/medium a uma lista de resultados já filtrada pra dentro da bbox.
    Retorna None se nenhum resultado cair dentro da bbox (chamador decide o 'failed')."""
    in_bbox = []
    for r in results:
        try:
            lat, lon = float(r["lat"]), float(r["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if _in_bbox(lat, lon, bbox):
            in_bbox.append((r, lat, lon))
    if not in_bbox:
        return None

    # prioriza nível de rua (class=highway) se existir algum
    for r, lat, lon in in_bbox:
        if r.get("class") == "highway" or r.get("type") in ("residential", "primary", "secondary", "tertiary", "unclassified"):
            return GeocodeResult(
                query="", nivel_confianca="strong", lat=lat, lon=lon,
                nominatim_nome_exibicao=r.get("display_name"), osm_class=r.get("class"),
                osm_type=r.get("type"), osm_id=str(r.get("osm_id")), reason=None,
            )
    r, lat, lon = in_bbox[0]
    return GeocodeResult(
        query="", nivel_confianca="medium", lat=lat, lon=lon,
        nominatim_nome_exibicao=r.get("display_name"), osm_class=r.get("class"),
        osm_type=r.get("type"), osm_id=str(r.get("osm_id")), reason=None,
    )


def geocode_one(
    logradouro: str,
    bairro: str,
    city_state_country: str,
    bbox: tuple[float, float, float, float],
    opener: urllib.request.OpenerDirector,
    cache: dict,
) -> GeocodeResult:
    query_full = f"{logradouro}, {bairro}, {city_state_country}".strip(", ")
    if query_full in cache:
        c = cache[query_full]
        return GeocodeResult(query=query_full, **c)

    result: GeocodeResult
    try:
        results = _call_nominatim(query_full, opener)
        classified = classify_result(results, bbox)
        if classified is not None:
            classified.query = query_full
            result = classified
        else:
            # fallback: só bairro (bairro-centroide)
            query_bairro = f"{bairro}, {city_state_country}".strip(", ")
            time.sleep(RATE_LIMIT_S)
            results_b = _call_nominatim(query_bairro, opener)
            classified_b = classify_result(results_b, bbox)
            if classified_b is not None:
                classified_b.query = query_full
                classified_b.nivel_confianca = "medium"
                classified_b.reason = "fallback_bairro_apos_rua_sem_resultado_em_bbox"
                result = classified_b
            else:
                result = GeocodeResult(
                    query=query_full, nivel_confianca="failed", lat=None, lon=None,
                    nominatim_nome_exibicao=None, osm_class=None, osm_type=None, osm_id=None,
                    reason="nenhum_resultado_dentro_da_bbox",
                )
    except Exception as exc:  # noqa: BLE001 -- geocoding real, precisa seguir mesmo se 1 falhar
        result = GeocodeResult(
            query=query_full, nivel_confianca="failed", lat=None, lon=None,
            nominatim_nome_exibicao=None, osm_class=None, osm_type=None, osm_id=None,
            reason=f"erro_http_ou_parsing: {exc}",
        )

    cache[query_full] = {
        "nivel_confianca": result.nivel_confianca, "lat": result.lat, "lon": result.lon,
        "nominatim_nome_exibicao": result.nominatim_nome_exibicao, "osm_class": result.osm_class,
        "osm_type": result.osm_type, "osm_id": result.osm_id, "reason": result.reason,
    }
    return result


def geocode_candidates(
    candidates: list[dict],
    city_state_country: str,
    bbox: tuple[float, float, float, float],
    cache_path: Path | str | None = None,
    sleep_s: float = RATE_LIMIT_S,
) -> list[dict]:
    cache: dict = {}
    if cache_path and Path(cache_path).exists():
        try:
            cache = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # cache truncado por interrupção anterior (escrita não-atômica de uma versão
            # antiga do script) -- recomeça do zero em vez de travar a rodada real.
            cache = {}

    opener = urllib.request.build_opener()
    out = []
    n_calls = 0
    for row in candidates:
        query_key_precheck = f"{row['logradouro']}, {row['bairro']}, {city_state_country}".strip(", ")
        was_cached = query_key_precheck in cache
        res = geocode_one(row["logradouro"], row["bairro"], city_state_country, bbox, opener, cache)
        merged = dict(row)
        merged.update(
            {
                "consulta_geocodificacao": res.query,
                "nivel_confianca": res.nivel_confianca,
                "lat": res.lat,
                "lon": res.lon,
                "nominatim_nome_exibicao": res.nominatim_nome_exibicao,
                "osm_class": res.osm_class,
                "osm_type": res.osm_type,
                "osm_id": res.osm_id,
                "motivo_geocodificacao": res.reason,
            }
        )
        out.append(merged)
        if not was_cached:
            n_calls += 1
            if cache_path:
                _atomic_write_json(cache_path, cache)
            time.sleep(sleep_s)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", required=True, help="CSV de extrair_reclamacoes_enchente_siac156.py")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None)
    p.add_argument("--city-state-country", default="Curitiba, PR, Brasil")
    p.add_argument("--bbox", required=True, help="lon_min,lat_min,lon_max,lat_max")
    args = p.parse_args(argv)

    bbox = tuple(float(v) for v in args.bbox.split(","))
    with open(args.candidates, encoding="utf-8", newline="") as f:
        candidates = list(csv.DictReader(f))

    geocoded = geocode_candidates(candidates, args.city_state_country, bbox, args.cache)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(geocoded[0].keys()) if geocoded else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in geocoded:
            writer.writerow(row)

    n_strong = sum(1 for r in geocoded if r["nivel_confianca"] == "strong")
    n_medium = sum(1 for r in geocoded if r["nivel_confianca"] == "medium")
    n_failed = sum(1 for r in geocoded if r["nivel_confianca"] == "failed")
    print(json.dumps({"total": len(geocoded), "strong": n_strong, "medium": n_medium, "failed": n_failed}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
