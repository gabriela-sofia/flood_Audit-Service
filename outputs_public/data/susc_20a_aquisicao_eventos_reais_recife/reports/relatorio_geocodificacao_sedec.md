# Recife SEDEC Geocoding Completion — Report

**Date executed**: 2026-07-21
**Method**: Live Nominatim (OpenStreetMap) geocoding, 1.1s rate limit, `countrycodes=br`, `addressdetails=1`, bbox-validated against Recife (-8.35..-7.85 lat, -35.20..-34.75 lon).
**Script**: `local_runs/recife_geocoding_completion/run_geocoding.py`
**Raw per-record log**: `progress.jsonl` (289 lines, one JSON object per QA record)

## Correction to prior framing

The task brief referenced a prior finding of "71 piloted, 6 reached strong confidence via Nominatim." That claim does **not** match the on-disk evidence. The actual prior pilot file
(`outputs/external_validation/recife_external_seced_geocoding_pilot_71_results_v1.csv`) shows **all 71** piloted records marked `blocked_no_safe_local_geocoder_available` /
`geocoded_successfully=False` — i.e., **zero** live geocode calls were made in that pilot. The pipeline's own status file (`recife_external_seced_geocoding_status.csv`) explicitly states
*"no automatic geocoding executed... No external geocoding API was called."* The pilot's `allowed_claim`/`prohibited_claim` columns flagged this as a **future controlled pilot pending
decision**, not a completed result. This run is that controlled pilot, executed for the first time with live external calls, covering all 289 QA-ready records (not just the 71) for full
methodological consistency, since none of the 71 had actually been geocoded before.

## Source and filtering

- Input: `outputs/external_validation/recife_external_seced_geocoding_qa_ready_records_v1.csv` (289 rows), derived from 11 yearly Recife Defesa Civil "registro de atendimentos" CSVs
  (2014–2025, `data/raw/recife/seced_defesa_civil/`).
- Occurrence-type check: verified programmatically that none of the 289 `hydrological_category_summary` values contain landslide/mass-movement keywords (`deslizam`, `movimento de massa`,
  `desabamento`, `barreira`, `cicatriz`, `encosta`). **0 records excluded** — the QA-ready set was already flood-only (Alagado/Alagamentos/Monitoramento Alagado/Vistoria Alagado categories)
  from the upstream pipeline stage; this run re-verified that filter rather than assuming it.
- Address handling: source `Endereco` field has its house-number suffix masked with symbol garbage at the source (confirmed in the raw CSVs themselves — this is government-side redaction,
  not something introduced by prior processing). Street name only (number stripped) + `Bairro` + "Recife, PE, Brasil" was used as the query string, consistent with the project's own
  `address_and_neighborhood_but_number_masked_or_uncertain` characterization.

## Confidence tiers (method, reused/extended from project convention)

- **strong** — Nominatim returned a street-level (`highway`/road-class) result inside the Recife bbox, i.e. an exact-street match. Reflects the street's geometry, **not** an exact house
  number (numbers are unavailable in the source data).
- **medium** — No street-level match; fell back to a neighbourhood/suburb/place-level result inside the bbox (bairro-centroid proxy).
- **failed** — No in-bbox result at all. **No coordinate fabricated**; logged with reason in `recife_ungeocoded_records_log.csv`.

## Results

| Tier | Count |
|---|---|
| strong (street-level) | 132 |
| medium (bairro-level fallback) | 9 |
| **Total geocoded (strong+medium)** | **141** |
| failed (no match, not geocoded) | 148 |
| excluded (landslide) | 0 |
| **Total processed** | **289** |

- **141 of 289 (48.8%)** flood-only records were successfully geocoded to a point with documented confidence and full source traceability (`qa_record_id`, `source_file`,
  `source_row_index`).
- **132 reach "strong" (street-level) confidence** — well above the ~15-20 high-confidence-point bar mentioned as the target threshold.
- Failures were not retried with fuzzy matching or interpolation; they remain explicitly marked `failed` with a reason string (e.g., "no in-bbox result returned").

## Outputs

- `local_runs/recife_geocoding_completion/recife_inundacao_geocoded_final.geojson` — 141 point features (strong + medium), each with `confidence_tier`, `confidence_definition`,
  `qa_record_id`, `source_file`, `source_row_index`, raw Nominatim response fields (`nominatim_display_name`, `class`, `type`, `osm_id`), and an explicit `prohibited_use` note.
- `local_runs/recife_geocoding_completion/recife_ungeocoded_records_log.csv` — 148 rows, one per failed record, with query string and failure reason (no fabricated coordinates).
- `local_runs/recife_geocoding_completion/progress.jsonl` — full raw per-record processing log (all 289), the audit trail for this run.

## Non-negotiable principles — compliance

- No fabricated coordinates: every point in the GeoJSON came from a live Nominatim response; failures were left uncoordinated.
- No interpolation: no spatial interpolation, centroid-of-centroid, or nearest-neighbor substitution was used.
- Flood-only: landslide/mass-movement keyword check applied and returned zero exclusions (already filtered upstream); documented rather than assumed.
- Explicit confidence: every feature carries a tier and the tier's operational definition.
- This output is **not** a REV-P canonical integration; it is a completed research artifact for TCC evidence purposes only, consistent with the source QA file's own `prohibited_use`
  language, carried forward into each output feature's `prohibited_use` field.
