# External data sources used in the validation study

| Source | What | Where on disk | Retrieved | Provenance |
|---|---|---|---|---|
| FHWA TMAS 2025 station file | Continuous-count station metadata (TMG 2001 pipe-delimited): id, direction, lane, F_System, lat/lon, county | `data/tmas/TN_2025_TMAS.STA` | 2026-09-18 | https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2025/2025_station_data.zip → `Station/TN_2025 (TMAS).STA`; sha256 088ea6972b172b70… |
| FHWA TMAS March 2025 volumes | Hourly volumes per station × direction × lane × day for Tennessee | `data/tmas/TN_Mar_2025_TMAS.VOL` | 2026-09-18 | https://www.fhwa.dot.gov/policyinformation/tables/tmasdata/2025/mar_2025_ccs_data.zip → `TN_Mar_2025 (TMAS).VOL`; sha256 6acf38fed9bf6637… |
| INRIX Hamilton County | 1-minute speeds per XD segment, 2025-03-10 and 2025-03-17 (6,805 segments) | `data/inrix/Hamilton-County-INRIX.csv`, `data/inrix/XD_Identification.csv` | pre-existing | proprietary, project licence |
| ACS 2021 5-year (Census API) | Held-out tables B08602, B08603, B08604 (workplace geography) and B08133 (residence). Block-group and tract rows come back null for B08602/B08603; usable at county (and place) level only. B08133 is populated for 9 of 87 Hamilton tracts | `<run_dir>/census_data/held_out/*.parquet` (cached on first fetch) | on demand | https://api.census.gov/data/2021/acs/acs5 |

Hamilton County TMAS coverage on 2025-03-17: 11 stations (10 reporting), on SR-153, SR-111, I-24 and
SR-21; 26 station-direction rows that day. The pooled hourly profile peaks at 7–8 AM (15,125) and
4–5 PM (19,186).

## Not yet acquired

- ~~CTPP portal download~~ — **superseded**: CTPP is fetched through its data API by
  `analysis/ctpp.py` using `CTPP_API_KEY` from `.env`. `POST https://ctppdata.transportation.org/api/data/2021`
  with `{"geo": "C1100US<state><county>", "get": "<table>_e1,...", "d-geo": "C3100US"}`. There is no
  metadata endpoint; the table inventory and column counts in `docs/EXPERIMENTS.md` were probed.
  Cached to `data/ctpp/<table>_<county>_tract_<year>.parquet`.
- MPO HBW trip tables, TDOT hourly counts, Advan Neighborhood Patterns: request drafts in
  `data_requests.md`.
- **Google Routes API**: `GOOGLE_MAPS_API_KEY` is in `.env` and the key is valid, but the Cloud
  project has no billing account, so every call returns 403 `BILLING_DISABLED`. Google requires a
  billing account even inside the free monthly caps (10,000 route-matrix elements, 5,000
  traffic-aware). Blocked pending that.
- **Replica origin-destination export** (`data/replica/replica-10_08_24-origin-destination.csv`,
  61 MB, git-ignored): 601,872 zone pairs over 7,719 custom zones covering the Chattanooga region,
  with trip counts split by purpose including `work_count`. Centroids only, no polygons, so zones
  are assigned to the block group containing their centroid. No time-of-day dimension. Wired up in
  `analysis/replica.py`. Replica is itself a model, so this is agreement between models and
  measurements, not validation against truth.
- `data/replica/ReplicaTripSummaries.xlsx`: region-wide marginals, no OD structure, and no region,
  season or year label anywhere in the file. Superseded by the export above; unused.
- `data/replica/Hamilton County_Replica_ForCARTA.mpkx`: 91 MB ArcGIS mobile map package. Unused;
  it would supply the zone polygons if centroid assignment ever needs replacing.
