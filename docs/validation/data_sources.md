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

- **CTPP 2017-2021 Part 3** tract-to-tract flows (A302100 workers; A302103 by means of transportation)
  for Tennessee: the portal at https://ctppdata.transportation.org is interactive. Steps: Part 3 →
  select table → Geography → *Bulk Selection* tab → Tract-to-Tract, State = Tennessee → Retrieve →
  Download CSV. Save to `data/ctpp/`.
- MPO HBW trip tables, TDOT hourly counts, Advan Neighborhood Patterns: request drafts in
  `data_requests.md`.
