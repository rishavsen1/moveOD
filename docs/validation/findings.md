# External validation — findings and how to state them

Status as of 2026-09-18, branch `feat/external-validation`. Every number below has a row in
`docs/EXPERIMENTS.md`; numbers were produced by `analysis/validate_external.py`,
`analysis/link_loads.py` and `analysis/station_counts.py` on the Hamilton County, TN run of
2025-03-17 (158,253 calibrated trips) unless stated otherwise.

## What the pipeline is validated against

| Test | Data the pipeline never reads | Resolution actually available | Verdict |
|---|---|---|---|
| Arrival time by workplace | ACS B08602 | county only (block-group and tract rows are null) | cannot separate methods (sampling floor) |
| Travel time by workplace | ACS B08603 | county only | near-tautological; measures the in-commuter universe gap |
| Mean travel time per departure block | ACS B08133 / B08302 | 9 of 87 tracts + county | pooled bias only; cell-level R² uninformative |
| Link loads vs observed speed drop | INRIX, 2025-03-10 and 03-17 | 6,105 XD segments | null; under-powered (3 % mean speed drop at 07:00) |
| Link loads vs hourly station counts | FHWA TMAS, 10–11 station-directions | per site, per hour | informative: correct AM shape, site-level loading errors |

## Statements that hold

1. **Constraint verification, not validation.** Matching B08302 (TVD 0.00) and B08303 is
   verification that the ILP satisfied its constraints. Say so in the paper and move the current
   Figures 1–2 under that heading.
2. **Commute-only signature at count stations.** Synthetic volume divided by observed volume,
   pooled over the matched stations, is 0.33 at 05:00, 0.68 at 06:00, 0.66 at 07:00, 0.58 at 08:00,
   0.29 at 09:00, 0.12 at 10:00 and 0.02–0.04 from late morning through the evening peak. The
   morning rise is reproduced and there are no return trips, as expected from a
   home-to-work-only dataset. Log-volume Pearson r over 60 station-hours is 0.745 (0.772 with I-24
   included at a 60° bearing tolerance).
3. **Per-site spread.** Site ratios run from 0.00 to 1.23 (median 0.22); dropping any one of the
   ten sites moves the pooled ratio within [0.32, 0.52]. Between-site correlation is 0.75 (bigger
   roads carry more trips); within-site hourly correlation is 0.48. Quote the pooled ratio only
   with this spread.
4. **Calibration vs no calibration, held-out tables.** On travel time by workplace, calibrated
   TVD is lower than uncalibrated in 17 of 18 usable run-days (7 counties; 6 are near-replicate
   Hamilton days); on arrival time 16 of 18, but the arrival margins (0.076 vs 0.086 for Hamilton)
   are below the ACS sampling floor of about 0.016 and the IPF baseline ties (0.074). Two further
   run-days are excluded as known pre-fix outputs (all departures in bin 0).
5. **Travel-time held-out test floor.** The pipeline's input table B08303 already sits at TVD 0.086
   from the workplace table B08603, so any method reproducing the input scores about 0.086; the
   calibrated output scores 0.093. Wasserstein-1 moves the other way (3.80 → 4.22 min).
6. **INRIX.** Spearman ρ between synthetic 07:00 load and observed speed drop is +0.01 on the
   2025-03-10 speeds and −0.04 on 2025-03-17; motorways alone +0.21 (n 118). County-wide mean
   speed drop at 07:00 is 3 %, so the test has no power here. Report as null and under-powered.
7. **Joint departure × travel time.** Synthetic means exceed ACS aggregate minutes by 3.5 min in
   the 9 published tracts and by 1.4 min county-wide. The county-wide part is within-bin placement
   (synthetic within-bin means sit at bin centres; the ACS aggregate implies lower-than-centre
   means), which any histogram matching produces. The remainder is concentrated in two tracts.
   ACS includes out-of-county and non-auto commuters, which lengthens the ACS side, so the
   confound is conservative.

## Modelling limitations the station comparison exposed

- **Through traffic is outside the model.** Every calibrated trip has origin and destination in
  county 47065. SR-111 at the county's northern tip (stations 000356/000156) carries 2,565
  observed AM vehicles and 1 synthetic; the corridor is through traffic to Sequatchie and Bledsoe.
  A multi-county run (`od_option` "Only Destination in County") would test this.
- **All-or-nothing assignment without capacity.** At the SR-153 cross-section (000311/000111)
  the synthetic AM split is 71 % southbound against 54 % observed, and the southbound synthetic
  count exceeds the full observed count (ratio 1.23), which a commute subset cannot do. Shortest
  path concentrates the Hixson → Chattanooga commute on one carriageway.
- **Single-building census units.** Face validity on Hamilton finds one home coordinate carrying
  all 605 commuters of block group 470650101032 and two work coordinates carrying all 5,237 jobs of
  470650016003. `origin_metadata.json` shows those units had one (or two) candidate locations: the
  tiered selection keeps the tagged-OSM set however small instead of falling back to Microsoft
  footprints. This is the concentration failure the paper's introduction argues against; a
  minimum-candidate threshold before falling back would remove it.
- **TMAS data contradiction at 000540.** The station file signs N/S, the volume file E/W; the E/W
  cone reaches only residential cross streets (ratios 0.008 and 0.002). Drop both rows from any
  headline figure: pooled AM ratio without them is 0.499 (n 8).

## Baselines: what is and is not fair

- IPF fitted to LODES (o,d) and B08302 (o,s), seeded from the uncalibrated frame, is a fair
  baseline for the **arrival** test only; it never sees a travel-time marginal, so its travel-time
  distribution is the uncalibrated one by construction. To claim the ILP beats IPF on travel time,
  add an (origin × travel-time-bin) B08303 marginal to the IPF.
- The uniform-departure baseline is a lower bound only.

## Before any of this goes into the paper

1. Regenerate every county under `move_OD/` with current code; 24 of 48 run-days lack an
   uncalibrated frame and several are pre-fix outputs (departures all in bin 0; Davidson with
   90.8 % of travel times imputed at 11.02 min; Madison with 41,201 of 168,927 workers).
2. Acquire CTPP 2017-2021 tract-to-tract flows for Tennessee (portal steps in
   `data_sources.md`). It is the only public external test with spatial content, and the one a
   reviewer will ask for. Compare LODES vs CTPP and synthetic vs CTPP separately so input error and
   synthesis error are not conflated.
3. When Advan Neighborhood Patterns arrive, use `work_behavior_device_home_areas` for a
   block-group home → work matrix and `stops_by_each_hour` for per-workplace-BG arrival profiles —
   the block-group resolution that ACS does not provide.
4. Plumb a `--seed` through `generate/calibrate_ilp.py` (seeds are hard-coded: 123, 42 and a
   per-origin CRC) and report mean ± sd over 5 seeds.
5. Add the ACS sampling floor (0.016) and the B08303–B08603 floor (0.086) as bands on the TVD
   figures.
