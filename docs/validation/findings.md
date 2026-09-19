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
| Tract-to-tract flows | CTPP 2017-2021 Part 3 (survey, independent of LODES) | tract pairs | flows inherited from LODES; CPC 0.67 for both |
| Departure and travel time **per flow** | CTPP b302104 / b302106 | 786 intra-county flows | fails at ~10x the sampling-noise floor |
| Flows vs a second independent source | Replica activity model, Oct 2024 | tract pairs | MoveOD sits inside the source-disagreement envelope |

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

8. **Spatial flows are LODES's, and CTPP scores both at 0.67.** Against CTPP's independent
   survey-based tract-to-tract flows, the common part of commuters is 0.666 for the synthetic
   trips and 0.666 for raw LODES, identical to four decimals on every basis tried. Coverage is
   excellent: only 1.3 % of CTPP workers sit on flows MoveOD lacks, and the structural ceiling
   imposed by CTPP's own suppression is 0.987. The shortfall is flow magnitude, not missing
   pairs. Say plainly that this checks the LODES input, not the synthesis.
9. **Per-flow timing fails, and the county aggregate hides it.** Worker-weighted per-flow TVD
   against CTPP is 0.624 for departure time and 0.554 for travel time, against Monte Carlo
   sampling-noise floors of 0.068 and 0.055, so nine to ten times the floor. The pooled
   county-level TVDs for the same quantities are 0.032 and 0.161. The opt-in `--ctpp-anchor`
   calibration cuts the departure figure to 0.233 and lifts the independent traffic-sensor profile
   correlation from 0.315 to 0.446, but does not transfer to flows CTPP does not publish. Report the ratio to the
   floor, never the raw TVD alone. CTPP travel times are self-reported and heaped on multiples of
   five, a confound the departure result does not carry.

10. **The flow score has a benchmark now, and it is favourable.** Four commute matrices exist for
   Hamilton: MoveOD, LODES, CTPP and Replica. On the flows all three non-LODES sources publish,
   MoveOD agrees with CTPP at Sørensen 0.671 and with Replica at 0.654, while **CTPP and Replica
   agree with each other at only 0.605**. The same ordering holds on every support and on rank
   correlation. So a score near 0.65, which looks weak in isolation, is inside the envelope of
   disagreement between two established measurements of the same county. State the benchmark
   whenever the flow score is quoted; the score alone is not interpretable.
   Caveat that must travel with it: MoveOD's tract flows are identical to LODES's, so this shows
   the **input** sits inside that envelope, not that the synthesis is good. Replica's work trips
   include those not starting at home, CTPP suppresses flows under three observations, and the
   three vintages differ (CTPP 2017-2021, Replica October 2024, MoveOD March 2025).

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
- **No per-destination timing structure.** The integer program constrains each origin's
  departure marginal and has nothing that distinguishes one destination from another, so it
  spreads a single origin profile across every destination that origin serves. Synthetic flows
  occupy a median 4.63 effective departure bins against CTPP's 2.81, and 2.73 against 2.29 for
  travel time: the synthetic profiles are too smooth and too alike, while real workplaces have
  lumpy shift structure. A workplace-side constraint is the obvious extension, though ACS
  publishes B08602 only at county level, so CTPP Part 2 or Advan would have to supply it.
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

1. Regenerate every county under `move_OD/` with current code, **including Hamilton**, whose
   archived run was found on 2026-09-19 to predate the anchor fix; 24 of 48 run-days lack an
   uncalibrated frame and several are pre-fix outputs (departures all in bin 0; Davidson with
   90.8 % of travel times imputed at 11.02 min; Madison with 41,201 of 168,927 workers).
2. Acquire CTPP 2017-2021 tract-to-tract flows for Tennessee (portal steps in
   `data_sources.md`). It is the only public external test with spatial content, and the one a
   reviewer will ask for. Compare LODES vs CTPP and synthetic vs CTPP separately so input error and
   synthesis error are not conflated.
3. Replica supplies no time-of-day dimension in its origin-destination export, so it cannot test
   the per-flow timing failure. When Advan Neighborhood Patterns arrive, use `work_behavior_device_home_areas` for a
   block-group home → work matrix and `stops_by_each_hour` for per-workplace-BG arrival profiles —
   the block-group resolution that ACS does not provide.
4. Plumb a `--seed` through `generate/calibrate_ilp.py` (seeds are hard-coded: 123, 42 and a
   per-origin CRC) and report mean ± sd over 5 seeds.
5. Add the ACS sampling floor (0.016) and the B08303–B08603 floor (0.086) as bands on the TVD
   figures.
