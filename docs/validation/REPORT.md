# Does MoveOD produce realistic commute data?

**Short answer:** yes for aggregate and spatial structure, no for the timing of individual
origin-destination pairs. Six independent tests, all reproducible, all numbers in
`docs/EXPERIMENTS.md`. Demonstrated on Hamilton County, TN (158,253 synthetic commute trips,
2025-03-17).

> **Numbers updated 2026-09-19.** The run these figures were first measured on turned out to
> predate a calibration fix. Every conclusion survives; some figures moved. See the correction at
> the end of `docs/EXPERIMENTS.md`. Regenerate all county outputs before publishing.

---

## Why the old claim was not enough

The paper compares the generated trips against two Census tables, B08302 (departure time) and
B08303 (travel time). Both are **inputs**: B08302 is a hard constraint in the calibration and
B08303 is what the calibration optimises against. Matching them proves the optimiser worked. It
does not prove the data resemble reality. Everything below uses data the pipeline never reads.

---

## The six tests

### 1. Traffic counts at road sensors — *informative, passes*

Ten continuous count stations in Hamilton County record hourly traffic. We route the synthetic
trips and count how many cross each sensor.

| Hour | Synthetic ÷ observed |
|---|---|
| 6–7 am | 0.72 |
| 7–8 am | 0.70 |
| 8–9 am | 0.58 |
| 11 am – 6 pm | 0.02–0.04 |

This is exactly the fingerprint a home-to-work-only dataset should leave: it explains most of the
morning rise and almost nothing else, because it contains no return trips. Correlation of hourly
volumes (log scale) is 0.75.

*Caveat:* per-site ratios range from 0.00 to 1.23. One freeway station gets almost no synthetic
traffic, and one gets more than the real total, which a commute-only subset cannot legitimately do.

### 2. Flows vs two independent sources — *passes, once benchmarked*

Two other origin-destination matrices exist for the county: **CTPP** (US Census survey) and
**Replica** (a commercial model calibrated to mobile-phone data). How similar are the flows?

| Pair | Similarity (Sørensen) |
|---|---|
| MoveOD vs CTPP | 0.671 |
| MoveOD vs Replica | 0.654 |
| **CTPP vs Replica** | **0.605** |

0.65 looks weak alone. But the two real-world sources agree with each other *less* than either
agrees with MoveOD. So MoveOD's flow structure is inside the range over which established
measurements of the same county disagree. **Always quote the benchmark with the score.**

*Caveat:* MoveOD's flows come straight from LODES and are identical to it. This validates the
**input**, not the synthesis.

### 3. Timing of individual flows — *fails, and this is the real finding*

CTPP publishes, for each home-tract to work-tract pair, the distribution of departure times and
travel times. Nothing in MoveOD is fitted to this.

| | MoveOD's error | Error a *perfect* model would still make | Ratio |
|---|---|---|---|
| Departure time, per flow | 0.624 | 0.068 | **9×** |
| Travel time, per flow | 0.554 | 0.055 | **10×** |

The second column is a sampling-noise floor, measured by simulation: with ~58 trips spread over 14
time bins, even a flawless model scores above zero. MoveOD is ten times worse than that.

Meanwhile the **county-wide** distributions are fine (0.032 and 0.161). The model gets the whole
right and the parts wrong. Section "Why" below explains the cause.

**A fix exists and is implemented** (`--ctpp-anchor`, off by default). Feeding CTPP's observed
per-flow departure profiles into the calibration cuts the error from 0.624 to 0.233, and raises the
morning profile correlation at traffic sensors from 0.315 to 0.446 with total volume unchanged.
It does not generalise to flows CTPP does not publish, so it imports the timing rather than
learning it. See `ctpp_joint_proposal.md`.

### 4. Held-out Census tables — *inconclusive, by construction*

ACS publishes arrival time and travel time by *workplace*, which the pipeline never reads.
Calibration does improve agreement (0.086 → 0.076 on arrival, 0.160 → 0.093 on travel time), but:

- Census sampling noise alone is worth about 0.016, so the arrival gap is not significant.
- The pipeline's own input table already sits 0.086 away from the workplace table, so any method
  reproducing the input scores about that regardless.

These tables are only published at county level, which is too coarse to separate methods. Report
them for completeness, not as evidence.

### 5. Road speeds vs traffic data — *null, underpowered*

Do the roads MoveOD loads most show the biggest observed slowdown? Correlation is +0.01 against
one day of INRIX speeds and −0.04 against another. The county averages only a 3% speed drop at
7 am, so there is barely any congestion to detect. Report as null.

### 6. Internal plausibility — *mostly passes, one real defect*

Implied speeds run 18 to 35 mph (10th to 90th percentile), with none absurd. Trip lengths are
longer than an all-purpose trip mix, as commutes should be.

But one home coordinate carries all **605** commuters of its block group, and two work coordinates
carry all 5,237 jobs of theirs. Cause: those census units had only one or two tagged buildings, and
the selection keeps the tagged set however small instead of falling back to the fuller Microsoft
footprint data. This is the exact failure the paper's introduction argues against.

---

## Why per-flow timing fails

The calibration **does** preserve both marginal distributions exactly, per origin:

- the split of commuters across destinations, and
- the split of commuters across departure-time blocks.

What it does not constrain is the **combination** of the two. The paper states the simplification
openly: departure time depends only on the origin, `P(S | D,O) = P(S | O)`. In code, departure
times are drawn at random within each origin block group without reference to destination
(`assign_departure_times_by_cbg` in `generate/lodes_combs.py` groups by `h_geocode` alone).

So every destination served by one origin inherits the same departure profile. Reality is lumpier:
a hospital, a school and an office draw from the same neighbourhood at different hours. The data
show it — synthetic flows spread over 9.0 effective departure bins where CTPP uses 2.8.

CTPP is the first dataset able to test that stated assumption, and the assumption does not hold.
Fixing it needs a workplace-side timing constraint, which ACS does not publish below county level.

---

## Summary

| Test | Verdict |
|---|---|
| Hourly traffic counts | **Passes** — correct commute shape, 0.7 of observed morning peak |
| Per-flow timing after the CTPP fix | Improves 2.7x in sample, confirmed by traffic sensors; no transfer to unpublished flows |
| Flows vs CTPP and Replica | **Passes** — inside the range over which real sources disagree |
| Per-flow timing | **Fails** — 10× the noise floor, while the county total is right |
| Held-out Census tables | Inconclusive — county level is too coarse |
| Road speeds vs INRIX | Null — too little congestion to measure |
| Internal plausibility | Passes, except commuters piling onto single buildings |

**What to say about MoveOD:** it reproduces county-level commute structure well, its spatial flows
are as good as the disagreement between real data sources, and it is not yet reliable for questions
that depend on *when* a specific origin-destination pair travels.

**Three known limits, each now measured rather than assumed:**

1. Trips stay inside one county, so through traffic is missing entirely (one freeway sensor
   receives 1 synthetic vehicle against 2,565 observed).
2. Routing is shortest-path with no capacity limit, so one carriageway takes 71% of a corridor's
   morning flow where reality splits 54/46.
3. Departure time is independent of destination, which the per-flow test disproves.

**Reproduce any number:** `docs/EXPERIMENTS.md` lists every result with its exact command.
Scripts: `analysis/validate_external.py`, `analysis/ctpp.py`, `analysis/replica.py`,
`analysis/link_loads.py`, `analysis/station_counts.py`.
