# MoveOD — outstanding issues

Status legend: `[x]` done · `[~]` closed, deliberately not changed · `[ ]` open · **P0/P1/P2** priority

---

## Done (committed `0399db0` `b67df44` `905126f` `e5d3646` `3eb95da` `a6d6601` — not pushed)

- [x] **Census API key leaked to logs.** `get_census_data` printed `response.url`
      including `key=<API key>`. Now redacted. (`generate/utils.py:478`)
- [x] **Departure times floored to the hour and carried downstream.**
      `.dt.floor(TIME_INTERVAL)` in the pre-calibration branch meant census bins
      2/4/6/8/11 were unreachable, 45% of candidates fell back to median-imputed
      travel times, and `arrival_time` ran up to 59 min early. Travel times
      verified unchanged by the fix. (`generate/generate_routing_df.py:141`)

---

## A. Correctness vs. the paper  — all done

- [x] **P0 — Mean road speed shift used the wrong formula.** (`generate/utils.py`,
      `calculate_speed_shift` + `apply_mssr_to_existing_graphs`)
      Paper: `psi = tau_init / tau_ACS`, applied as `v' = psi * v`.
      Code was: `mssr = (tau_ACS - tau_init)/tau_ACS`, applied as `1/(1+mssr)` —
      the two agree only when `tau_init == tau_ACS`.
      **Fixed:** `calculate_speed_shift` now returns `mean_simulated /
      mean_census`; `apply_mssr_to_existing_graphs` applies it directly. The
      dead twin `create_hourly_graphs_with_speed_shift` was updated in step so
      the two formulas cannot drift apart.
      **Verified (Adams):** psi 0.8457 -> 0.8176; post-MSSR mean 20.961 -> 21.682,
      exactly `tau_ACS`; gap 0.722 min -> 0.000.

- [x] **P0 — ILP travel-time constraint used a 0/1 indicator instead of the
      fraction `pi_{o,d,s,k}`.** (`generate/calibrate_ilp.py`)
      Paper constraint (4) weights each cell by the fraction of its commuters
      whose travel time falls in bin k, so `sum_k pi = 1` per cell and the
      constraints are mutually consistent. The code added each cell at full
      weight to *every* bin it touched: on Adams, 41 of 66 cells span >1
      travel-time bin (up to 5), making `sum_b LHS_b` equal 1.76x-3.42x `N_o`
      against a right-hand side summing to `N_o`. The excess was absorbed by the
      `eta` slacks the objective minimises.
      **Fixed:** coefficient is now (rows in cell with `time_bin == b`) / (rows in
      cell), built from two groupbys (also cheaper than the old per-cell mask).
      **Verified:** unit check confirms `sum_b pi[cell][b] == 1` and
      `sum_b LHS_b == N_o`. End-to-end on Adams travel-time TVD 29.79 -> 27.98 pts
      (right tail: 45-59 min error 7.31 -> 3.23, 60-89 min 3.05 -> 0.39), with
      departure TVD still 0.00 and trip count still 1544.
      *Caveat:* Adams has only 4 CBGs, so most of the residual 28 pts is
      geography the ILP cannot reach (`5-9 min` bin is 22% of ACS but barely
      exists in the county). Expect a larger effect on Hamilton — re-measure
      there.

- [x] **P2 — Integer rounding residual dumped into the last bin.** Biased the
      last bin/destination and could drive a target negative → infeasible ILP →
      silent IPF fallback. **Fixed:** `_apportion()` uses largest remainder, so
      every target stays within 1 of its exact share and none can go negative.
      Unit-checked on skewed/degenerate weightings; Adams still 4/4 ILP, 0
      fallbacks, departure TVD 0.00.

- [x] **P2 — Objective coefficients named opposite to the paper.** The code
      pinned the eta term at 1 and applied its `alpha` to the zeta term, so its
      alpha was the paper's beta and the paper's alpha was inexpressible.
      **Fixed:** both named as in the paper and threaded through
      `calibrate_with_ilp`, defaulting to 1.0, so the paper's [0,1]x[0,1] sweep
      is now reproducible. **Verified:** at alpha=beta=1 the output is
      byte-identical to the previous objective.

---

## B. Silent failures  — all done

- [x] **P0 — Resumed runs routed nothing.** `serialize_graphs` stringified the
      Timestamp keys; `deserialize_graphs` never converted them back, so every
      `hourly_graphs.get(ts)` missed. Observed: `Done: 0/1544 succeeded`, exit
      code 0, CSV still written.
      **Fixed** in `cli.py` and `app.py` (the latter edited in binary to preserve
      its CRLF line endings).
      **Verified (Adams):** a resumed run goes `0/1544` -> `1544/1544`.

- [x] **P1 — Figures silently lost both ACS reference curves.** The pipeline
      writes the census tables as `.parquet`; the figure script read `.csv`, so
      Figure 1 lost its Census comparison and Figure 2 lost ACS, while still
      exiting 0.
      **Fixed:** census tables are located by stem, preferring `.parquet` and
      falling back to `.csv`. **Verified (Adams):** both figures now draw every
      series; Fig 1's Generated and Census curves coincide, matching TVD 0.00.

- [x] **P1 — Figure 2's "Initial" series was never drawn.** `find_initial_csv`
      globbed `lodes_combs/lodes_<day>.csv`, written only by
      `lodes_combs_county.py`, which nothing imports; the directory stays empty
      and a missing frame warned about nothing.
      **Fixed:** the Initial series now reads
      `intermediate/<day>/post_mssr_routing_df.parquet` — the pre-calibration
      assignment the ILP actually calibrates — falling back to the pre-speed-shift
      routing frame, then the legacy CSV, and warning when none is found.
      *Interpretive choice:* "Initial" = post-speed-shift, pre-calibration. The
      paper is ambiguous; this isolates the ILP's contribution.

- [x] **P2 — Misleading log line hid the above.** `lodes_combs.py` logged
      `"Saved results for day {day}"` after an in-memory append. **Fixed** to
      `"Processed day {day}"`.

---

## C. Performance — cores and RAM  — all done

- [x] **P1 — Routing never ran in parallel.** `get_routed(..., parallel=False)`
      was the default and all six call sites omitted the argument, while the log
      claimed "chunks for 31 workers". Measured at 154,400 OD pairs: **79.65s
      sequential vs 11.35s parallel**.
      **Fixed:** `parallel=True` by default. The pool now uses ordered `imap`
      rather than `imap_unordered`, because `routing_df` row order feeds
      downstream sampling — unordered made results depend on worker scheduling
      (travel-time TVD moved 27.98 -> 27.07). With ordered `imap` the parallel
      output **reproduces the sequential result exactly** (27.98).

- [x] **P1 — A whole routing pass was discarded.** The post-calibration
      `get_routed` in `cli.py` computed a frame and `del`'d it unread — roughly a
      third of all routing work. **Removed.**

- [x] **P1 — 48 hourly graphs held only 2 distinct contents** (peak / off-peak).
      **Fixed:** `create_graphs_from_osm_speeds` builds each once and shares the
      object; `apply_mssr_to_existing_graphs` memoises on source identity so the
      MSSR pass doesn't re-expand them; `serialize_graphs` stores each distinct
      graph once under a slot id plus a key->slot map (`shared-v1`), and
      `deserialize_graphs` reads both that and the legacy layout.
      **Verified (Adams):** `hourly_graphs.json` **58 MB -> 2.5 MB**; reloads as
      48 keys backed by 2 distinct objects; legacy files still load.

- [x] **P1 — No RAM guard on routing.** `mp.Pool(..., initargs=(hourly_graphs,))`
      pickles a full copy of the graph dict into every worker (not
      copy-on-write). **Fixed:** `_safe_routing_workers` sizes one distinct
      graph, scales by the distinct-graph count and a measured
      pickled->resident factor, and throttles against available RAM.

- [x] **P1 — The second routing pass was redundant.** A uniform speed scaling
      leaves shortest paths unchanged, so the post-MSSR pass re-derived identical
      routes and divided every travel time by `psi` (verified: identical
      distances, travel-time ratio constant to 2e-16).
      **Fixed:** `mean_speed_shift_is_uniform()` checks whether any edge kept a
      measured INRIX speed; when none did — the documented default —
      `rescale_routing_df()` derives the frame arithmetically. The INRIX path
      still re-routes. **Verified equivalent** on Adams to 4.3e-14, with every
      bin and distribution unchanged. A default run now routes the county
      **once instead of three times**.

- [x] **P1 — Routing chunks were sized one per worker.** Hourly buckets are very
      uneven, so cores idled waiting on the largest. Now ~4 chunks per worker
      (Hamilton: 46 -> ~150 chunks).

- [~] **P2 — Dask threaded scheduler — measured, NOT worth changing.** The whole
      `lodes_combs` stage is **28 s of a 287 s Hamilton run (9.8%)**, and that
      includes the census API calls, so the dask compute is a fraction of it.
      Switching to `scheduler="processes"` would pickle `origin_buildings` and
      `dest_buildings` (tens of MB of GeoDataFrames) into every task. Upside
      bounded under 10%, downside is an OOM on large counties. Left alone
      deliberately.

- [~] **P2 — OSM fetch sequential under Streamlit — NOT changing.** The guard is
      correct: forking breaks Streamlit's script runner. The work is Overpass
      API-bound rather than CPU-bound, Overpass rate-limits, and the result is
      cached in `county_all_buildings.geojson` after the first run. Left alone
      deliberately.

---

## E. Found while fixing the above

- [x] **P0 — The Streamlit path crashed after calibration.** `app.py` called
      `get_routed(..., hourly_graphs=...)` but the parameter is
      `hourly_graphs_arg`, so the final routing pass raised `TypeError` and the
      app never wrote its output. Pre-existing on `origin/main`. It also wrote
      that routed frame rather than `calibrated_df`, so the app and CLI were
      specified to emit *different* files.
      **Fixed:** removed the pass (the same redundant one already dropped from
      `cli.py`) and write `calibrated_df`. Streamlit and CLI outputs now match.
      **Verified:** all four `get_routed` call sites checked against the
      function signature by AST walk.

- [x] **P1 — Calibrated trips were stamped with a hardcoded date.**
      `post_calibrating_assignment` built `departure_datetime` from a literal
      `pd.to_datetime("2025-03-10")`, so every run carried that date regardless
      of the requested range and every day of a multi-day run collapsed onto it.
      A Hamilton run for 2025-03-11 produced rows dated 2025-03-10.
      **Fixed:** `desired_date` threaded through `calibrate_with_ilp` from both
      call sites. Verified on Adams for 2025-06-17.

---

## D. Minor / latent  — all done

- [x] **P2 — Fabricated distances.** `travel_distance_mi = travel_time_min / 2`
      invented a distance at an implied 30 mph. **Fixed:** imputes the observed
      median distance for the same O-D pair. Adams implied speeds now span
      14-34 mph (median 23) instead of exactly 30 on every imputed row.
- [x] **P2 — Fragile column slicing.** `dep_pos` used positional
      `columns[2:-4:2]`. **Fixed:** selects by `_estimate` suffix like the rest
      of the file. Verified both forms pick the same 14 columns in order.
- [x] **P2 — `calibrated_weight` was vestigial in the output.** After expansion
      one row is one trip and the column just repeated the parent cell's count;
      summing it gives sum(w^2), which is what produced a bogus "33.8x trip
      inflation" reading earlier in this branch. **Dropped from the output.**
      Note: output schema change. Trip count is `len(df)`.
- [x] **P2 — RNGs not seeded deterministically.** `random` was never seeded here
      and `np.random.seed(42)` ran *after* the departure sampling; workers also
      drew from inherited state, so results depended on how origins were spread
      across the pool. **Fixed:** both seeded up front, plus a per-origin seed
      from `zlib.crc32` (stable across processes, unlike salted `hash()`).
      Verified: two consecutive Adams runs byte-identical.
- [x] **P2 — Dead code deleted.** `process_od_pair_with_geoid` +
      `generate_travel_times_single_process` (the former set `departure_time_bin`
      to an hour index 0-23 against census bins 0-13 — a landmine if ever wired
      up) and `add_shortest_path_and_speeds_parallel` (`int(cpu_count()/8)`,
      raises on <8 cores). Confirmed no callers first.

---

## Verified as matching the paper

- ~150K synthetic trips for Hamilton County — actual **158,253**.
- Fig. 1, "block-level departure times are reproduced exactly" — verified, per-trip
  departure distribution vs ACS B08302 has **TVD = 0.00**.
- Trip counts equal the LODES-adjusted totals exactly (Adams 1,544; Hamilton
  158,253).
