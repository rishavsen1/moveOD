# Using CTPP to fix the per-flow timing failure

A one-line change to the calibration that removes the paper's conditional-independence
assumption. Estimated to cut per-flow departure error by about 3.4x on Hamilton County.

## The paper today

For each origin census unit `o`, the paper solves (Section V-A):

```
min  α Σ_k (η_k⁺ + η_k⁻)  +  β Σ_(d,s) (ζ⁺_{o,d,s} + ζ⁻_{o,d,s})

(1)  Σ_(d,s) a_{o,d,s} = N_o
(2)  Σ_s     a_{o,d,s} = N_o · p_{o,d}                                    ∀d
(3)  Σ_d     a_{o,d,s} = N_o · p_{o,s}                                    ∀s
(4)  Σ_(d,s) π_{o,d,s,k} a_{o,d,s} + η_k⁻ − η_k⁺ = N_o · p_{o,k}          ∀k
(5)           a_{o,d,s} + ζ⁻_{o,d,s} − ζ⁺_{o,d,s} = m_{o,d,s}             ∀(d,s)
```

Constraints (2) and (3) pin the two **marginals**: how many commuters go to each destination, and
how many leave in each time block. Neither says anything about their **combination**. That is set
by the anchor `m` in (5), which Section IV builds under the paper's stated simplification
`P(S | D,O) = P(S | O)`. In expectation the anchor is therefore a product of marginals:

```
m_{o,d,s} = N_o · p_{o,d} · p_{o,s}
```

Every destination an origin serves receives a copy of the same departure profile. Measured against
CTPP, per-flow departure error is 0.688 against a sampling-noise floor of 0.068 — ten times the
floor — while the county aggregate is correct at 0.026.

## The change

CTPP table **B302104** publishes, for each home-tract to work-tract pair, the number of workers
leaving in each of the 14 ACS departure blocks. Let `τ(·)` map a block group to its tract and let
`c_{T,T',s}` be that count. Define the observed conditional profile

```
q_{T,T',s} = c_{T,T',s} / Σ_s' c_{T,T',s'}          when Σ_s' c_{T,T',s'} > 0
```

and replace the anchor in (5), leaving everything else untouched:

```
m'_{o,d,s} = N_o · p_{o,d} · q_{τ(o),τ(d),s}        if the tract pair is published
m'_{o,d,s} = N_o · p_{o,d} · p_{o,s}                otherwise   ← the paper's anchor, as back-off
```

In probability terms, the chain rule in Section IV goes from

```
P(O,D,S,E) = P(E | S,D,O) · P(S | O)   · P(D | O) · P(O)     ← assumption
```

to

```
P(O,D,S,E) = P(E | S,D,O) · P(S | D,O) · P(D | O) · P(O)     ← estimated from CTPP
```

## Why it is safe

`m'` generally violates (3), because `Σ_d m'_{o,d,s} ≠ N_o · p_{o,s}`. That is fine and is exactly
what the `ζ` slacks exist for. Constraints (1)–(4) stay hard, so the ACS origin marginals are still
matched exactly; the solver returns the feasible point closest to the CTPP joint. ACS governs the
marginals, CTPP shapes the interior.

Cost to the solver: none. Same variables, same constraints, one small program per origin. Only the
right-hand side of (5) changes. The CTPP fetch already exists in `analysis/ctpp.py`.

## Expected gain, and its limits

| | Hamilton County |
|---|---|
| Synthetic trips on tract pairs CTPP publishes | 78.7 % |
| Per-flow departure error today | 0.688 |
| Sampling-noise floor | 0.068 |
| Estimate if covered flows reach the floor | **≈ 0.20** (about 3.4x better) |

That estimate is an upper bound on the gain, for three reasons:

1. Constraints (2)–(4) pull the solution away from the anchor.
2. CTPP values are rounded to 5, so on a median 80-worker flow a bin share is quantised to about
   6 %, and they are additionally perturbed for disclosure control.
3. CTPP is tract-to-tract; applying `q` to every block-group pair inside a tract pair assumes the
   profile is uniform within it. Much weaker than assuming independence from destination, but it
   should be stated.

## Circularity: what must then be held out

Once B302104 is a constraint it can no longer validate. Keep at least two of:

- **CTPP B302106** (travel time by flow) — different quantity, same survey.
- **Replica** — independent commercial model, already wired up in `analysis/replica.py`.
- **TMAS hourly station profiles** — independent observed traffic.
- **Split-half** — fit on a random half of the published flows, validate on the other half.
  Cleanest, and recommended alongside Replica.

## The more principled alternative, for future work

Add a destination-side marginal mirroring (3), constraining arrival time at each workplace:

```
(3')  Σ_o a_{o,d,s} = N_d · p^work_{d,s}              ∀d, s
```

This is the right fix in spirit, since it constrains the workplace side directly rather than
importing a joint. But it couples all origins, so the per-origin decomposition that keeps the ILP
tractable is lost: Hamilton would become one program with roughly 269 × 270 × 14 ≈ 10⁶ variables
instead of 269 small ones. ACS publishes arrival time by workplace (B08602) only at county level,
so the data would have to come from CTPP Part 2 or Advan regardless. Recommend the anchor change
first.
