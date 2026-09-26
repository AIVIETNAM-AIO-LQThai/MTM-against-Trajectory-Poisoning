# E3 Protocol — CSDPC Candidate-Objective Attainment Audit

## Status

`POSTFINAL DIAGNOSTIC ONLY`

E3 is a new post-final source-fidelity diagnostic. It does not modify the frozen `walker2d-final-v1` evidence base. The frozen Group-2 conclusion remains `SOURCE-FIDELITY DISCREPANCY UNRESOLVED`.

## Scientific question

The CSDPC paper states that each rare raw sequence receives multiple stealth-constrained candidate poisoned sequences; candidates are clustered into decision patterns; and the candidate whose resulting pattern has the highest clean-data occurrence count is selected. The public description does not specify the proposal distribution or candidate count `n`.

The independent reproduction uses `n = 100` candidates sampled independently and uniformly inside the relative L-infinity stealth box.

E3 asks:

> How well does that existing canonical n=100 proposal pool attain the paper's stated pattern-frequency objective?

E3 does **not** introduce an alternative candidate generator.

## Frozen context

Use the frozen Group-2 source-level runtime: `WSL + g2-csdpc`.

Dataset: `walker2d-medium-v2`

SHA256:

```text
cf00f43add04c17fdfc2958dd581dea0851b2e5bedbe6fda073758a8f841aeda
```

Settings:

```text
k = 8
L = 5
eta = 0.05
rho = 0.01
attack seeds = 0, 1, 2
n candidates = 100
action range = [-1, 1]
```

Rare windows use the canonical deterministic rare-pattern-first, non-overlapping selection. The same seeded NumPy RNG is consumed sequentially across windows, matching the independent reproduction.

## Per-window metrics

Let `F_source` be the clean occurrence count of the selected rare source pattern, `F_best` the highest occurrence count found among its canonical 100 candidates, and `F_global_max` the maximum clean pattern frequency for the seed.

Report:

```text
chosen_pattern_changed
frequency_improved
best_to_source_ratio = F_best / max(F_source, 1)
best_to_global_max_ratio = F_best / F_global_max
normalized_frequency_progress = (F_best - F_source) / (F_global_max - F_source)
```

`F_global_max` is an objective reference only; it is not assumed geometrically reachable from every source sequence.

Also report candidate-pool diversity and whether the pool contains any pattern change, any frequency improvement, or a global-max-frequency pattern.

## Aggregate metrics

Aggregate separately for seeds 0, 1, 2 and overall:

```text
number of selected windows
fraction chosen patterns changed
fraction chosen frequencies improved
mean/median source frequency
mean/median best target frequency
mean/median best/source ratio
mean/median best/global-max ratio
mean/median normalized frequency progress
mean candidate-pattern diversity
fraction pools with any pattern change
fraction pools with any frequency improvement
fraction pools containing a global-max-frequency pattern
```

There is no numerical PASS/FAIL threshold.

## Interpretation boundary

If the canonical pools frequently fail to change rare patterns or make little progress toward high-occurrence patterns, candidate generation remains a plausible source-fidelity bottleneck.

If they routinely find substantially more frequent patterns, the proposal distribution is less compelling as the primary explanation, although the true source generator remains unknown.

E3 must not be used to select a new generator based on CQL/DT performance, increase `n` until performance collapses, or claim source attack reproduction.

## Output

```text
experiments/postfinal_controls/e3_csdpc_candidate_objective_attainment.json
```

No poisoned HDF5 artifact is written. No learner is trained or evaluated.
