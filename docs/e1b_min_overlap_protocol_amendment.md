# E1B Protocol Amendment v2 — Minimum-Overlap Run Relocation

## Status

`PREDECLARED AMENDMENT AFTER CONTROL-CONSTRUCTION FEASIBILITY FAILURES`

E1B is a post-final mechanistic control performed after the frozen
`walker2d-final-v1` study.

The original zero-overlap run-relocation control and the later
circular-shift construction both failed before completing the full
experiment. Their partial console output is not part of the evidence
base and must not be interpreted scientifically.

The scientific question remains unchanged:

> Does the Group 4E historical-state mechanism depend on the temporal
> locations selected by CSDPC?

No model is retrained.

---

## 1. Reason for This Amendment

The original E1B control required all relocated modified states to avoid
all original CSDPC-modified states.

That condition is not always feasible in trajectories with dense
corruption.

A later circular-shift control attempted to preserve the complete
trajectory-level perturbation pattern with one common offset. That
construction is also too restrictive because some trajectories have no
nonzero circular shift that preserves the same linear contiguous-run
structure.

E1B therefore uses a minimum-overlap run-placement control.

The new construction keeps the scientific variable of interest —
perturbation location — while preserving the important temporal
structure of each CSDPC artifact.

---

## 2. Source Runs

Within every completed trajectory, identify the state-modified mask:

```text
state_modified[t] =
    any(
        poisoned_observation[t]
        != clean_observation[t]
    )
```

Partition the modified indices into maximal contiguous runs.

For every source run preserve:

- trajectory identity;
- run order;
- run length;
- exact ordered state-delta sequence.

For source run `j`:

```text
delta_j[k] =
    poisoned_observation[source_start_j + k]
    - clean_observation[source_start_j + k]
```

---

## 3. Target Layout

For one trajectory of length `L`, suppose the ordered source run lengths
are:

```text
l_1, l_2, ..., l_m
```

Choose target starts:

```text
s_1, s_2, ..., s_m
```

subject to:

```text
0 <= s_1
```

```text
s_j + l_j < s_(j+1)
```

```text
s_m + l_m <= L
```

The strict separation condition keeps at least one clean transition
between adjacent target runs.

Therefore:

- target runs never overlap each other;
- target runs never merge;
- source run order is preserved;
- the exact ordered run-length sequence is preserved.

---

## 4. Minimum Source-Location Overlap

Target runs are allowed to overlap original CSDPC-modified locations
when unavoidable.

For a target layout define:

```text
location_overlap =
    count(
        target_modified_mask
        AND
        source_modified_mask
    )
```

The target layout is selected to minimize:

```text
location_overlap
```

over all feasible ordered target-run placements.

This optimization is solved exactly with dynamic programming.

The original source layout is always feasible, so a valid solution
always exists.

This removes the feasibility problem in the earlier E1B constructions.

---

## 5. Five Deterministic Replicates

For every trajectory, the primary optimization criterion is the integer
source-location overlap count.

When several layouts have the same minimum overlap, a very small
deterministic random tie-break cost is used.

Five fixed tie-break seeds produce:

```text
replicate = 0, 1, 2, 3, 4
```

All five replicates minimize the primary overlap objective.

If the optimum is unique, multiple replicates may produce the same
layout. This is reported rather than artificially forcing a worse
control.

The five controls are not independent model replications.

---

## 6. Delta Transfer

For source run `j` relocated to target start `s_j`:

```text
control_observation[s_j + k] =
    clean_observation[s_j + k]
    + delta_j[k]
```

for every offset `k` in the run.

No state-delta vector is:

- rescaled;
- sign-randomized;
- reordered inside its source run.

Actions remain clean.

---

## 7. Preserved Quantities

E1B preserves:

- clean DT and DT+MTM models;
- trajectory identity;
- number of modified state transitions per trajectory;
- number of contiguous runs per trajectory;
- source run order;
- exact ordered run-length sequence;
- exact ordered state-delta sequence inside every run;
- poison condition;
- poison rate;
- rewards;
- terminals;
- timeouts;
- RTG construction;
- state normalization.

E1B changes:

- absolute temporal placement of the state-perturbation runs;
- the clean state onto which each delta vector is transferred;
- the amount of overlap with CSDPC-selected temporal locations.

---

## 8. Required Invariants

Before any model inference, every control must satisfy:

```text
source_modified_count
==
target_modified_count
```

For every trajectory:

```text
source_run_lengths_in_order
==
target_run_lengths_in_order
```

For every transferred run:

```text
stored_control_observation
==
clean_target_observation
+ source_delta
```

at dataset storage precision.

Also report:

```text
source_target_overlap_count
source_target_overlap_fraction
moved_away_fraction
```

where:

```text
moved_away_fraction =
    1 - source_target_overlap_fraction
```

---

## 9. Mandatory Preflight

Before running any DT or DT+MTM inference, run the construction-only
preflight across:

```text
12 artifacts x 5 replicates = 60 controls
```

The preflight must complete without error.

It verifies all relocation invariants and reports the achieved overlap.

If preflight fails, the model-level E1B experiment must not be run.

---

## 10. Endpoint Populations

Because E1B changes perturbation locations, direct and history-only
endpoint populations are recomputed from each target mask.

For each control:

- direct endpoints are target-modified endpoints;
- history-only endpoints are clean endpoints with at least one target
  modification in the preceding causal context.

The same Group 4E deterministic sample-size limits are reused.

E1B does not claim endpoint-level pairing with the original CSDPC
condition.

---

## 11. Primary Outcome

The primary model-level quantity remains:

```text
C_block2 =
    mean(block2_shift_joint)
    - mean(block2_shift_dt)
```

Report:

- `C_block2` for every control realization;
- artifact-level mean and standard deviation across five controls;
- seed-level mean;
- overall mean;
- ratio to the frozen E1 CSDPC reference.

---

## 12. Secondary Outcomes

Also report:

```text
A_direct
A_history
D_block2
```

using the same definitions as E1.

---

## 13. Interpretation

E1 already showed that randomizing perturbation direction while keeping
CSDPC locations leaves the historical-state mechanism largely intact.

E1B now tests whether changing location while preserving run structure
changes that mechanism.

### Evidence for location-insensitive behavior

The location-insensitive interpretation becomes stronger if:

- `C_block2` remains positive for all three clean model seeds;
- `A_history` remains positive;
- the effect remains substantial relative to the E1 CSDPC reference;
- results are stable across the minimum-overlap controls.

### Evidence for location-sensitive behavior

The location-sensitive interpretation becomes stronger if:

- minimum-overlap relocation substantially reduces `C_block2`;
- or seed-level `C_block2` changes sign after relocation;
- while the E1 CSDPC reference remains positive.

No new post-hoc numerical threshold is introduced.

---

## 14. Claim Boundary

E1B is a mechanistic location control.

It is not:

- a new poisoning attack;
- a policy-return robustness benchmark;
- evidence that the mechanism explains Group 4D stress-response gap
  `G`.

The aborted zero-overlap and circular-shift partial runs are
implementation diagnostics only.

---

## 15. Outputs

Preflight:

```text
experiments/postfinal_controls/e1b_min_overlap_preflight.json
```

Model-level E1B result:

```text
experiments/postfinal_controls/e1b_min_overlap_run_control.json
```

Checkpoint during model inference:

```text
experiments/postfinal_controls/e1b_min_overlap_run_control.partial.json
```
