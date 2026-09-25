# E1B Protocol Amendment — Circular-Shift Location Control

## Status

`PREDECLARED AMENDMENT AFTER FEASIBILITY FAILURE`

This amendment was introduced after the original E1B relocation implementation failed on a dense trajectory before completing the experiment.

The failure was:

```text
trajectory=297
trajectory length=114
source_modified_count=38
source_run_lengths=[19, 9, 5, 5]
```

The original E1B protocol required every relocated modified transition to avoid every original CSDPC-modified transition.

That constraint is not guaranteed to be feasible for dense trajectories.

The partial rows printed before the failure are not used for scientific interpretation.

---

## 1. Reason for the Amendment

The scientific question remains unchanged:

> Does the Group 4E historical-state mechanism depend on the temporal locations selected by CSDPC?

The amendment changes only the construction of the location control.

The original control attempted to repack each contiguous modified run into completely untouched positions.

The amended control instead applies a within-trajectory circular shift to the complete state-perturbation field.

This guarantees a well-defined location transformation whenever valid nonzero shifts exist and preserves temporal structure more faithfully than independently relocating runs.

---

## 2. Circular-Shift Construction

For one completed trajectory of length `L`, let the original local modified indices be:

```text
I_source
```

For a nonzero shift `d`:

```text
1 <= d <= L - 1
```

define:

```text
target_index =
    (source_index + d) mod L
```

The exact source state-delta vector is transferred:

```text
delta_source[i] =
    poisoned_observation[i]
    - clean_observation[i]

control_observation[target_index] =
    clean_observation[target_index]
    + delta_source[i]
```

No perturbation vector is rescaled or randomized.

---

## 3. Valid Shift

A shift is valid only if all of the following hold.

### 3.1 Nonzero location shift

```text
d != 0
```

### 3.2 Same modified-state count

The shifted mask contains exactly the same number of modified transitions as the source mask.

### 3.3 Same linear run-length multiset

The shifted modified mask must preserve:

```text
sorted(source_run_lengths)
==
sorted(shifted_run_lengths)
```

This rejects circular shifts that split a run across the trajectory boundary or merge two runs in linear time.

### 3.4 Same trajectory

Every shifted perturbation remains inside the same completed trajectory.

---

## 4. Source-Target Overlap

Unlike the original infeasible protocol, the amended protocol does not require:

```text
source_target_overlap == 0
```

Some dense trajectories cannot satisfy that requirement.

Instead, every valid shift is scored by:

```text
overlap_count =
    count(
        source_modified_mask
        AND
        shifted_modified_mask
    )
```

and:

```text
overlap_fraction =
    overlap_count
    / source_modified_count
```

Lower overlap means more of the perturbation pattern has moved away from CSDPC-selected locations.

---

## 5. Five Deterministic Replicates

For every modified trajectory:

1. enumerate all valid nonzero shifts;
2. sort shifts by increasing `overlap_count`;
3. break equal-overlap ties with a deterministic frozen RNG;
4. use the first five ranked shifts.

Global E1B replicate `r` uses ranked shift `r` for every modified trajectory.

Therefore:

```text
replicate = 0, 1, 2, 3, 4
```

The five replicates are intentionally low-overlap location controls.

They are not independent model replications.

---

## 6. Preserved Quantities

The amended E1B control preserves:

- trajectory identity;
- exact number of modified transitions per trajectory;
- exact linear contiguous-run length multiset;
- exact source state-delta vectors;
- one-to-one mapping of source deltas to shifted target positions;
- clean models;
- poison condition and poison rate.

It changes:

- absolute temporal location of the perturbation pattern;
- the clean state on which each perturbation vector is applied;
- CSDPC's original location targeting.

---

## 7. Required Invariants

For every trajectory and every replicate:

```text
source_modified_count
==
shifted_modified_count
```

and:

```text
sorted(source_run_lengths)
==
sorted(shifted_run_lengths)
```

and every transferred delta must satisfy:

```text
stored_control_observation
==
clean_target_observation
+ source_delta
```

at dataset storage precision.

Also report:

- selected shift;
- source-target overlap count;
- source-target overlap fraction;
- fraction of modified positions moved away from original CSDPC locations.

---

## 8. Primary Outcome

The primary outcome remains:

```text
C_block2 =
    mean(block2_shift_joint)
    - mean(block2_shift_dt)
```

Report:

- `C_block2` for every artifact and shift replicate;
- artifact-level mean and standard deviation across five shifts;
- seed-level mean;
- overall mean;
- ratio to the frozen E1 CSDPC reference.

---

## 9. Secondary Outcomes

Also report:

```text
A_direct
A_history
D_block2
```

using the shifted perturbation mask to define direct and history-only populations.

---

## 10. Interpretation

E1 already established that perturbation direction is not necessary for the observed mechanism when CSDPC locations are held fixed.

The amended E1B control tests whether substantially moving the temporal perturbation pattern changes the mechanism.

### Location-insensitive evidence

The location-insensitive interpretation becomes stronger if:

- shifted `C_block2` remains positive across all three clean model seeds;
- shifted `A_history` remains positive;
- these effects remain substantial relative to the E1 CSDPC reference;
- results are stable across the five low-overlap shifts.

### Location-sensitive evidence

The location-sensitive interpretation becomes stronger if:

- shifting the perturbation pattern substantially reduces `C_block2`;
- or seed-level `C_block2` changes sign after shifting;
- while the frozen E1 CSDPC reference remains positive.

No new post-hoc numerical threshold is introduced.

---

## 11. Claim Boundary

This remains a mechanistic location control.

It is not:

- a new poisoning attack;
- a return-level robustness benchmark;
- evidence that the mechanism explains Group 4D `G`.

The partial output from the failed zero-overlap implementation is not part of the evidence base.

---

## 12. Output

The amended analysis writes:

```text
experiments/postfinal_controls/e1b_circular_shift_control.json
```

Checkpoint file during execution:

```text
experiments/postfinal_controls/e1b_circular_shift_control.partial.json
```
