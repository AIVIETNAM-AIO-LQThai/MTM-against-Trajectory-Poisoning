# E1B — Within-Trajectory Relocated-Run Control

## Status

`POST-FINAL CONTROL — PREDECLARED`

E1B begins after E1 and after the frozen `walker2d-final-v1` study.

It does not modify any frozen final-study artifact or conclusion.

---

## 1. Research Question

E1 showed that the Group 4E historical-state mechanism survives randomization of CSDPC perturbation direction when the original CSDPC-selected transition locations are retained.

E1B asks:

> Does the historical-state mechanism also survive when the perturbations are moved away from the CSDPC-selected locations?

No model is retrained.

---

## 2. Frozen Models

Use exactly the clean models already used by Group 4E and E1:

- clean vanilla DT;
- clean DT policy branch extracted from DT+MTM.

Training seeds:

```text
0, 1, 2
```

All models remain in inference mode.

---

## 3. Source Artifacts

Use the same 12 frozen CSDPC artifacts:

| Condition       | Poison rate | Seeds     |
| --------------- | ----------: | --------- |
| `canonical`     |      `0.01` | `0, 1, 2` |
| `canonical`     |      `0.05` | `0, 1, 2` |
| `s2_overlap_r0` |      `0.01` | `0, 1, 2` |
| `s2_overlap_r0` |      `0.05` | `0, 1, 2` |

Only state perturbations are used in E1B.

Actions remain clean.

---

## 4. Why E1B Relocates Runs Instead of Individual Transitions

CSDPC state modifications are temporally clustered because selected attack windows modify contiguous transitions.

Randomizing each modified transition independently would change both:

- attack location;
- temporal clustering.

That would confound the interpretation.

E1B therefore identifies every contiguous run of state-modified transitions and relocates the entire run as a unit.

---

## 5. Relocation Construction

For every completed trajectory:

1. identify all original CSDPC state-modified transition indices;
2. partition them into maximal contiguous runs;
3. for every run, preserve:
   - run length;
   - exact ordered sequence of state-delta vectors;
4. choose a new run start uniformly from valid locations in the same trajectory;
5. require the relocated run to:
   - remain entirely inside the same trajectory;
   - not overlap any original CSDPC-modified state transition;
   - not overlap any previously relocated run;
   - not be directly adjacent to another relocated run;
6. apply the original ordered delta sequence at the relocated run.

For a source run:

```text
source_delta[k] =
    poisoned_observation[source_start + k]
    - clean_observation[source_start + k]
```

the relocated control is:

```text
control_observation[target_start + k] =
    clean_observation[target_start + k]
    + source_delta[k]
```

for every offset `k` in the run.

No perturbation vector is rescaled.

---

## 6. Preserved Quantities

E1B preserves exactly:

- the number of state-modified transitions;
- the number of modified transitions per trajectory;
- the multiset of contiguous-run lengths per trajectory;
- the exact ordered state-delta vectors inside every relocated run;
- the global perturbation-vector magnitude distribution;
- trajectory identity;
- poison condition;
- poison rate;
- clean model pair.

E1B changes:

- the absolute temporal location of the modified runs;
- the clean state to which each delta vector is applied;
- the rare-pattern targeting induced by the original CSDPC selection.

---

## 7. Required Invariants

Every relocation replicate must pass all of the following.

### 7.1 No source-location reuse

```text
source_modified_indices
INTERSECT
relocated_modified_indices
==
empty
```

### 7.2 Same total number of modified states

```text
count(relocated_modified_indices)
==
count(source_modified_indices)
```

### 7.3 Same count per trajectory

For every completed trajectory:

```text
relocated_modified_count
==
source_modified_count
```

### 7.4 Same run-length multiset per trajectory

For every completed trajectory:

```text
sorted(relocated_run_lengths)
==
sorted(source_run_lengths)
```

### 7.5 Exact delta transfer

For every relocation record and every offset in the run:

```text
relocated_delta
==
source_delta
```

up to floating-point storage tolerance.

### 7.6 Untouched quantities

E1B must not modify:

- actions;
- rewards;
- terminals;
- timeouts;
- RTG construction;
- trajectory boundaries;
- normalization;
- model parameters.

---

## 8. Relocation Replicates

A single random relocation can be unusually favorable or unfavorable.

E1B therefore uses five deterministic relocation replicates for every artifact:

```text
replicate = 0, 1, 2, 3, 4
```

Each replicate uses a predeclared deterministic RNG seed.

This produces:

```text
12 artifacts x 5 relocation replicates = 60 control realizations
```

These 60 rows are not independent model replications.

They reuse only three distinct clean model pairs.

---

## 9. Endpoint Populations

Unlike E1, E1B changes the perturbation locations.

Therefore the direct and history-only endpoint populations necessarily change.

For each relocation replicate:

- direct endpoints are defined from the relocated state-modified mask;
- history-only endpoints are defined from the relocated state-modified mask;
- deterministic sampling is then applied using the same sample-size limits as Group 4E.

E1B does not claim endpoint-level pairing with the original CSDPC condition.

The primary comparison is performed at artifact and clean-model-seed level.

---

## 10. Primary Outcome

The primary quantity remains:

```text
C_block2 =
    mean(block2_shift_joint)
    - mean(block2_shift_dt)
```

For E1B report:

- `C_block2` for every relocation replicate;
- mean and standard deviation across the five relocation replicates for every artifact;
- mean relocated `C_block2` for every clean model seed;
- overall relocated mean `C_block2`;
- original E1 CSDPC `C_block2` as a fixed reference.

---

## 11. Secondary Outcomes

For every relocation replicate also report:

### History-only action sensitivity

```text
A_history =
    mean(action_shift_joint)
    - mean(action_shift_dt)
```

### Direct action sensitivity

```text
A_direct =
    mean(action_shift_joint)
    - mean(action_shift_dt)
```

### Block-2 attention rerouting

```text
D_block2 =
    mean(rerouting_joint_block2)
    - mean(rerouting_dt_block2)
```

Full layerwise values may also be retained in the JSON output.

---

## 12. Interpretation

E1 already showed:

```text
original CSDPC direction
approximately equals
random-sign direction
```

when locations are held fixed.

E1B tests whether location is necessary.

### Location-insensitive pattern

Evidence for a generic perturbation-propagation effect becomes stronger if:

- relocated `C_block2` remains positive for all three clean model seeds;
- positive history-only excess sensitivity remains present;
- the relocated effect remains substantial relative to the original E1 CSDPC reference across relocation replicates.

### Location-sensitive pattern

Evidence that CSDPC-selected locations matter becomes stronger if:

- relocation consistently and substantially reduces `C_block2`;
- or the seed-level relocated `C_block2` changes sign while the original CSDPC reference remains positive.

### Mixed result

Mixed seed behavior or large relocation-replicate variance is reported as:

`INCONCLUSIVE`

No post-hoc threshold is introduced to force a categorical conclusion.

---

## 13. Important Interpretation Limit

Relocating an unchanged delta vector to a different clean state changes its magnitude relative to that target state's local scale.

E1B is therefore a mechanistic location control, not a valid alternative poisoning attack.

It must not be interpreted as an attack-strength comparison.

---

## 14. Relationship to Group 4D

E1B does not test policy return.

It does not establish a causal explanation for Group 4D stress-response gap `G`.

Behavioral and mechanistic findings remain separate.

---

## 15. Output

E1B writes:

```text
experiments/postfinal_controls/e1b_relocated_run_control.json
```

The output contains:

- relocation invariants;
- relocation records;
- five control replicates per artifact;
- action-sensitivity metrics;
- full layerwise metrics;
- attention-routing metrics;
- per-artifact relocation summaries;
- seed-level summaries;
- comparison to the frozen E1 CSDPC reference.
