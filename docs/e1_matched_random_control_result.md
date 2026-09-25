# E1 — Matched Random-Direction Historical-State Control Result

## Status

`COMPLETE`

E1 is a post-final mechanistic control performed after the frozen `walker2d-final-v1` study.

No model was retrained and no frozen final-study conclusion was modified.

---

## Question

E1 tested whether the Group 4E historical-state effect depends on the specific direction of the CSDPC state perturbations.

The control kept the original CSDPC-modified transition indices and preserved the absolute magnitude of every state-perturbation component, while independently randomizing perturbation signs.

---

## Control Validation

All 12 artifacts passed the matching invariants.

For every artifact:

- the state-modified transition indices were identical to CSDPC;
- component-wise absolute perturbation magnitudes were preserved;
- per-transition L1, L2, and Linf perturbation magnitudes were preserved up to floating-point tolerance;
- actions remained clean;
- rewards, terminals, timeouts, normalization, and model parameters remained unchanged.

The largest reported component-wise magnitude mismatch was approximately:

```text
4.77e-7
```

The largest reported L2 mismatch was below:

```text
4.18e-7
```

These differences are floating-point storage effects.

---

## Main Result

| Metric | CSDPC state | Matched random-sign control |
| --- | ---: | ---: |
| Mean direct action-sensitivity gap `A_direct` | `-0.098942` | `-0.097112` |
| Positive `A_direct` artifacts | `0/12` | `0/12` |
| Mean history-only gap `A_history` | `+0.008444` | `+0.007930` |
| Positive `A_history` artifacts | `12/12` | `12/12` |
| Mean block-2 propagation gap `C_block2` | `+0.104385` | `+0.102296` |
| Positive `C_block2` artifacts | `12/12` | `12/12` |
| Mean block-2 attention-rerouting gap `D_block2` | `+0.00136793` | `+0.00139770` |
| Positive `D_block2` artifacts | `10/12` | `11/12` |

The matched random-sign control retained approximately:

```text
0.97998
```

of the original mean `C_block2` magnitude.

The paired control-minus-CSDPC block-2 difference was:

```text
mean   = -0.00208929
median = -0.00548719
```

---

## Seed-Level Block-2 Result

| Clean model seed | CSDPC mean `C_block2` | Random-sign mean `C_block2` |
| ---: | ---: | ---: |
| `0` | `+0.087076` | `+0.086461` |
| `1` | `+0.124253` | `+0.126367` |
| `2` | `+0.101827` | `+0.094060` |

All three distinct clean DT / DT+MTM model pairs retained positive block-2 amplification under the matched random-sign control.

The 12 artifact rows are not treated as 12 independent model replications.

---

## Interpretation

E1 does not support the explanation that the Group 4E block-2 effect requires the particular state-perturbation direction selected by CSDPC.

Randomizing perturbation direction while preserving attacked locations and exact component-wise magnitudes left the main mechanism result nearly unchanged.

The supported post-final interpretation is therefore:

> Joint MTM training changes how the causal DT propagates historical state perturbations, and this effect is not strongly dependent on the specific CSDPC perturbation direction.

The direct/history reversal also remained intact:

- direct state corruption: DT+MTM was less sensitive than vanilla DT;
- historical state corruption: DT+MTM was more sensitive than vanilla DT.

---

## Remaining Ambiguity

E1 preserved the exact CSDPC-selected transition locations.

Therefore E1 does not determine whether the mechanism depends on where CSDPC places perturbations.

The next control, E1B, tests location dependence while preserving:

- trajectory identity;
- contiguous-run length;
- temporal clustering within each relocated run;
- exact state-perturbation vectors and their order;
- per-trajectory corruption counts.

---

## Relationship to Return-Level Results

E1 is a mechanistic control only.

It does not establish that block-2 amplification explains Group 4D stress-response gap `G`.

The behavioral and mechanistic findings remain separate.

---

## Output

Raw E1 results:

```text
experiments/postfinal_controls/e1_matched_random_control.json
```
