# Group 4B — Clean DT+MTM compatibility result

## Status

`GROUP 4B FINAL CLEAN GATE: PASS`

This stage evaluates clean compatibility only. It is not evidence of poisoning robustness.

## Frozen configuration

- dataset: `walker2d-medium-v2`
- DT+MTM lambda: `1.0`
- training seeds: `0, 1, 2`
- updates per seed: `100000`
- DT batch size: `64`
- MTM batch size: `512`
- evaluation environment: `Walker2d-v3`
- target return: `5000`
- evaluation episodes per seed: `100`
- evaluation seed base: `30000`
- Group-4 implementation commit: `507a8604072cf4b14692b088282aff2d2cdee138`

## Clean policy results

| seed | vanilla DT | DT+MTM | paired difference (DT+MTM - DT) |
|---:|---:|---:|---:|
| 0 | 69.3220115 | 69.9739698 | +0.6519582 |
| 1 | 68.6764985 | 64.2322717 | -4.4442269 |
| 2 | 71.4296530 | 68.2281774 | -3.2014755 |

Three-seed means:

- vanilla DT: `69.8093877` (reference value used by the frozen protocol: `69.80938768165248`)
- DT+MTM: `67.4781396`
- mean difference: approximately `-2.3312481` normalized-return points
- DT+MTM seed standard deviation: approximately `2.403288`

Frozen compatibility threshold:

`0.90 * 69.80938768165248 = 62.82844891348723`

Observed DT+MTM mean:

`67.47813962719309 >= 62.82844891348723`

Therefore:

`GROUP 4B FINAL CLEAN GATE: PASS`

Interpretation: the joint DT+MTM model is a viable clean baseline under the predeclared compatibility rule. The result does not support a claim that MTM improves clean performance.

## Seed-2 environment incident

During seed-2 training, checkpoints `62000` through `69000` were found to contain NumPy-2-style pickle references (`numpy._core`), while earlier seed-2 checkpoints through `61000`, the seed-0 final checkpoint, the seed-1 final checkpoint, and the restored training environment used NumPy-1-style serialization.

The `69000` checkpoint passed ZIP integrity checking, so the issue was an environment-version discontinuity rather than checkpoint corruption. To keep the final clean seed comparable to the other clean seeds, seed 2 was rolled back to the last NumPy-1-compatible exact checkpoint at update `61000` and retrained through update `100000` under the restored environment. The final seed-2 run completed normally.

## Claim boundary

Allowed:

> Clean DT+MTM training passed the predeclared three-seed compatibility gate and was retained as the clean baseline for robustness experiments.

Not established by this stage:

- that MTM improves clean return;
- that MTM improves poisoning robustness;
- that the independent CSDPC implementation reproduces the attack strength reported by the source paper.
