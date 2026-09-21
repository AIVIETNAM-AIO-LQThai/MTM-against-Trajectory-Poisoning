# Group 4C — Vanilla DT Stress-Transfer Result

## Clean compatibility bridge

Fresh Group 4C clean DT controls:

| Training seed | Normalized return mean |
| --- | ---: |
| 0 | 73.716021030119 |
| 1 | 65.341193981272 |
| 2 | 67.012623942427 |

Three-seed clean mean: **68.689946317939**

Frozen compatibility floor: **62.828448913487**

Result: **PASS**

The clean runner/environment is therefore compatible with the
previously validated vanilla-DT baseline.

## Frozen stress-transfer results

Paired degradation is defined as:

`Delta_DT = J_clean(seed) - J_poison(condition, rho, seed)`

Positive values indicate degradation.

The predeclared practical-effect floor was:

`0.05 * 68.689946317939 = 3.434497315897`

A condition required:

- mean paired degradation >= 3.434497315897, and
- degradation in all 3 paired seeds

to be classified as consistent degradation.

| Condition | rho | Mean degradation | Median degradation | Positive pairs | Classification |
| --- | ---: | ---: | ---: | ---: | --- |
| canonical | 0.01 | -1.3607 | +0.2530 | 2/3 | no detectable degradation |
| canonical | 0.05 | -2.8701 | -2.6468 | 0/3 | no detectable degradation |
| s2_overlap_r0 | 0.01 | -0.4472 | +0.3804 | 2/3 | no detectable degradation |
| s2_overlap_r0 | 0.05 | -1.9879 | -2.3422 | 0/3 | no detectable degradation |

## Conclusion

None of the four frozen Group-2 stress conditions produced detectable
vanilla-DT degradation under the predeclared Group 4C criterion.

Therefore these results do not support a claim that the frozen
trajectory perturbations constitute an effective attack on vanilla
Decision Transformer.

The conditions remain frozen and are carried into Group 4D for matched
DT-vs-DT+MTM comparison and mechanism analysis.

Any Group 4D difference must therefore be described as a robustness or
behavioral difference under the frozen stress conditions, not as
evidence that MTM successfully defends DT from an attack demonstrated
to degrade vanilla DT.

Group 4C was completed without attack retuning or post-hoc condition
selection.
