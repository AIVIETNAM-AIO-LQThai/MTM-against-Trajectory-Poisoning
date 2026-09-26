# Group 4D — Matched DT vs DT+MTM Stress-Comparison Result

## Status

`GROUP 4D COMPLETE`

Group 4D compares vanilla Decision Transformer and the frozen DT+MTM model under exactly the same Group-2 trajectory-perturbation stress conditions.

Group 4C found no detectable vanilla-DT degradation under the predeclared stress-transfer criterion. Therefore Group 4D is a matched stress-response comparison and is not interpreted as evidence of defense against an attack already shown to degrade vanilla DT.

## Definition

For training seed `s`:

`Delta_DT = J_DT_clean(s) - J_DT_poison(condition, rho, s)`

`Delta_joint = J_joint_clean(s) - J_joint_poison(condition, rho, s)`

The matched stress-response gap is:

`G = Delta_DT - Delta_joint`

Interpretation:

- positive `G`: DT+MTM loses less relative to its own clean baseline than vanilla DT;
- negative `G`: DT+MTM loses more relative to its own clean baseline than vanilla DT.

Because Group 4C did not establish an effective vanilla-DT attack, `G` is a descriptive stress-response quantity rather than a defense score.

## Frozen clean references

Fresh Group-4C vanilla-DT clean controls:

| Seed |   Clean DT return |
| ---: | ----------------: |
|    0 | `73.716021030119` |
|    1 | `65.341193981272` |
|    2 | `67.012623942427` |

Mean:

`68.689946317939`

Frozen Group-4B DT+MTM clean controls:

| Seed | Clean DT+MTM return |
| ---: | ------------------: |
|    0 | `69.97396976583133` |
|    1 | `64.23227166791362` |
|    2 | `68.22817744783433` |

Mean:

`67.47813962719309`

## Per-seed matched result

| Condition     |  rho | Seed |  Delta_DT | Delta_joint |          G |
| ------------- | ---: | ---: | --------: | ----------: | ---------: |
| canonical     | 0.01 |    0 | `+1.1828` |   `-2.5767` |  `+3.7595` |
| canonical     | 0.01 |    1 | `-5.5180` |   `+5.2642` | `-10.7822` |
| canonical     | 0.01 |    2 | `+0.2530` |   `-3.0959` |  `+3.3489` |
| canonical     | 0.05 |    0 | `-0.3120` |   `-0.8737` |  `+0.5617` |
| canonical     | 0.05 |    1 | `-5.6514` |   `+0.3601` |  `-6.0115` |
| canonical     | 0.05 |    2 | `-2.6468` |   `+2.2796` |  `-4.9264` |
| s2_overlap_r0 | 0.01 |    0 | `+2.8241` |   `+3.4926` |  `-0.6685` |
| s2_overlap_r0 | 0.01 |    1 | `+0.3804` |   `-0.7169` |  `+1.0972` |
| s2_overlap_r0 | 0.01 |    2 | `-4.5460` |   `+2.2018` |  `-6.7478` |
| s2_overlap_r0 | 0.05 |    0 | `-2.4807` |   `+8.9466` | `-11.4273` |
| s2_overlap_r0 | 0.05 |    1 | `-2.3422` |   `-4.0348` |  `+1.6926` |
| s2_overlap_r0 | 0.05 |    2 | `-1.1409` |   `-1.9830` |  `+0.8421` |

## Cell-level summary

| Condition     |  rho | Mean Delta_DT | Mean Delta_joint |    Mean G |  Median G | G > 0 |
| ------------- | ---: | ------------: | ---------------: | --------: | --------: | ----: |
| canonical     | 0.01 |     `-1.3607` |        `-0.1361` | `-1.2246` | `+3.3489` | `2/3` |
| canonical     | 0.05 |     `-2.8701` |        `+0.5887` | `-3.4587` | `-4.9264` | `1/3` |
| s2_overlap_r0 | 0.01 |     `-0.4472` |        `+1.6592` | `-2.1064` | `-0.6685` | `1/3` |
| s2_overlap_r0 | 0.05 |     `-1.9879` |        `+0.9763` | `-2.9642` | `+0.8421` | `2/3` |

All four cell-level mean `G` values are negative.

However, individual responses are strongly heterogeneous across seeds.

There is also no monotonic improvement in `G` as `rho` increases.

## Interpretation

The matched comparison does not support a consistent robustness advantage for DT+MTM.

The supported result is:

> DT and DT+MTM respond differently to the same frozen trajectory perturbations, but DT+MTM does not show a consistent robustness advantage over vanilla DT.

This stage also cannot establish that DT+MTM is a successful poisoning defense because Group 4C did not establish consistent vanilla-DT degradation under the frozen conditions.

## Seed-design limitation

The frozen paired design uses:

`attack_seed = training_seed`

Same-seed clean baselines control much of the training-seed baseline variation, but the current experiment cannot fully separate:

- attack-artifact seed effects;
- optimization/initialization seed effects.

A crossed attack-seed × training-seed experiment would be required to disentangle them.

Such an experiment is future work and is not part of the finalized evidence.

## Claim boundary

Supported:

- vanilla DT and DT+MTM exhibit heterogeneous responses to the same frozen perturbations;
- all four cell-level mean `G` values are negative;
- DT+MTM does not show a consistent robustness advantage.

Not supported:

- the frozen Group-2 artifacts constitute an effective vanilla-DT attack;
- DT+MTM successfully defends DT against demonstrated poisoning;
- the source-paper CSDPC attack strength was reproduced.

Group 4D is closed.

Mechanistic follow-up is documented in:

`docs/group4e_result.md`
