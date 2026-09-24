# MTM-Enhanced Decision Transformer Under Trajectory-Poisoning Stress

This repository studies whether Masked Trajectory Modeling (MTM) changes the robustness of a causal Decision Transformer (DT) under trajectory-level perturbations in offline reinforcement learning.

The completed study focuses on:

- dataset: `walker2d-medium-v2`
- vanilla Decision Transformer
- standalone Masked Trajectory Model
- joint DT + MTM
- frozen CSDPC-derived trajectory-perturbation stress conditions
- poison rates `rho = 0.01` and `rho = 0.05`
- seeds `0, 1, 2`

## Research question

The original hypothesis was:

> DT + MTM improves the robustness of Decision Transformer against trajectory poisoning.

The completed experiments do **not** support that claim.

However, the study identifies a narrower and consistent mechanistic effect:

> MTM changes how the causal Decision Transformer processes corrupted historical state information. Although the shared state representation becomes less locally sensitive to perturbations, corrupted historical states propagate more strongly through deeper causal temporal processing, with the clearest amplification emerging around the second Transformer block.

## Final project status

The Walker2d study is complete.

| Stage    | Purpose                                                | Final status                                          |
| -------- | ------------------------------------------------------ | ----------------------------------------------------- |
| Group 1  | Clean Decision Transformer baseline                    | PASS                                                  |
| Group 2  | Independent CSDPC reproduction and vulnerability study | SOURCE-FIDELITY DISCREPANCY UNRESOLVED; CLOSED        |
| Group 3  | Standalone MTM reproduction                            | PASS                                                  |
| Group 4A | DT + MTM architecture integration                      | PASS                                                  |
| Group 4B | Clean DT + MTM compatibility                           | PASS                                                  |
| Group 4C | Frozen perturbation transfer to vanilla DT             | NO DETECTABLE DEGRADATION                             |
| Group 4D | Matched DT vs DT + MTM stress comparison               | NO CONSISTENT ROBUSTNESS ADVANTAGE                    |
| Group 4E | Mechanism analysis                                     | HISTORICAL-STATE PROCESSING EFFECT IDENTIFIED; CLOSED |

## Group 1 — Clean Decision Transformer

The clean three-seed DT baseline achieved normalized returns of approximately:

- seed 0: `69.3220`
- seed 1: `68.6765`
- seed 2: `71.4297`
- mean: `69.8094`

The clean DT baseline passed the project acceptance gate and became the frozen reference learner.

See: `docs/group1_clean_result.md`

## Group 2 — Independent CSDPC reproduction

CSDPC was independently implemented from the published specification and subjected to structural, integrity, budget, perturbation, reachability, replay, and learner-response audits.

The implementation did **not** reproduce the attack strength reported by the source paper.

Final Group-2 status:

`SOURCE-FIDELITY DISCREPANCY UNRESOLVED`

The generated datasets are therefore retained as:

- independent CSDPC reproduction artifacts;
- CSDPC source-semantics sensitivity artifacts;
- frozen audited trajectory-perturbation stress conditions.

They must not be described as a successful reproduction of the source paper's reported attack strength.

See:

`docs/group2_csdpc_final_status.md`

## Group 3 — Standalone MTM

The standalone project-aligned MTM implementation successfully learned the clean Walker2d trajectory-reconstruction task.

Final status:

`MTM REPRODUCTION: PASS`

The standalone MTM checkpoint is used as a validation artifact only. It is not used to initialize the primary joint DT + MTM model.

See:

`docs/mtm_stage_a_result.md`

## Group 4 — DT + MTM

### Clean compatibility

The joint DT + MTM model:

- trains from scratch;
- keeps separate DT and MTM Transformer passes;
- shares the DT state/action input representations through the frozen project architecture;
- uses `lambda_mtm = 1.0`;
- preserves the vanilla DT policy architecture.

Clean DT + MTM normalized returns:

- seed 0: `69.9740`
- seed 1: `64.2323`
- seed 2: `68.2282`
- mean: `67.4781`

The predeclared clean compatibility threshold was `62.8284`.

Result:

`GROUP 4B FINAL CLEAN GATE: PASS`

This establishes clean compatibility, not clean-performance improvement.

See:

`docs/group4_clean_result.md`

### Vanilla-DT stress transfer

Before interpreting DT + MTM robustness, the frozen perturbations were tested against vanilla DT.

None of the four frozen conditions satisfied the predeclared criterion for consistent vanilla-DT degradation.

| Condition     |  rho | Mean paired DT degradation | Result                    |
| ------------- | ---: | -------------------------: | ------------------------- |
| canonical     | 0.01 |                  `-1.3607` | no detectable degradation |
| canonical     | 0.05 |                  `-2.8701` | no detectable degradation |
| s2_overlap_r0 | 0.01 |                  `-0.4472` | no detectable degradation |
| s2_overlap_r0 | 0.05 |                  `-1.9879` | no detectable degradation |

Therefore the project does not establish an effective vanilla-DT poisoning attack under the frozen stress conditions.

See:

`docs/group4c_stress_transfer_result.md`

### Matched DT vs DT + MTM stress response

For matched comparisons:

`G = Delta_DT - Delta_joint`

Positive `G` means DT + MTM loses less relative to its own clean baseline.

Negative `G` means DT + MTM loses more.

Cell-level mean stress-response gaps were:

| Condition     |  rho |    Mean G |
| ------------- | ---: | --------: |
| canonical     | 0.01 | `-1.2246` |
| canonical     | 0.05 | `-3.4587` |
| s2_overlap_r0 | 0.01 | `-2.1064` |
| s2_overlap_r0 | 0.05 | `-2.9642` |

All four cell-level mean `G` values are negative, and individual seed responses are heterogeneous.

Therefore:

> DT + MTM does not show a consistent robustness advantage over vanilla DT under the frozen stress conditions.

Because Group 4C did not establish an effective vanilla-DT attack, this result must not be interpreted as a successful or failed defense against a validated attack. It is a matched stress-response comparison.

See: `docs/group4d_result.md`

## Group 4E — Mechanism analysis

Although the original robustness hypothesis was not supported, the mechanism analysis identified a consistent effect of MTM training.

### Main finding

Direct endpoint corruption and historical corruption affect DT + MTM differently.

For direct endpoint corruption:

> DT + MTM is less locally sensitive than vanilla DT.

For history-only corruption:

> DT + MTM is more sensitive than vanilla DT.

This history-only effect was positive across all 12 frozen artifacts.

### Temporal distance

The excess DT + MTM historical sensitivity persisted across the entire 20-step causal context.

Mean joint-minus-DT sensitivity difference:

- distance 1-3: `+0.018905`
- distance 4-7: `+0.007533`
- distance 8-12: `+0.006121`
- distance 13-19: `+0.005262`

The effect is strongest for recent history but remains positive even for distant historical corruption.

### State vs action

The historical sensitivity effect is almost entirely carried by observation/state corruption.

Mean excess sensitivity:

- state-only: `+0.008445`
- action-only: `+0.000023`
- state + action: `+0.008429`

Historical action corruption contributes essentially no additional DT + MTM sensitivity.

### State representation

MTM does not make the shared state input representation more sensitive.

Instead:

- raw state-embedding perturbation is approximately `0.57x` vanilla DT;
- pre-Transformer token perturbation is approximately `0.94x` vanilla DT.

Therefore the historical amplification does not originate in the shared state embedding.

### Layerwise propagation

The amplification emerges during causal temporal processing.

Mean joint-minus-DT hidden-state sensitivity:

- Transformer input: `0.000000`
- block 1: `+0.003759`
- block 2: `+0.104363`
- block 3: `+0.102596`
- action prediction: `+0.008391`

Block 2 is the first layer where the amplification becomes strongly and consistently positive across all 12 artifacts.

### Attention routing

Block 2 is also the only Transformer layer showing a clear tendency toward increased perturbation-induced attention routing to corrupted historical state positions.

- block 1: positive in `3/12`
- block 2: positive in `10/12`
- block 3: positive in `4/12`

This is consistent with block-2 attention contributing to the historical-state amplification, but it does not establish attention routing as the sole causal mechanism.

See:

`docs/group4e_result.md`

## Final thesis verdict

### Original thesis

> DT + MTM improves robustness against trajectory poisoning.

**Not supported by the completed experiments.**

The frozen perturbations did not consistently degrade vanilla DT, and DT + MTM did not show a consistent matched robustness advantage.

### Refined supported conclusion

> MTM changes how a causal Decision Transformer propagates corrupted historical state information through deeper temporal processing.

This effect is:

- consistent across the frozen artifacts;
- driven primarily by historical state corruption;
- not caused by increased input-embedding sensitivity;
- generated inside the causal Transformer;
- clearest around Transformer block 2.

## Scientific claim boundary

Supported:

- the clean DT baseline passed the project acceptance gate;
- standalone MTM passed the project-aligned reconstruction gate;
- clean DT + MTM passed the compatibility gate;
- the frozen perturbations did not produce detectable vanilla-DT degradation under the predeclared Group-4C criterion;
- DT + MTM did not show a consistent robustness advantage;
- MTM consistently changes historical-state processing in the causal DT policy.

Not supported:

- successful reproduction of the source paper's CSDPC attack strength;
- an effective vanilla-DT poisoning attack under the current frozen artifacts;
- DT + MTM as a demonstrated trajectory-poisoning defense;
- attention rerouting alone as the cause of the observed block-2 amplification;
- generalization beyond the completed Walker2d study.

## Branch structure

The research was developed incrementally:

`baseline/DT-walker2d`

→ clean Decision Transformer baseline

`exp/csdpc-vulnerability`

→ independent CSDPC reproduction and Group-2 audits

`exp/mtm-reproduction`

→ standalone MTM reproduction

`exp/dt-mtm-defense`

→ joint DT + MTM integration, stress comparison, and mechanism analysis

## Important documents

## Important documents

- `docs/project_final_report.md`
- `docs/group1_clean_result.md`
- `docs/group2_csdpc_final_status.md`
- `docs/mtm_stage_a_result.md`
- `docs/group4_clean_result.md`
- `docs/group4c_stress_transfer_result.md`
- `docs/group4d_result.md`
- `docs/group4e_result.md`

## Environments and testing

This project spans several historically distinct software environments.

A single Python environment is not intended to execute every test accumulated across Groups 1–4.

The main environment records are:

- Group 1 DT: `environment.reference.yml`
- Group 2 CQL: `configs/environments/g2-cql.yml`
- Group 3 MTM: `environment.group3-mtm.yml`
- Group 4 DT+MTM: `configs/environments/g4-dt-mtm.yml`

Additional runtime and lock snapshots are retained where available.

Tests should be executed in the environment corresponding to the experimental stage they validate.

In particular, the MTM/DT+MTM environment is not expected to contain the legacy D4RL/Gym/MuJoCo stack required by Group-2 tests.

The clean-DT evaluation-reproducibility integration test also requires a local trained checkpoint that is intentionally excluded from the final Git tree.

See:

`docs/testing_environments.md`

and:

`tests/README.md`

## Future work

The following were part of the broader research plan but are **not part of the completed evidence base**:

- Hopper and HalfCheetah;
- larger seed counts;
- RDT and additional robust baselines;
- Behavior Cloning transfer studies;
- crossed attack-seed × training-seed experiments;
- adaptive or stronger independently validated attacks;
- MTM mask-length and attack-length ablations;
- dataset-quality generalization.

Any future extension should be treated as a new experiment rather than changing the frozen conclusions of this Walker2d study.
