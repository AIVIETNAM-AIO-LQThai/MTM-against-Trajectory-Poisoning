# Research Matrix

## Final completed scope

The finalized study focuses on:

- dataset: `walker2d-medium-v2`
- vanilla Decision Transformer (DT)
- standalone Masked Trajectory Model (MTM)
- joint DT + MTM
- independently reproduced CSDPC-derived trajectory perturbations
- canonical and `s2_overlap_r0` perturbation conditions
- poison rates `rho = 0.01` and `rho = 0.05`
- development/training seeds `0, 1, 2`

The completed study does not include Hopper, HalfCheetah, RDT, adaptive attacks, or a five-seed final matrix.

## Completed experiment sequence

### Group 1 — Clean DT foundation

1. Freeze the Walker2d development configuration.
2. Audit the dataset, trajectory segmentation, RTG computation, normalization, padding, and causality.
3. Train the clean causal Decision Transformer.
4. Evaluate three clean DT seeds.
5. Pass Gate A.

Final status:

`PASS`

Clean DT mean normalized return:

`69.8094`

### Group 2 — Independent CSDPC reproduction

1. Independently implement CSDPC from the public specification.
2. Audit pattern extraction, perturbation budgets, candidate generation, sequence semantics, overlap behavior, and reproducibility.
3. Evaluate the frozen perturbations against CQL.
4. Investigate source-fidelity discrepancies.
5. Freeze the canonical and `s2_overlap_r0` artifacts.
6. Close Group 2 without post-hoc attack-strength tuning.

Final status:

`SOURCE-FIDELITY DISCREPANCY UNRESOLVED`

The project does not claim successful reproduction of the source paper's reported attack strength.

The resulting poisoned datasets remain valid as frozen, audited stress conditions.

### Group 3 — Standalone MTM

1. Audit the reference MTM implementation.
2. Reproduce the clean trajectory-reconstruction task.
3. Verify checkpoint/resume behavior.
4. Evaluate reconstruction performance.
5. Freeze the standalone MTM result.

Final status:

`MTM REPRODUCTION: PASS`

The standalone MTM checkpoint is a validation artifact and is not used to initialize the primary Group-4 joint model.

### Group 4A — DT + MTM architecture integration

The primary joint model was built with:

- separate causal DT and MTM Transformer passes;
- shared DT state/action representations through the project-defined bridges;
- independent DT and MTM objectives;
- training from scratch;
- no initialization from the Group-3 standalone MTM checkpoint.

Final status:

`PASS`

### Group 4B — Clean DT + MTM compatibility

Frozen configuration:

- `lambda_mtm = 1.0`
- DT batch size: `64`
- MTM batch size: `512`
- training updates: `100000`
- seeds: `0, 1, 2`

Clean DT + MTM mean normalized return:

`67.4781`

Predeclared compatibility threshold:

`62.8284`

Final status:

`PASS`

This establishes clean compatibility only. It does not establish improved clean performance.

### Group 4C — Vanilla-DT stress transfer

The four frozen perturbation cells were tested against vanilla DT before interpreting DT + MTM robustness:

| Condition     |  rho | Mean paired DT degradation | Classification            |
| ------------- | ---: | -------------------------: | ------------------------- |
| canonical     | 0.01 |                  `-1.3607` | no detectable degradation |
| canonical     | 0.05 |                  `-2.8701` | no detectable degradation |
| s2_overlap_r0 | 0.01 |                  `-0.4472` | no detectable degradation |
| s2_overlap_r0 | 0.05 |                  `-1.9879` | no detectable degradation |

Final status:

`NO DETECTABLE VANILLA-DT DEGRADATION`

Therefore the frozen stress conditions do not establish an effective vanilla-DT attack under the predeclared Group-4C criterion.

### Group 4D — Matched DT vs DT + MTM stress response

For each matched condition:

`G = Delta_DT - Delta_joint`

Positive `G` means DT + MTM loses less relative to its own clean baseline.

Negative `G` means DT + MTM loses more.

Cell-level mean results:

| Condition     |  rho |    Mean G |
| ------------- | ---: | --------: |
| canonical     | 0.01 | `-1.2246` |
| canonical     | 0.05 | `-3.4587` |
| s2_overlap_r0 | 0.01 | `-2.1064` |
| s2_overlap_r0 | 0.05 | `-2.9642` |

Final status:

`NO CONSISTENT DT+MTM ROBUSTNESS ADVANTAGE`

All four cell-level mean stress-response gaps are negative, with substantial seed-level heterogeneity.

Because Group 4C did not establish a successful vanilla-DT attack, these results are interpreted as stress-response differences rather than defense effectiveness.

### Group 4E — Mechanism analysis

Mechanism analyses examined:

- aggregate optimization diagnostics;
- temporal optimization diagnostics;
- parameter displacement magnitude;
- parameter displacement direction;
- raw poison localization;
- sequential poison exposure;
- attack-selection localization;
- perturbation-vector geometry;
- frozen clean-policy sensitivity;
- temporal distance of historical corruption;
- state vs action modality;
- state-representation sensitivity;
- layerwise causal propagation;
- attention routing.

Several simple explanations for the seed-dependent stress-response gap were not supported.

A consistent model-level effect was identified:

1. direct endpoint corruption produces lower DT + MTM sensitivity than vanilla DT;
2. history-only corruption produces higher DT + MTM sensitivity;
3. the historical effect persists across the full 20-step causal context;
4. the effect is almost entirely caused by observation/state corruption rather than historical actions;
5. the shared state representation itself becomes less sensitive, not more sensitive;
6. the extra historical sensitivity emerges during causal temporal processing;
7. the strongest consistent amplification appears by Transformer block 2;
8. block 2 also shows the clearest tendency toward increased attention routing to corrupted historical state positions.

Final status:

`HISTORICAL-STATE PROCESSING EFFECT IDENTIFIED; GROUP 4E CLOSED`

## Final thesis status

### Original hypothesis

> DT + MTM improves robustness against trajectory poisoning.

Result:

`NOT SUPPORTED`

The frozen perturbations did not consistently degrade vanilla DT, and the joint DT + MTM model did not show a consistent matched robustness advantage.

### Refined supported conclusion

> MTM changes how a causal Decision Transformer processes corrupted historical state information.

The evidence indicates that MTM contracts the immediate state representation of the perturbation while increasing its downstream influence through deeper causal temporal processing.

## Completed evidence matrix

| Component                      | Dataset / conditions | Seeds         | Final result                                  |
| ------------------------------ | -------------------- | ------------- | --------------------------------------------- |
| Clean DT                       | Walker2d medium-v2   | 0, 1, 2       | PASS                                          |
| Independent CSDPC reproduction | Walker2d medium-v2   | 0, 1, 2       | source-fidelity discrepancy unresolved        |
| Standalone MTM                 | clean Walker2d       | reference run | PASS                                          |
| Clean DT + MTM                 | clean Walker2d       | 0, 1, 2       | compatibility PASS                            |
| Vanilla-DT stress transfer     | 4 frozen cells       | 0, 1, 2       | no detectable degradation                     |
| DT vs DT + MTM comparison      | 4 frozen cells       | 0, 1, 2       | no consistent robustness advantage            |
| Mechanism analysis             | 12 frozen artifacts  | 0, 1, 2       | historical-state processing effect identified |

## Future work

The following are not part of the finalized evidence base:

- Hopper;
- HalfCheetah;
- five-seed final evaluation;
- Behavior Cloning transfer baseline;
- RDT or other robust sequence-model baselines;
- crossed attack-seed × training-seed experiments;
- stronger independently validated poisoning attacks;
- adaptive attacks;
- MTM mask-length ablations;
- attack-length × mask-length studies;
- regularization/capacity controls;
- dataset-quality generalization.

These must be treated as new experiments.

Future experiments must not retroactively alter the frozen conclusions of the completed Walker2d study.
