# Final Project Report — MTM + Decision Transformer Under Trajectory-Poisoning Stress

## Project status

`FINALIZED WALKER2D STUDY`

The completed project studies:

> When and why does masked trajectory modeling change the robustness behavior of a causal Decision Transformer under fixed trajectory perturbations?

The original project hypothesis was:

> DT + MTM improves Decision Transformer robustness against trajectory poisoning.

The completed experiments do not support that stronger hypothesis.

They do support a narrower mechanistic conclusion:

> MTM changes how a causal Decision Transformer processes corrupted historical state information.

## Experimental scope

The finalized evidence base is restricted to:

- `walker2d-medium-v2`;
- vanilla Decision Transformer;
- standalone Masked Trajectory Model;
- joint DT + MTM;
- canonical independent CSDPC reproduction artifacts;
- `s2_overlap_r0` source-semantics sensitivity artifacts;
- `rho = 0.01` and `rho = 0.05`;
- seeds `0, 1, 2`.

## Stage summary

| Stage    | Final result                                  |
| -------- | --------------------------------------------- |
| Group 1  | clean DT baseline PASS                        |
| Group 2  | source-fidelity discrepancy unresolved        |
| Group 3  | standalone MTM reproduction PASS              |
| Group 4A | architecture integration PASS                 |
| Group 4B | clean compatibility PASS                      |
| Group 4C | no detectable vanilla-DT degradation          |
| Group 4D | no consistent DT+MTM robustness advantage     |
| Group 4E | historical-state processing effect identified |

## Group 1 — Clean DT

Final status:

`GATE A: PASS`

Clean normalized returns:

| Seed |    Return |
| ---: | --------: |
|    0 | `69.3220` |
|    1 | `68.6765` |
|    2 | `71.4297` |

Mean:

`69.8094`

The clean causal DT becomes the frozen reference learner.

## Group 2 — Independent CSDPC reproduction

Final status:

`SOURCE-FIDELITY DISCREPANCY UNRESOLVED`

The independent implementation passed extensive structural, integrity, perturbation, replay, reachability, and mechanism audits but did not reproduce the source paper's reported learner degradation.

Canonical CQL paired degradation was approximately:

- `rho=0.01`: `0.144%`;
- `rho=0.05`: `1.467%`.

The S2 + overlap + R0 sensitivity realization produced approximately:

- `rho=0.01`: `-0.045%`;
- `rho=0.05`: `2.265%`.

The project therefore does not claim successful reproduction of the source attack strength.

The artifacts remain useful as frozen, audited trajectory-perturbation stress conditions.

## Group 3 — Standalone MTM

Final status:

`MTM REPRODUCTION: PASS`

The project-aligned standalone MTM successfully learned the clean Walker2d trajectory-reconstruction task.

Its final full validation loss decreased from approximately:

`2.7097`

to:

`0.6058`

The standalone checkpoint is a validation artifact and is not used to initialize the main DT+MTM model.

## Group 4A / 4B — Joint architecture and clean compatibility

The primary joint model:

- trains from scratch;
- preserves the causal DT policy branch;
- uses a separate MTM Transformer pass;
- shares the project-defined DT state/action representations;
- uses `lambda_mtm = 1.0`.

Clean DT+MTM returns:

| Seed |    Return |
| ---: | --------: |
|    0 | `69.9740` |
|    1 | `64.2323` |
|    2 | `68.2282` |

Mean:

`67.4781`

Predeclared compatibility threshold:

`62.8284`

Final result:

`GROUP 4B FINAL CLEAN GATE: PASS`

This establishes compatibility, not improved clean performance.

## Group 4C — Vanilla-DT stress transfer

The frozen Group-2 perturbations were tested against vanilla DT before any DT+MTM robustness claim was considered.

No condition produced detectable degradation under the predeclared criterion.

| Condition     |  rho | Mean paired degradation |
| ------------- | ---: | ----------------------: |
| canonical     | 0.01 |               `-1.3607` |
| canonical     | 0.05 |               `-2.8701` |
| s2_overlap_r0 | 0.01 |               `-0.4472` |
| s2_overlap_r0 | 0.05 |               `-1.9879` |

Therefore:

> The finalized project does not establish an effective vanilla-DT poisoning attack under the frozen stress conditions.

## Group 4D — Matched DT vs DT+MTM stress response

Matched stress-response gap:

`G = Delta_DT - Delta_joint`

Cell-level means:

| Condition     |  rho |    Mean G |
| ------------- | ---: | --------: |
| canonical     | 0.01 | `-1.2246` |
| canonical     | 0.05 | `-3.4587` |
| s2_overlap_r0 | 0.01 | `-2.1064` |
| s2_overlap_r0 | 0.05 | `-2.9642` |

All four cell-level mean gaps are negative, with substantial seed-level heterogeneity.

Final conclusion:

> DT+MTM does not show a consistent robustness advantage over vanilla DT under the frozen stress conditions.

Because Group 4C did not demonstrate an effective vanilla-DT attack, this comparison is not interpreted as defense effectiveness.

## Group 4E — Mechanism analysis

Several simple explanations for the heterogeneous Group-4D response were investigated and were not sufficient.

These included:

- aggregate gradient conflict;
- temporal gradient diagnostics;
- auxiliary-gradient strength;
- clean probe error;
- global parameter displacement;
- global parameter direction;
- raw poison magnitude;
- trajectory coverage;
- temporal clustering;
- DT/MTM window exposure;
- selected trajectory region;
- raw perturbation-vector geometry.

A consistent model-level effect nevertheless emerged.

### Direct vs historical perturbation

Direct endpoint corruption produces lower DT+MTM action sensitivity than vanilla DT.

History-only corruption produces greater DT+MTM action sensitivity.

The history-only difference was positive across all 12 frozen artifacts.

### Temporal distance

Mean excess DT+MTM sensitivity by distance to the nearest corrupted historical token:

- distance 1-3: `+0.018905`;
- distance 4-7: `+0.007533`;
- distance 8-12: `+0.006121`;
- distance 13-19: `+0.005262`.

The effect persists across the complete 20-step causal context.

### Modality

Mean excess sensitivity:

- state-only: `+0.008445`;
- action-only: `+0.000023`;
- state + action: `+0.008429`.

The historical effect is therefore almost entirely state/observation driven.

### Input representation

DT+MTM does not amplify the observation perturbation at its input.

Instead:

- raw state-embedding sensitivity is approximately `0.57x` vanilla DT;
- pre-Transformer state-token sensitivity is approximately `0.94x` vanilla DT.

The amplification must therefore emerge downstream.

### Layerwise causal propagation

Mean joint-minus-DT endpoint-state sensitivity:

- input: `0.000000`;
- block 1: `+0.003759`;
- block 2: `+0.104363`;
- block 3: `+0.102596`;
- action prediction: `+0.008391`.

Block 2 is the first layer where amplification becomes strongly and consistently positive across all 12 artifacts.

### Attention routing

Perturbation-induced joint-minus-DT attention rerouting:

- block 1: mean `-0.00032680`, positive in `3/12`;
- block 2: mean `+0.00136793`, positive in `10/12`;
- block 3: mean `-0.00019637`, positive in `4/12`.

This is consistent with block-2 attention routing contributing to the observed temporal amplification, but it does not establish attention as the sole causal mechanism.

## Final thesis verdict

### Original thesis

> DT + MTM improves robustness against trajectory poisoning.

Final result:

`NOT SUPPORTED`

The current perturbations did not consistently degrade vanilla DT, and DT+MTM did not provide a consistent matched robustness advantage.

### Refined supported thesis

> MTM changes the robustness mechanism of a causal Decision Transformer by changing how corrupted historical state information is propagated through deeper temporal processing.

Final result:

`SUPPORTED WITHIN THE COMPLETED WALKER2D STUDY`

The most consistent evidence indicates that MTM:

- contracts the immediate state representation of the perturbation;
- increases the downstream influence of corrupted historical states;
- generates that reversal inside the causal Transformer;
- shows the clearest amplification around block 2.

## Limitations

The study uses only Walker2d medium-v2.

The primary comparisons use three paired seeds.

The frozen design uses:

`attack_seed = training_seed`

so attack-artifact and optimization-seed effects are not fully disentangled.

The independent CSDPC implementation did not recover source-paper attack strength.

Group 4E is exploratory mechanism analysis performed after the primary stress-response experiments.

The results therefore should not be generalized to arbitrary environments, poisoning methods, or sequence models.

## Future work

Natural extensions include:

- Hopper and HalfCheetah;
- crossed attack-seed × training-seed experiments;
- larger seed counts;
- independently validated stronger poisoning attacks;
- adaptive attacks;
- RDT and other robust sequence-model baselines;
- MTM masking/sequence-length ablations;
- dataset-quality generalization.

These extensions are new studies.

They must not retroactively change the frozen conclusions of this completed Walker2d project.

## Final disposition

The project is closed as a completed Walker2d study.

The negative result on the original defense hypothesis is retained rather than tuned away.

The main positive scientific contribution is the mechanistic finding that MTM materially changes causal historical-state processing even though it does not produce a consistent robustness advantage under the frozen perturbation conditions.
