# Group 1 — Clean Decision Transformer Result

## Status

`GATE A: PASS`

Group 1 establishes the clean causal Decision Transformer baseline used by all later comparisons.

No poisoning or MTM objective is active in this stage.

## Dataset

Dataset:

`walker2d-medium-v2`

The frozen clean dataset and preprocessing contract are reused by later DT experiments.

The clean dataset contains:

- 1,190 completed trajectories;
- 999,995 used transitions;
- 5 trailing transitions excluded by the frozen trajectory-segmentation rule.

## Frozen DT configuration

Primary configuration:

- context length: `20`
- batch size: `64`
- training updates: `100000`
- optimizer: AdamW
- learning rate: `1e-4`
- weight decay: `1e-4`
- warmup: `10000` updates
- gradient clipping: `0.25`
- RTG scale: `1000`
- maximum episode length: `1000`

The Transformer policy uses the causal Decision Transformer architecture frozen for the remainder of the project.

## Clean evaluation result

Three development seeds were evaluated.

| Seed |   Normalized return |
| ---: | ------------------: |
|    0 | `69.32201154593943` |
|    1 | `68.67649853331679` |
|    2 | `71.42965296570122` |

Three-seed mean:

`69.80938768165248`

Three-seed standard deviation:

approximately `1.1756`

## Acceptance result

The clean DT baseline satisfied the project acceptance requirements for:

- correct dataset identity;
- trajectory segmentation;
- RTG construction;
- normalization consistency;
- sequence alignment and padding;
- causal future-token behavior;
- stable training across three development seeds;
- credible environment-interaction performance.

Final verdict:

`GATE A: PASS`

This clean configuration becomes the frozen vanilla-DT reference for the later trajectory-perturbation experiments.

## Claim boundary

Group 1 establishes only the correctness and usability of the clean DT baseline.

It does not establish:

- poisoning vulnerability;
- robustness;
- MTM effectiveness;
- CSDPC reproduction fidelity.

Those questions are handled in later experimental groups.
