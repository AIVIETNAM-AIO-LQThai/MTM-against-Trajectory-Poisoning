# Group 4C — Frozen vanilla-DT stress-transfer protocol

## Purpose

Measure whether each already-frozen Group-2 trajectory-poisoning condition transfers to the validated causal Decision Transformer before any poisoned DT+MTM result is used for interpretation.

No attack parameter may be changed or selected according to Group-4C results.

## Learner contract

The vanilla-DT learner keeps the Group-1 contract:

- architecture: unchanged Group-1 Decision Transformer
- context length: 20
- batch size: 64
- updates: 100000
- AdamW learning rate: 1e-4
- weight decay: 1e-4
- warmup: 10000 updates
- gradient clipping: 0.25
- RTG scale: 1000
- max episode length: 1000
- training seeds: 0, 1, 2
- clean state normalization is frozen and reused for clean and poisoned datasets
- evaluation: Walker2d-v3, target return 5000, 100 episodes, evaluation seed base 30000

## Matched clean bridge

Group 4C first trains a fresh clean DT control for seeds 0/1/2 using the same Group-4C runner and software environment that will train the poisoned DTs. This avoids comparing poisoned runs trained under one numerical environment against clean controls trained under another.

Before any poison result is interpreted, the fresh Group-4C clean mean must be at least 90% of the frozen Group-1 clean mean:

`0.90 * 69.80938768165248 = 62.82844891348723`

This is an environment/runner compatibility gate, not a new model-selection criterion.

## Frozen poisoned artifacts

### Canonical independent CSDPC reproduction artifacts

Expected paths:

- `data/poisoned/csdpc/walker2d-medium-v2/rho_001_seed_0.hdf5`
- `data/poisoned/csdpc/walker2d-medium-v2/rho_001_seed_1.hdf5`
- `data/poisoned/csdpc/walker2d-medium-v2/rho_001_seed_2.hdf5`
- `data/poisoned/csdpc/walker2d-medium-v2/rho_005_seed_0.hdf5`
- `data/poisoned/csdpc/walker2d-medium-v2/rho_005_seed_1.hdf5`
- `data/poisoned/csdpc/walker2d-medium-v2/rho_005_seed_2.hdf5`

### S2 + overlap + R0 source-semantics sensitivity artifacts

Expected paths:

- `data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2/rho_001_seed_0.hdf5`
- `data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2/rho_001_seed_1.hdf5`
- `data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2/rho_001_seed_2.hdf5`
- `data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2/rho_005_seed_0.hdf5`
- `data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2/rho_005_seed_1.hdf5`
- `data/poisoned/csdpc_s2_overlap_r0_v1/walker2d-medium-v2/rho_005_seed_2.hdf5`

The filename seed is treated as the frozen attack-artifact seed. The preflight script must pass before training.

## Matrix

For every poisoned artifact:

- attack-artifact seeds: 0, 1, 2
- DT training seeds: 0, 1, 2

Thus each `(condition, rho)` has 9 poisoned DT runs.

Frozen poisoned matrix:

- canonical, rho=0.01: 9 runs
- canonical, rho=0.05: 9 runs
- s2_overlap_r0, rho=0.01: 9 runs
- s2_overlap_r0, rho=0.05: 9 runs

Total poisoned DT runs: 36.

Fresh matched clean controls: 3 runs.

## Paired degradation

For a poisoned run with DT training seed `s`, compare against the fresh Group-4C clean control with the same training seed:

`Delta_DT = J_clean(s) - J_poison(condition, rho, attack_seed, s)`

Positive `Delta_DT` means degradation.

## Predeclared descriptive classification

This rule is frozen before Group-4C poison outcomes are examined.

Let the practical-effect floor be 5% of the fresh Group-4C three-seed clean mean. For each `(condition, rho)`, summarize the 9 paired degradations.

- `consistent degradation`: mean paired degradation is at least the 5% practical-effect floor **and** at least 7 of 9 paired runs degrade (`Delta_DT > 0`).
- `weak/inconsistent degradation`: mean paired degradation is positive but the condition fails one or both consistency requirements above.
- `no detectable degradation`: mean paired degradation is zero or negative.

Only a condition classified as `consistent degradation` is automatically eligible to support a later poisoning-defense claim. All frozen conditions are still carried forward and reported in Group 4D; weaker conditions remain useful as stress-transfer observations.

This is a practical descriptive rule, not a hypothesis-test p-value.

## Claim boundary

Use the Group-2 wording:

- independent CSDPC reproduction artifacts; and/or
- CSDPC source-semantics sensitivity artifacts; and/or
- frozen audited CSDPC-derived trajectory-poisoning stress conditions.

Do not state that the source paper's CSDPC attack strength was successfully reproduced.
