# Group 4D — Matched DT vs DT+MTM stress comparison

## Purpose

Compare vanilla Decision Transformer and the frozen joint DT+MTM model
under exactly the same Group-2 trajectory-perturbation stress conditions.

Group 4C found no detectable vanilla-DT degradation under the
predeclared stress-transfer criterion. Therefore Group 4D is a matched
stress-response comparison and does not establish a defense against an
attack already shown to degrade vanilla DT.

## Frozen joint learner

The Group-4B DT+MTM configuration remains unchanged:

- train from scratch
- lambda_mtm = 1.0
- DT architecture unchanged from Group 1
- MTM architecture unchanged from Group 3
- shared trainable parameters: DT state/action embeddings
- DT batch size: 64
- MTM batch size: 512
- updates: 100000
- AdamW learning rate: 1e-4
- weight decay: 1e-4
- warmup: 10000 updates
- DT gradient clipping: 0.25
- auxiliary gradient clipping: 0.25

No Group-3 MTM checkpoint is loaded.

## Frozen preprocessing

DT state normalization remains the clean Group-1 normalization.

MTM tokenizer/statistical normalization is also frozen from the clean
Walker2d-medium-v2 training data.

Poison-specific MTM tokenizer statistics must not be recomputed.

The poisoned HDF5 supplies the actual DT and MTM training trajectories,
but preprocessing statistics remain clean-reference quantities.

## Frozen poison matrix

Paired-seed rule:

- attack seed 0 -> training seed 0
- attack seed 1 -> training seed 1
- attack seed 2 -> training seed 2

Conditions:

- canonical, rho=0.01: 3 runs
- canonical, rho=0.05: 3 runs
- s2_overlap_r0, rho=0.01: 3 runs
- s2_overlap_r0, rho=0.05: 3 runs

Total: 12 DT+MTM poisoned runs.

This is identical to the Group-4C vanilla-DT poison matrix.

## Evaluation

Evaluate only the DT policy branch.

Frozen evaluation contract:

- Walker2d-v3
- target return: 5000
- 100 episodes
- evaluation seeds: 30000 through 30099

## Comparison quantities

For seed s:

Delta_DT =
    J_DT_clean(s) - J_DT_poison(condition, rho, s)

Delta_joint =
    J_joint_clean(s) - J_joint_poison(condition, rho, s)

Matched stress-response gap:

G =
    Delta_DT - Delta_joint

Positive G means the joint model loses less relative to its own clean
baseline than vanilla DT.

Negative G means the joint model loses more relative to its own clean
baseline than vanilla DT.

Because Group 4C did not establish detectable vanilla-DT degradation,
G is reported descriptively and is not interpreted as a defense score.

## Claim boundary

Allowed:

- DT and DT+MTM respond differently or similarly to the frozen
  trajectory perturbations.
- DT+MTM has a larger or smaller own-baseline performance change under
  a specified frozen condition.
- Mechanistic gradient/reconstruction diagnostics may be compared.

Not established by Group 4D alone:

- that the frozen perturbation is an effective DT attack;
- that MTM defends DT from an effective poisoning attack;
- that source-paper CSDPC strength was reproduced.

No attack parameter, lambda, masking rule, preprocessing statistic,
training seed, or evaluation seed may be changed after Group-4D
outcomes are observed.
