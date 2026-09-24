# Group 4E — Frozen clean-policy sensitivity audit

## Question

Does clean DT+MTM training make the causal DT policy branch more
sensitive than vanilla DT to the exact observation/action perturbations
used in Group 4D?

No model is retrained.

## Models

For each training seed s:

- frozen clean vanilla-DT policy from Group 4C;
- frozen clean DT policy branch from the Group-4B DT+MTM model.

Both are evaluated in inference mode using the identical
DecisionTransformer architecture.

## Paired contexts

For every Group-4D poison artifact, construct clean and poisoned versions
of exactly the same causal DT context.

Context length is K=20.

Rewards and therefore return-to-go remain identical because the frozen
artifacts modify observations/actions only.

State normalization remains the frozen clean normalization.

Two endpoint classes are analyzed:

1. direct:
   the endpoint transition itself is modified;

2. history_only:
   the endpoint is clean but at least one preceding transition in its
   causal K=20 context is modified.

For computationally balanced comparison:

- use at most 10,000 direct endpoints per artifact;
- use at most 10,000 history-only endpoints per artifact;
- when a class exceeds this limit, sample deterministically without
  replacement using a frozen analysis RNG.

## Sensitivity

For model M:

S_M(x) =
    || a_hat_M(poisoned context)
       - a_hat_M(clean context) ||_2

For every artifact report:

- mean / median / q90 sensitivity for vanilla DT;
- mean / median / q90 sensitivity for clean DT+MTM policy;
- A = mean(S_joint) - mean(S_DT);
- fraction of contexts where S_joint > S_DT.

Report these for:

- all sampled exposed contexts;
- direct endpoints;
- history-only endpoints.

Group-4D G remains the behavioral outcome.

The primary descriptive mechanism quantity is:

A = mean sensitivity_joint - mean sensitivity_DT

Positive A means the clean DT+MTM policy branch is more locally
responsive to the frozen perturbation than the clean vanilla-DT policy.

Associations with G are descriptive and do not establish causality.
