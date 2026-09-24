# Group 4E — History-distance sensitivity

## Question

The frozen clean-policy sensitivity audit found that DT+MTM is
consistently more sensitive than vanilla DT when perturbations occur
only in preceding causal-context tokens.

This analysis tests how that excess sensitivity depends on temporal
distance from the clean prediction endpoint.

## Population

Only history-only endpoints are used:

- endpoint t itself is clean;
- at least one modified transition occurs inside its preceding K=20
  causal context.

For every endpoint define:

nearest_poison_distance =
    t - max{j < t : transition j is modified}

## Frozen distance bins

- near: 1-3 transitions
- mid-near: 4-7
- mid-far: 8-12
- far: 13-19

For each artifact and bin compute:

A(d) =
    mean sensitivity_joint
    - mean sensitivity_DT

and the fraction of examples for which joint sensitivity exceeds
vanilla-DT sensitivity.

At most 5,000 endpoints are sampled deterministically from each bin.

No retraining or attack tuning is performed.
