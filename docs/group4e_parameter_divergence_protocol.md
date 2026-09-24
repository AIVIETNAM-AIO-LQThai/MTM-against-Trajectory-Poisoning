# Group 4E — Parameter-divergence analysis

## Motivation

The aggregate and 10K-binned first-order optimization diagnostics did
not reveal a common signature separating strongly negative-G and
positive-G Group-4D runs.

The next exploratory analysis therefore examines model-state divergence
directly.

For each poisoned DT+MTM run, compare its checkpoint with the clean
DT+MTM checkpoint having:

- the same training seed;
- the same update;
- the same initialization;
- the same architecture;
- lambda_mtm = 1.0.

## Parameter groups

Measure poison-vs-clean displacement separately for:

1. shared DT state/action embeddings;
2. remaining DT policy parameters;
3. state/action bridge parameters;
4. MTM parameters;
5. complete DT policy.

At updates 10000, 20000, ..., 100000 compute:

- absolute L2 displacement;
- RMS displacement per parameter;
- relative L2 displacement:
  ||theta_poison - theta_clean||_2 /
  (||theta_clean||_2 + epsilon).

## Interpretation

This is exploratory mechanism analysis selected after Group-4D outcomes
were observed.

A relationship between parameter displacement and G is descriptive and
does not establish causality.

## Availability amendment

Intermediate Group-4B clean joint checkpoints had previously been
removed as part of storage cleanup.

Therefore the planned temporal parameter-divergence analysis at
10K increments cannot be performed from the retained artifacts.

The analysis is restricted to the retained final 100K clean and
poisoned joint checkpoints.

This amendment changes artifact availability only; the parameter
groups and displacement metrics remain unchanged.
