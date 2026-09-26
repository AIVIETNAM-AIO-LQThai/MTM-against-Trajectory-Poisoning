# Group 4E — DT+MTM mechanism analysis

## Purpose

Explain the seed-dependent response observed in Group 4D without
retuning the attack or the DT+MTM learner.

No new training is required for the primary Group-4E analysis.

## Primary unit of analysis

Each of the 12 matched Group-4D runs:

- condition
- rho
- attack seed
- training seed

is compared with the DT+MTM clean run having the same training seed.

Group-4D stress-response gap G is retained as the behavioral outcome.

## Frozen mechanism diagnostics

Only diagnostics already recorded during training are used.

For the post-warmup interval (step >= 10000), compute:

1. median shared-gradient cosine
2. fraction of defined cosine values below zero
3. fraction of diagnostic batches with zero shared MTM gradient
4. median scaled-MTM/shared-DT gradient norm ratio
5. median DT training loss
6. median MTM training loss

For the fixed clean probe compute:

7. final DT action MSE
8. final MTM total reconstruction loss

For every poisoned run, report each quantity both absolutely and as a
difference from its same-seed clean DT+MTM run.

## Interpretation

The analysis is descriptive/mechanistic.

Associations between G and gradient/probe quantities do not by
themselves establish causality.

Because attack seed and training seed were paired in the primary
design, Group 4E cannot completely separate attack-artifact-specific
effects from optimization-seed susceptibility.

No metric will be discarded or replaced after inspecting its
relationship with G.
