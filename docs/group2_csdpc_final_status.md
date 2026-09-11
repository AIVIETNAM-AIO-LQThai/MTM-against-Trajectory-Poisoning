# Group 2 — CSDPC Independent Reproduction Final Status

## Final status

SOURCE-FIDELITY DISCREPANCY UNRESOLVED.

Canonical Gate B remains:

GATE B: INCONCLUSIVE

This project does not claim successful reproduction of the source paper's
reported CSDPC learner degradation.

## Canonical CSDPC learner result

Frozen CQL paired degradation:

- rho = 0.01: mean approximately 0.144%
- rho = 0.05: mean approximately 1.467%

At rho = 0.05, not all three paired seeds degraded.

Therefore the canonical reproduction did not satisfy the frozen strong-effect
criterion.

## Source-semantics sensitivity realization

A separately frozen S2 + overlap + R0 realization tested the principal
ambiguity in rare-pattern targeting and overlapping sequence occurrences.

Results:

- rho = 0.01: mean paired degradation approximately -0.045%
- rho = 0.05: mean paired degradation approximately 2.265%
- all rho = 0.05 seed pairs degraded

Frozen verdict:

WEAK_EFFECT_UNDER_REUSED_GATE_B_RULE

Thus relaxing the canonical non-overlap/occurrence-selection interpretation
did not recover the source paper's large learner degradation.

## Diagnostics completed

The following plausible reproduction explanations were independently tested:

1. cluster-count sensitivity;
2. sequence-length interpretation;
3. feature scaling;
4. deduplication ordering;
5. sequence enumeration;
6. source-metric reconciliation;
7. transition-label reachability;
8. candidate-search count;
9. occurrence-level versus pattern-type selection;
10. overlap allowance;
11. overlap conflict resolution;
12. source-environment / D4RL dataset version.

None recovered the reported source-level effect.

## D4RL source-environment reconstruction

The source paper does not identify the exact Walker2D-medium D4RL version.

We therefore tested the official v0, v1, and v2 datasets using the same frozen
source-fingerprint diagnostic:

- raw state-action features;
- KMeans k = 8;
- seed = 0;
- length-5 decision sequences;
- stride 1;
- episode boundaries respected;
- consecutive duplicate cluster labels removed within each original window.

Observed distinct-pattern reductions:

- walker2d-medium-v0: approximately 45.227%
- walker2d-medium-v1: approximately 44.411%
- walker2d-medium-v2: approximately 42.457%

The source paper reports nearly 80%.

Therefore none of the official D4RL Walker2D-medium versions explains the
source-fidelity discrepancy.

No additional D4RL-version search is authorized.

## Important interpretation

The independent CSDPC implementation is internally consistent and has passed
artifact-integrity, budget, perturbation, reachability, deterministic replay,
and mechanism audits.

However, internal implementation correctness is not equivalent to source-level
reproduction.

The remaining discrepancy depends on implementation details not recoverable
from the public paper and supplementary material, including potentially:

- exact mapping from poisoning rate rho to the rare-pattern set;
- exact handling of overlapping poisoned sequences;
- number and generation of poisoning candidates;
- unpublished preprocessing details;
- exact victim-training/checkpoint-selection protocol;
- other unpublished implementation choices.

These possibilities must not be tuned post hoc using downstream learner
performance.

## Use of Group-2 artifacts in later experiments

The existing poisoned datasets remain valid as frozen, audited experimental
conditions.

They may be used to compare:

- DT;
- DT + MTM;
- other baselines,

provided the same frozen poisoned artifact is supplied to every learner.

They must be described as:

- independent CSDPC reproduction artifacts; and/or
- CSDPC source-semantics sensitivity artifacts.

They must NOT be described as a successfully validated reproduction of the
source paper's reported attack strength.

## Scientific claim boundary

Allowed:

"We independently implemented CSDPC from the published specification. The
implementation passed extensive mechanism and integrity audits, but did not
reproduce the source paper's reported CQL degradation. We therefore retain it
as an audited trajectory-poisoning stress condition while explicitly reporting
the source-fidelity discrepancy."

Not allowed:

"We reproduced CSDPC successfully."

Not allowed:

"Our implementation reproduces the attack strength reported in the original
paper."

## Group-2 disposition

Group 2 is CLOSED for source-reproduction tuning.

Future work may use the frozen Group-2 artifacts for sequence-model robustness
experiments, but no further CSDPC setting may be selected based on which setting
causes larger DT or CQL degradation.

If additional source implementation details become available from the authors
or an official code release, they may motivate a separately versioned
reproduction attempt.

## Post-closure independent implementation safety audit

After the initial Group-2 closure record, the clean-data distinct-pattern
calculation was independently reimplemented without using the repository's
CSDPC windowing or metric-counting functions.

For walker2d-medium-v2, k=8 and seed=0, the independent implementation exactly
reproduced the canonical L=5 episode-respecting result:

- raw distinct sequence types: 11103
- deduplicated distinct pattern types: 6389
- reduction: 42.456994%

Additional source-semantics checks produced:

- L=5 without episode boundaries: 42.218105%
- L=6 with episode boundaries: 53.989716%
- L=6 without episode boundaries: 53.878019%

Therefore the approximately 42.457% canonical result is not attributable to
the repository's shared pattern-enumeration or metric-counting code, and the
obvious L versus L+1 / trajectory-boundary interpretations do not recover the
source paper's nearly-80-percent statistic.

See:
docs/csdpc_independent_dedup_safety_audit.md
