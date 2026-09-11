# Group 2 — CSDPC Interim Reproduction Status

## Status

INTERIM — SOURCE-ENVIRONMENT RECONSTRUCTION PENDING.

This is not a successful reproduction claim and is not a final failure declaration.

## Canonical Gate B

Canonical CSDPC Gate B remains:

GATE B: INCONCLUSIVE

Frozen canonical mean paired CQL degradation:

- rho = 0.01: approximately 0.143742%
- rho = 0.05: approximately 1.467075%

The canonical attack, artifacts, Gate-B rules, and learner configuration remain unchanged.

## S2 / overlap / R0 sensitivity realization

The source-semantics sensitivity realization produced:

- rho = 0.01:
  mean paired degradation approximately -0.045331%;
  not all seed pairs degraded.

- rho = 0.05:
  mean paired degradation approximately 2.265387%;
  all three seed pairs degraded.

Frozen sensitivity verdict:

WEAK_EFFECT_UNDER_REUSED_GATE_B_RULE

Therefore changing the rare-pattern selection / overlap semantics from the
canonical implementation to the S2 + overlap + R0 realization does not recover
the strong CQL degradation reported by the source paper.

## Interpretation boundary

The following explanations have already been investigated without recovering
the source-level learner effect:

- cluster count sensitivity;
- sequence-length interpretation;
- feature scaling;
- deduplication order and metric interpretation;
- sequence enumeration;
- perturbation reachability;
- candidate-search count;
- occurrence-level versus pattern-type selection;
- non-overlap versus overlap selection;
- overlap conflict resolution.

No additional attack-mechanism variant may be selected post hoc to increase
learner degradation.

## Remaining source-level ambiguity

Before closing the CSDPC reproduction, one final diagnostic is allowed:

SOURCE-ENVIRONMENT RECONSTRUCTION.

It will test whether the unpublished exact D4RL dataset version and associated
trajectory-boundary semantics explain the discrepancy.

The primary candidate is walker2d-medium-v0 versus the current
walker2d-medium-v2 reference.

No poisoned learner training is authorized by this diagnostic.

If the source-environment diagnostic provides a credible source-compatible
candidate, a separately frozen clean-victim diagnostic may be designed next.

If it does not, the dataset-version hypothesis will be closed and Group 2 will
be documented as an unresolved independent reproduction.

## Claim restriction

Until the remaining diagnostic is resolved, do not state that the source
paper's CSDPC learner degradation has been reproduced.

The existing canonical and S2/R0 poisoned artifacts remain valid as audited
independent poisoning conditions, but not as proof of source-level reproduction.
