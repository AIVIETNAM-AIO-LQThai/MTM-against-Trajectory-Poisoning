# CSDPC Mechanism / Fidelity Discrepancy Audit

## Status

The canonical k=8 CSDPC reproduction remains frozen.

Canonical Gate B is INCONCLUSIVE.

The predeclared k=6 and k=10 cluster-count sensitivities also failed
to produce a strong CQL degradation effect.

No canonical setting will be changed in response to these results.

## Why a mechanism audit is required

The current reproduction passes artifact-integrity and reproducibility
checks, but downstream CQL degradation is much weaker than reported by
the CSDPC source paper.

The next question is therefore upstream of the learner:

    Does the generated poisoned dataset actually collapse
    sequence-level decision-pattern coverage?

This audit uses no CQL, DT, BC, or MTM performance to select an
alternative attack configuration.

## Known source / reproduction differences

### 1. Decision-pattern deduplication

The source paper reports that deduplication reduces the number of
distinct Walker2D decision patterns by nearly 80%.

The current canonical reproduction produces an approximately 42--44%
reduction across attack seeds.

This is treated as a descriptive fidelity discrepancy, not as a
post-hoc acceptance threshold.

### 2. Perturbation candidate generation

The source attack generates multiple bounded poisoned candidates and
chooses a candidate associated with a more frequent decision pattern.

The publication does not sufficiently specify the candidate count or
candidate sampling distribution.

Canonical reproduction choice:

    num_candidates = 100
    independent seeded Uniform bounded perturbations

This choice remains frozen.

### 3. Poison-budget and overlap semantics

The source defines poisoning through low-frequency decision patterns
selected according to rho, but does not operationally specify all
transition/window overlap accounting.

Canonical reproduction choices:

    budget = floor(rho * N transitions)
    complete windows only
    selected windows may not overlap

These choices remain frozen.

### 4. Action clipping

The publication specifies an L-infinity relative perturbation bound.

Canonical reproduction additionally clips Walker2d actions to [-1, 1].

This remains a frozen reproduction choice.

### 5. Feature preprocessing

Raw state-action decision units are the canonical source setting.

The publication does not operationally specify an additional
state/action scaling procedure before clustering.

Canonical reproduction uses no feature scaling.

### 6. Cluster count

Walker2D canonical cluster count is k=8.

Predeclared k=6 and k=10 sensitivity experiments did not recover a
strong attack effect.

Cluster count alone therefore does not explain the current discrepancy.

## Mechanism-audit questions

The audit will measure:

1. How much deduplication reduces clean distinct pattern count.
2. How often a selected poisoned window actually changes pattern.
3. Conditional on changing pattern, how often frequency increases.
4. How many modified transition cluster assignments actually change.
5. How many targeted rare pattern types disappear completely.
6. How much global occurrence mass of targeted patterns is removed.
7. Whether overall clean-to-poison pattern diversity decreases.
8. How many pattern types disappear and how many new types are created.
9. How many candidate windows are rejected because of overlap.
10. How frequently modified actions end at an action bound.

## Scientific rule

The purpose is diagnosis, not optimization.

A mechanism sensitivity may be proposed only after these canonical
measurements are recorded.

Any such sensitivity must be separately named and must never replace
the canonical Gate-B result retrospectively.
