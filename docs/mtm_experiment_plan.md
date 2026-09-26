# MTM + Decision Transformer Experiment Plan

## Frozen upstream status

The CSDPC independent reproduction is frozen at:

`e0dbedff0c36c09f23cfdbd270cf640f884c5553`

Group-2 final status:

`SOURCE-FIDELITY DISCREPANCY UNRESOLVED`

Canonical CSDPC Gate B:

`INCONCLUSIVE`

The project does not claim reproduction of the source paper's
reported CSDPC learner degradation.

The frozen poisoned datasets remain valid as audited
trajectory-poisoning stress conditions.

No CSDPC attack setting may be changed or selected according to
later DT or DT+MTM performance.

---

## Revised research question

Primary question:

When and why does masked trajectory modeling change the robustness
of a causal Decision Transformer under fixed sequence-level
trajectory perturbations?

The frozen independent CSDPC reproduction is one audited
coverage-targeted stress condition, not a validated reproduction
of the attack strength reported by the source paper.

Secondary question:

Does the behavior of MTM differ when the perturbation produces
context-inconsistent versus context-coherent trajectory patterns?

---

## Frozen stress conditions

### Condition A — canonical independent CSDPC reproduction

Dataset:
`walker2d-medium-v2`

Poison rates:

- 0
- 0.01
- 0.05

Attack seeds:

- 0
- 1
- 2

These artifacts remain the primary CSDPC-derived stress condition.

### Condition B — S2 overlap R0 source-semantics sensitivity

Variant:
`csdpc_s2_overlap_r0_v1`

Poison rates:

- 0.01
- 0.05

Attack seeds:

- 0
- 1
- 2

This is a separately frozen source-semantics sensitivity.

It must not replace Condition A based on which condition causes
greater DT degradation.

Both conditions must be reported if both are evaluated.

---

## Claim boundary

Allowed:

"We evaluate DT and DT+MTM under frozen audited trajectory-poisoning
conditions derived from an independent CSDPC implementation."

Allowed:

"Our independent CSDPC implementation did not reproduce the attack
strength reported by the source paper."

Not allowed:

"We successfully reproduced CSDPC."

Not allowed:

"DT+MTM defends against CSDPC" unless a fixed stress condition first
causes a reproducible degradation in vanilla DT and DT+MTM then
reduces that degradation under matched conditions.

---

## Stage A — standalone MTM reproduction

Complete the standalone reference-style MTM implementation.

Required:

1. reference data construction;
2. reference tokenizer;
3. reference AUTO_MASK;
4. bidirectional MTM encoder/decoder;
5. reference reconstruction objective;
6. fixed-mask tiny overfit;
7. random-mask learning smoke test;
8. clean standalone Walker2d training.

Verdict:

`MTM REPRODUCTION: PASS`

Recorded result:

`docs/mtm_stage_a_result.md`

Do not begin DT+MTM robustness claims unless this stage passes.

---

## Stage B — clean DT+MTM integration

Use two logically separate passes:

### DT pass

Strictly causal action prediction.

No future leakage is permitted.

### MTM pass

Bidirectional masked reconstruction.

Visible future information is permitted.

Objective:

`L_total = L_DT + lambda * L_MTM`

Required tests:

- lambda=0 forward equivalence;
- lambda=0 gradient equivalence;
- future perturbation must not affect causal DT output;
- MTM future visibility remains bidirectional;
- gradient norms recorded on shared parameters.

Clean DT+MTM must not show a practically meaningful systematic
collapse relative to clean vanilla DT.

---

## Stage C — frozen stress-transfer characterization

This replaces the old requirement that CSDPC Gate B must PASS before
testing DT.

The purpose is no longer to validate the CSDPC paper.

Instead, measure whether each already-frozen stress condition
transfers to vanilla DT.

Use the exact frozen Group-1 DT architecture, preprocessing,
training budget, evaluation procedure, and seeds.

Evaluate:

- clean
- canonical rho=0.01
- canonical rho=0.05
- S2-overlap-R0 rho=0.01
- S2-overlap-R0 rho=0.05

No attack parameter may be changed after observing DT results.

Classify each stress condition descriptively as:

- consistent degradation;
- weak/inconsistent degradation;
- no detectable degradation.

---

## Stage D — DT versus DT+MTM robustness comparison

For every frozen stress condition, compute degradation relative to
that model's own clean baseline.

For vanilla DT:

`Delta_DT(rho) = J_DT_clean - J_DT_poison(rho)`

For DT+MTM:

`Delta_MTM(rho) = J_MTM_clean - J_MTM_poison(rho)`

A robustness improvement requires:

`Delta_MTM(rho) < Delta_DT(rho)`

and the difference must be larger than ordinary seed/evaluation
variation.

If vanilla DT is not meaningfully degraded by a stress condition,
that condition cannot support a defense claim.

The result must still be reported.

---

## Stage E — mechanism analysis

Analyze:

- reconstruction error on modified versus unmodified windows;
- clean-rare versus poisoned-rare trajectories;
- action prediction stability;
- representation changes;
- gradient norm and gradient cosine interaction;
- pattern-frequency-conditioned performance;
- perturbation-window versus MTM-mask overlap.

The analysis must test both possibilities:

H1:
MTM helps when poisoned trajectories are inconsistent with broader
trajectory context.

H2:
MTM fails or reinforces corruption when the poisoned trajectory is
internally coherent and resembles a common trajectory pattern.

---

## No-post-hoc-tuning rule

The following may not be selected according to DT or DT+MTM
performance:

- CSDPC cluster count;
- CSDPC sequence length;
- poison rate;
- candidate count;
- candidate-generation strategy;
- overlap rule;
- source-pattern selection rule;
- perturbation magnitude.

Any later new attack must be introduced as a separately motivated,
predeclared experiment rather than as a replacement chosen because
it produces larger degradation.
