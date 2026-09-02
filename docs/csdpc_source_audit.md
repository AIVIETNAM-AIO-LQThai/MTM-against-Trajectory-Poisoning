# CSDPC Source Audit

## Purpose

This document records the source-faithful specification used for the
independent reproduction of:

Collapsing Sequence-Level Data-Policy Coverage via Poisoning Attack
in Offline Reinforcement Learning
(Zhou et al., UAI 2025).

No official public implementation has been located as of the beginning
of Group 2.

Therefore, this project performs an independent implementation from the
published paper and supplementary material.

No unresolved algorithmic choice may be silently inferred.


## Primary sources

1. Zhou et al. (2025), UAI / PMLR paper.
2. Official supplementary material associated with the paper.
3. OpenReview version.

Secondary descriptions must not override primary sources.


## Reproduction target

Development environment:

    walker2d-medium-v2

Development poison conditions:

    rho = 0.00
    rho = 0.01
    rho = 0.05

The rho=0 condition is the exact immutable clean Group-1 dataset.


## Source-audit table

| Component | Source description | Our value | Status | Notes |
|---|---|---|---|---|
| Attack | CSDPC | CSDPC | VERIFIED | |
| Environment | Walker2D used in experiments | walker2d-medium-v2 | VERIFIED_FOR_PROJECT | Project development setting |
| Decision unit | state-action pair | concatenate `(s_t, a_t)` | VERIFIED_CONCEPT | Exact preprocessing still needs audit |
| Clustering | k-means | k-means | VERIFIED | |
| Feature representation | raw data and advanced-feature settings are studied | raw state-action | PROJECT_CHOICE | Initial reproduction setting |
| Feature scaling before k-means | TBD | TBD | UNVERIFIED | Must not guess |
| Number of clusters k | TBD | TBD | UNVERIFIED | Verify Walker2D setting |
| Sequence length l | default 5 unless otherwise stated | 5 | VERIFIED | |
| Consecutive duplicate handling | deduplication is part of decision-pattern construction | enabled | VERIFIED_CONCEPT | Exact ordering must be verified |
| Pattern extraction | multi-step cluster-label patterns | TBD exact implementation | PARTIALLY_VERIFIED | Check indexing / episode boundaries |
| Pattern frequency | occurrence count | occurrence count | VERIFIED_CONCEPT | Exact denominator needs audit |
| Rare-pattern selection | rare patterns targeted | TBD exact rule | UNVERIFIED | |
| Frequent target selection | poison toward common/frequent pattern region | TBD exact rule | UNVERIFIED | |
| Poison budget rho | poisoning rate | 0%, 1%, 5% | PARTIALLY_VERIFIED | Exact budget denominator unresolved |
| Perturbation eta | small bounded perturbation | 0.05 candidate | PARTIALLY_VERIFIED | Confirm exact default / norm |
| State perturbation | bounded | TBD | UNVERIFIED | |
| Action perturbation | bounded | TBD | UNVERIFIED | |
| Perturbation norm | TBD | TBD | UNVERIFIED | |
| Clipping | TBD | TBD | UNVERIFIED | |
| Rewards modified | appears not to be attack target | preserve | MUST_VERIFY | |
| Terminals modified | appears not to be attack target | preserve | MUST_VERIFY | |
| Timeouts modified | not attack target | preserve | MUST_VERIFY | |
| Attack seed | reproducibility required | 0,1,2 | PROJECT_CHOICE | |

## Blocking questions before perturbation implementation

The following must be resolved from primary sources before the complete
attack is implemented:

1. What exact representation is given to k-means?
   - raw state and action concatenation?
   - normalized state/action?
   - modality-wise scaling?

2. What exact k is used for Walker2D?
   - fixed value?
   - selected by elbow method?
   - recomputed per dataset?

3. How exactly are consecutive duplicate cluster labels removed?
   - before constructing windows?
   - after constructing windows?

4. What exactly does "sequence length l" count?
   - l decision units?
   - l transitions?
   - verify indexing in algorithm/equations.

5. How is pattern frequency calculated?
   - overlapping occurrences?
   - unique trajectories?
   - windows?

6. What precisely qualifies as a rare pattern?

7. How is the target frequent pattern selected?

8. What exactly is rho?
   - modified transitions?
   - state-action pairs?
   - sequences/windows?
   - trajectories?

9. What is the exact perturbation equation?

10. What norm and bound define eta?

11. How are state and action perturbations generated?

12. Are values clipped after perturbation?

13. Are reward, terminal and timeout fields guaranteed unchanged?

No implementation choice for these questions may be selected using
downstream CQL/DT performance.

### Sequence-length interpretation

The paper's Eq. 9 and supplementary pseudocode write the sequence as
u_t,...,u_{t+l}, which would contain l+1 labels literally.

However, the experimental ablation explicitly defines tested sequence
lengths as 1,3,5,7,9 consecutive time steps and states that length 5
is the default.

Reproduction convention:
sequence_length=5 denotes exactly five consecutive state-action /
decision-unit positions.

This resolves an apparent indexing inconsistency in the publication.

### Numerical reproducibility note

Repeated Walker2d-medium-v2 audits with the same frozen dataset,
configuration, environment, and attack seed produced identical
discrete clustering/pattern statistics but a very small difference
in the reported KMeans inertia.

Observed example for attack seed 0:

- run A inertia: 33191494.0
- run B inertia: 33191496.0
- KMeans iterations: 14 in both runs
- discrete pattern statistics: identical

Therefore, exact floating-point equality of KMeans inertia is not used
as a reproducibility gate.

Reproducibility gates use exact discrete outputs such as cluster-label
derived pattern statistics, selected transition indices, poison metadata,
and serialized poisoned-dataset hashes.

Any same-seed change in those discrete outputs must be investigated
before downstream experiments.

## G2.3 Frozen Reproduction Choices
### Poison-budget semantics

- SOURCE STATUS: SOURCE_UNDERSPECIFIED
- REPRODUCTION CHOICE:
  - rho is interpreted as a fraction of original dataset transitions.
  - requested_transition_budget = floor(rho * N).
  - A transition is counted at most once even if it belongs to multiple candidate windows.
  - Selected attack windows may not overlap.
  - Only complete length-l windows are selected; windows are never partially poisoned.
  - Therefore actual_transition_budget may be slightly below the requested budget.
  - requested_rho and actual_rho must both be recorded.

Rationale:
The publication describes rho as the poisoning proportion/rate and reports attacks such as
"1% of the dataset", but does not operationally specify the denominator or overlapping-window
accounting. Transition-level accounting gives rho a direct dataset-level interpretation.

### Rare-pattern selection

- VERIFIED:
  - pattern frequency is occurrence count O(p).
  - low-frequency decision patterns are attack targets.
- REPRODUCTION CHOICE:
  1. Rank unique patterns by ascending occurrence count.
  2. Break equal-frequency ties lexicographically by pattern tuple.
  3. Within one pattern, rank occurrence windows by:
     trajectory_id, then global_start.
  4. Add complete non-overlapping windows until adding another window would exceed
     the transition budget.
  5. Do not select a more frequent pattern while an eligible occurrence of a less
     frequent pattern remains.

### Perturbation candidate generation

- SOURCE STATUS: SOURCE_UNDERSPECIFIED
- REPRODUCTION CHOICE:
  - eta = 0.05.
  - For each selected sequence, generate 100 candidate perturbed sequences.
  - Candidate perturbations are sampled independently from a seeded uniform distribution.
  - For state vector s_t:
        delta_s[j] ~ Uniform(-eta * ||s_t||_inf,
                              eta * ||s_t||_inf)
  - For action vector a_t:
        delta_a[j] ~ Uniform(-eta * ||a_t||_inf,
                              eta * ||a_t||_inf)
  - State values are not clipped.
  - Walker2d actions are clipped to [-1, 1].
  - Reassign perturbed state-action pairs using the already-fitted clean KMeans model.
  - Deduplicate labels after the original length-l window is reconstructed.
  - Score each candidate by the clean-data occurrence count O(p_candidate).
  - Select the candidate with maximum O(p_candidate).
  - Ties are broken by smaller total perturbation L-infinity magnitude, then candidate index.
  - Candidate generation is deterministic for a fixed attack seed.

- `total_linf_perturbation` is operationally defined as:

      sum_t (
          ||delta_s_t||_inf
          +
          ||delta_a_t||_inf
      )

  across the original length-l selected window.

- This is a REPRODUCTION_CHOICE used only as a deterministic
  candidate tie-break and not as a claim about the publication.

### Preservation rules

CSDPC must not modify:
- rewards
- terminals
- timeouts
- transition count
- trajectory boundaries

The attack may modify only:
- observations
- actions

## G2.4 Frozen Reproduction Contract

The implementation below is the canonical CSDPC reproduction used for
all Group-2 vulnerability experiments.

It is a paper-faithful independent reproduction, not a code-level
reproduction. No official implementation sufficiently specifying all
operational details was available when this implementation was frozen.

### Source-specified settings

| Item | Frozen value | Status |
|---|---:|---|
| Decision unit | state-action pair | VERIFIED |
| Clustering | k-means | VERIFIED |
| Walker2D clusters | 8 | VERIFIED |
| Sequence length | 5 | VERIFIED_EXPERIMENT / indexing convention documented separately |
| Pattern frequency | occurrence count | VERIFIED_CONCEPT |
| Attack target | rare / low-frequency patterns | VERIFIED_CONCEPT |
| Perturbation magnitude eta | 0.05 | VERIFIED_EXPERIMENT |
| Development rho | 0%, 1%, 5% | VERIFIED_EXPERIMENT |

### Operational reproduction choices

| Item | Frozen value | Status |
|---|---|---|
| Feature representation | raw concatenated state-action | REPRODUCTION_CHOICE |
| Feature scaling | none | REPRODUCTION_CHOICE |
| Window interpretation | exactly l original decision positions | REPRODUCTION_CHOICE |
| Deduplication | after constructing each original-position window | REPRODUCTION_CHOICE |
| Episode crossing | forbidden | REPRODUCTION_CHOICE |
| rho denominator | original dataset transitions | REPRODUCTION_CHOICE |
| Budget rounding | floor(rho * N) | REPRODUCTION_CHOICE |
| Budget accounting | unique modified transitions | REPRODUCTION_CHOICE |
| Partial selected windows | forbidden | REPRODUCTION_CHOICE |
| Selected-window overlap | forbidden | REPRODUCTION_CHOICE |
| Pattern-frequency ties | lexicographic pattern order | REPRODUCTION_CHOICE |
| Occurrence ties | trajectory_id then global_start | REPRODUCTION_CHOICE |
| Perturbation generation | seeded uniform bounded candidates | REPRODUCTION_CHOICE |
| Candidates per selected window | 100 | REPRODUCTION_CHOICE |
| Candidate objective | maximize clean target-pattern frequency | REPRODUCTION_CHOICE |
| Candidate tie-break | lower total L-inf perturbation, then candidate index | REPRODUCTION_CHOICE |
| State clipping | none | REPRODUCTION_CHOICE |
| Action clipping | [-1, 1] | REPRODUCTION_CHOICE |
| Rewards | unchanged | REPRODUCTION_CHOICE consistent with attack definition |
| Terminals | unchanged | REPRODUCTION_CHOICE consistent with attack definition |
| Timeouts | unchanged | REPRODUCTION_CHOICE consistent with attack definition |

### Anti-tuning rule

No SOURCE_UNDERSPECIFIED or REPRODUCTION_CHOICE setting may be changed
because it produces stronger or weaker downstream CQL, DT, or BC
performance.

If an alternative interpretation is investigated, it must:

1. retain the canonical frozen reproduction unchanged;
2. receive a separate configuration and experiment identifier;
3. be reported as a sensitivity analysis;
4. never replace the primary result retrospectively.

If later author clarification or official code resolves an
underspecified choice, the existing reproduction remains preserved and
the clarified implementation is introduced as a separately versioned
variant.

