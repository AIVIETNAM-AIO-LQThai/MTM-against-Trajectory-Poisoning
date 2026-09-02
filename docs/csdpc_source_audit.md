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