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