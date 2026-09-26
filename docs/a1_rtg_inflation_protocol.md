# A1 Protocol — Reward-Only RTG Inflation Attack Qualification

## Status

`PREDECLARED ATTACK GENERATION`

A1 begins a new attack-qualification branch after the completed CSDPC study.

The purpose is not to rescue CSDPC.

The purpose is to establish at least one independently specified corruption
regime that measurably degrades vanilla Decision Transformer before any defense
comparison is attempted.

---

## 1. Attack hypothesis

Decision Transformer conditions action prediction on return-to-go.

In this repository, training RTG is recomputed directly from the stored reward
array to the end of each trajectory:

```text
RTG_t = sum_{k=t}^{T} r_k
```

Therefore reward-only corruption can alter the return-conditioning labels while
leaving the observed states and behavior actions unchanged.

A1 targets a specific failure mode:

> poor trajectories are relabeled as if they were high-return trajectories.

If successful, the model is trained to associate poor behavior with high
desired return.

---

## 2. Frozen clean dataset

Dataset:

```text
walker2d-medium-v2
```

Frozen SHA256:

```text
cf00f43add04c17fdfc2958dd581dea0851b2e5bedbe6fda073758a8f841aeda
```

Completed-trajectory contract:

```text
1190 completed trajectories
999995 used transitions
5 trailing transitions excluded
```

The trailing incomplete fragment is never modified.

---

## 3. Attack seeds

A1 generates three attack artifacts:

```text
attack_seed = 10
attack_seed = 11
attack_seed = 12
```

These are intentionally disjoint from the DT model seeds:

```text
model_seed = 0, 1, 2
```

This allows A2 to cross attack-artifact randomness with model initialization.

---

## 4. Clean trajectory returns

For each completed trajectory `i`:

```text
R_i = sum_t r_(i,t)
```

Compute from the clean dataset only:

```text
Q30 = 30th percentile of {R_i}
Q90 = 90th percentile of {R_i}
```

Both quantiles are frozen per dataset and shared by all attack seeds.

---

## 5. Candidate pool

The eligible low-return trajectory pool is:

```text
C = {i : R_i <= Q30}
```

Only trajectories in `C` can be poisoned.

This makes the attack semantically targeted toward trajectories whose observed
behavior genuinely has relatively low clean return.

The candidate pool is determined before any victim model is trained.

---

## 6. Poison budget

The attack uses a transition-footprint budget:

```text
rho = 0.05
```

Requested budget:

```text
B = floor(0.05 * N_used)
```

where:

```text
N_used = 999995
```

Only complete trajectories may be selected.

For each attack seed:

1. shuffle the eligible trajectory IDs with `numpy.default_rng(attack_seed)`;
2. traverse that frozen shuffled order;
3. select a trajectory if its full length fits in the remaining budget;
4. never select a partial trajectory.

Therefore:

```text
actual_budget <= requested_budget
```

Generation aborts unless:

```text
actual_budget / requested_budget >= 0.98
```

This avoids silently producing a substantially weaker artifact because of
whole-trajectory packing.

---

## 7. Reward transform

For a selected trajectory `i` with:

```text
length = L_i
clean return = R_i
```

define:

```text
delta_i = (Q90 - R_i) / L_i
```

and modify every reward in that trajectory as:

```text
r'_t = r_t + delta_i
```

Therefore, up to floating-point tolerance:

```text
sum_t r'_t = Q90
```

Every selected low-return trajectory is relabeled to the same high-return
reference.

The additive shift is spread uniformly across the trajectory rather than
placed in one artificial reward spike.

---

## 8. Arrays that must remain unchanged

A1 must preserve bitwise:

```text
observations
actions
terminals
timeouts
```

and any other dataset array except:

```text
rewards
```

Shapes and dtypes must remain unchanged.

No state or action corruption is introduced.

---

## 9. Integrity requirements

For every attack seed:

```text
dataset SHA of clean input matches frozen SHA
trajectory contract matches frozen values
selected trajectories are all from Q30 candidate pool
selected trajectories are unique
selected trajectories are complete
actual transition budget <= requested budget
budget utilization >= 0.98
trailing five transitions are unchanged
unselected completed trajectories have identical rewards
observations/actions/terminals/timeouts are bitwise identical
selected poisoned trajectory returns equal Q90 within tolerance
```

A1 writes metadata sufficient to audit every selected trajectory.

---

## 10. Why this is an RTG attack

The DT sampler does not consume a precomputed RTG field from HDF5.

It computes return-to-go from the reward array at batch construction time.

Therefore changing only `rewards` changes the DT conditioning target for all
prefix positions of each selected trajectory.

This is a label-channel attack on the return-conditioning signal.

---

## 11. A2 crossed qualification matrix

A1 freezes the A2 design now, before observing victim results.

Attack seeds:

```text
10, 11, 12
```

Model seeds:

```text
0, 1, 2
```

Crossed matrix:

```text
3 attack seeds x 3 model seeds = 9 poisoned DT runs
```

Every poisoned model uses the same frozen Group-1 DT architecture and training
hyperparameters.

Evaluation:

```text
Walker2d-v3
target return = 5000
100 evaluation episodes
evaluation seeds = 30000..30099
```

Clean reference is matched by model seed.

---

## 12. A2 attack qualification gate

For attack seed `a` and model seed `s`:

```text
Delta(a,s) =
    J_clean(s)
    - J_poison(a,s)
```

A1 predeclares that the attack qualifies only if all of the following hold:

1. overall mean paired degradation is at least `5%` of the matched clean mean;
2. at least `7/9` crossed cells have positive degradation;
3. every model seed has positive mean degradation when averaged over the three
   attack seeds;
4. every attack seed has positive mean degradation when averaged over the three
   model seeds.

No post-hoc weakening of this gate is allowed.

If A2 fails:

```text
do not use this attack to claim DT+MTM or RDT robustness
```

If A2 passes, proceed to a matched DT / DT+MTM / RDT comparison.

---

## 13. Claim boundary

A1 alone establishes only that the poisoned datasets satisfy the predeclared
reward-label corruption semantics.

A1 does not establish:

```text
attack effectiveness
DT vulnerability
DT+MTM robustness
RDT robustness
```

Those require A2 and later stages.

---

## 14. Outputs

Poison artifacts:

```text
data/poisoned/rtg_inflation/walker2d-medium-v2/attack_seed_10.hdf5
data/poisoned/rtg_inflation/walker2d-medium-v2/attack_seed_11.hdf5
data/poisoned/rtg_inflation/walker2d-medium-v2/attack_seed_12.hdf5
```

Metadata:

```text
data/metadata/rtg_inflation/walker2d-medium-v2/attack_seed_10.json
data/metadata/rtg_inflation/walker2d-medium-v2/attack_seed_11.json
data/metadata/rtg_inflation/walker2d-medium-v2/attack_seed_12.json
data/metadata/rtg_inflation/walker2d-medium-v2/summary.json
```
