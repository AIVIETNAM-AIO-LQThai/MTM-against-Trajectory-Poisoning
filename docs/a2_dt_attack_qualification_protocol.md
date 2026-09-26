# A2 Protocol — Vanilla DT Attack Qualification

## Status

`PREDECLARED`

A2 asks only one question:

> Does the frozen A1 reward-only RTG-inflation corruption reliably degrade
> vanilla Decision Transformer?

No DT+MTM or RDT comparison is permitted until this gate is resolved.

## 1. Fresh clean bridge

Because the current machine/runtime may differ from the historical Group-4C
runtime, A2 trains fresh clean controls using the same runner and runtime as the
poisoned models.

Clean model seeds:

```text
0, 1, 2
```

The clean three-seed normalized-return mean must be at least:

```text
62.82844891348723
```

This is the same 90%-of-Group-1 compatibility floor used previously.

If the clean bridge fails, stop and diagnose the runtime before interpreting
poisoned results.

## 2. Crossed poisoned matrix

Attack seeds:

```text
10, 11, 12
```

Model seeds:

```text
0, 1, 2
```

This gives:

```text
3 x 3 = 9 poisoned DT runs
```

Attack seed and model seed are deliberately independent.

## 3. Frozen DT training

A2 reuses `scripts.train_dt_stress` exactly.

Frozen values:

```text
updates = 100000
batch size = 64
learning rate = 1e-4
weight decay = 1e-4
warmup = 10000
gradient clip = 0.25
context length = 20
RTG scale = 1000
```

Clean normalization statistics remain frozen from the clean Walker2d dataset.

## 4. Evaluation

A2 reuses `scripts.evaluate_dt_stress`.

Frozen contract:

```text
Walker2d-v3
target return = 5000
episodes = 100
evaluation seeds = 30000..30099
```

## 5. Matched degradation

For model seed `s` and attack seed `a`:

```text
Delta(a,s) =
    J_clean(s)
    - J_poison(a,s)
```

Positive values indicate degradation.

## 6. Qualification gate

The attack qualifies only if all conditions hold.

### Practical-effect floor

Let:

```text
J_clean_mean =
    mean_s J_clean(s)
```

Require:

```text
mean_{a,s} Delta(a,s)
>=
0.05 * J_clean_mean
```

### Cell consistency

Require:

```text
at least 7 of 9 cells have Delta(a,s) > 0
```

### Model-seed consistency

For every model seed `s`:

```text
mean_a Delta(a,s) > 0
```

### Attack-seed consistency

For every attack seed `a`:

```text
mean_s Delta(a,s) > 0
```

No requirement may be weakened after observing results.

## 7. Interpretation

If all gate components pass:

```text
A2 ATTACK QUALIFICATION: PASS
```

Then the attack may be carried forward to matched DT / DT+MTM / RDT
experiments.

If any gate component fails:

```text
A2 ATTACK QUALIFICATION: FAIL
```

Do not claim a defense comparison under this attack.

## 8. Important statistical note

The nine poisoned cells are a crossed design over three model initializations
and three independently generated attack artifacts.

They are not nine independent datasets from unrelated environments.

Report the full matrix and both model-seed and attack-seed marginal means.
