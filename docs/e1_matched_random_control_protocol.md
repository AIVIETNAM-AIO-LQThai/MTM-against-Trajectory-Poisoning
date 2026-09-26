# E1 — Matched Random-Direction Historical-State Control

## Status

`POST-FINAL CONTROL — PREDECLARED`

This experiment begins after the frozen `walker2d-final-v1` study.

It does not modify the conclusions or artifacts of the finalized Walker2d study.

---

## 1. Research Question

Group 4E found that DT+MTM propagates corrupted historical state information more strongly than vanilla DT during deeper causal processing, with the clearest amplification emerging by Transformer block 2.

E1 asks:

> Is this effect specific to the direction and structure of the CSDPC state perturbations, or is it a more generic consequence of training Decision Transformer jointly with the MTM objective?

No model is retrained.

---

## 2. Frozen Models

For each training seed `s in {0, 1, 2}`, use exactly the frozen clean models from Group 4E:

- clean vanilla Decision Transformer from Group 4C;
- clean DT policy branch extracted from the Group 4B DT+MTM model.

Both models remain in inference mode.

No parameter is updated during E1.

---

## 3. Source Perturbation Artifacts

Use exactly the same 12 frozen perturbation artifacts used by Group 4E:

| Condition       | Poison rate | Seeds     |
| --------------- | ----------: | --------- |
| `canonical`     |      `0.01` | `0, 1, 2` |
| `canonical`     |      `0.05` | `0, 1, 2` |
| `s2_overlap_r0` |      `0.01` | `0, 1, 2` |
| `s2_overlap_r0` |      `0.05` | `0, 1, 2` |

The existing CSDPC artifacts are read-only inputs.

They must not be regenerated or modified.

---

## 4. Control Construction

For each state-modified transition `t`, define the original CSDPC state perturbation:

```text
delta_csdpc[t] =
    poisoned_observation[t]
    - clean_observation[t]
```

For every state dimension `j`, draw a deterministic random sign:

```text
r[t, j] in {-1, +1}
```

Construct:

```text
delta_control[t, j] =
    r[t, j] * delta_csdpc[t, j]
```

Then:

```text
control_observation[t] =
    clean_observation[t]
    + delta_control[t]
```

The random signs are generated from a frozen deterministic analysis seed.

### Important

E1 randomizes only the direction of state perturbations.

It preserves the absolute magnitude of every original CSDPC state-perturbation component.

Actions remain clean because the finalized Group 4E mechanism finding was almost entirely state-channel driven.

---

## 5. Required Matching Invariants

All invariants must pass before any scientific result is interpreted.

### 5.1 Modified transition indices

The state-modified transition indices must be identical:

```text
control_modified_indices
==
csdpc_modified_indices
```

No transition may be added or removed.

### 5.2 Component-wise perturbation magnitude

For every modified transition `t` and state dimension `j`:

```text
abs(delta_control[t, j])
==
abs(delta_csdpc[t, j])
```

up to floating-point storage tolerance.

Therefore the control also preserves the original per-transition:

- L1 norm;
- L2 norm;
- Linf norm.

### 5.3 Untouched quantities

The control must not modify:

- actions;
- rewards;
- terminals;
- timeouts;
- return-to-go construction;
- trajectory boundaries;
- state normalization;
- clean model parameters.

---

## 6. Primary Outcome

The primary E1 quantity is the existing Group 4E layerwise historical-state statistic:

```text
C_l =
    mean(shift_joint[l])
    - mean(shift_dt[l])
```

where `shift[l]` is the hidden-representation change caused by historical state corruption at Transformer level `l`.

The primary layer is:

```text
block2
```

For every artifact report:

- `C_block2` under original CSDPC state perturbation;
- `C_block2` under the matched random-direction control;
- paired difference:

```text
delta_C_block2 =
    C_block2_control
    - C_block2_csdpc
```

---

## 7. Secondary Outcomes

E1 repeats selected frozen Group 4E measurements for both perturbation sources.

### 7.1 Action sensitivity

For each model:

```text
S =
    L2(
        predicted_action(perturbed_context)
        - predicted_action(clean_context)
    )
```

Report separately:

- direct-endpoint sensitivity;
- history-only sensitivity.

Define:

```text
A =
    mean(S_joint)
    - mean(S_dt)
```

Positive `A` means the DT+MTM policy is more sensitive than vanilla DT.

### 7.2 Layerwise historical-state propagation

Report:

- `C_input`;
- `C_block1`;
- `C_block2`;
- `C_block3`;
- `C_action`.

### 7.3 Attention routing

Using exactly the same historical state-token positions as the original CSDPC artifact, compute:

```text
D_l =
    mean(rerouting_joint[l])
    - mean(rerouting_dt[l])
```

Report:

- `D_block1`;
- `D_block2`;
- `D_block3`.

---

## 8. Sampling Contract

Reuse the deterministic endpoint-sampling rules already frozen by Group 4E.

Do not introduce new endpoint samples for the control condition.

For every artifact:

- CSDPC and matched-control measurements use the same direct endpoints;
- CSDPC and matched-control measurements use the same history-only endpoints;
- layerwise measurements use the same sampled endpoints;
- attention-routing measurements use the same sampled endpoints.

This keeps the comparison paired.

---

## 9. Interpretation Rule

E1 is a mechanistic control.

It is not a new robustness benchmark.

The 12 artifact rows reuse only three distinct clean DT / DT+MTM model pairs.

Therefore:

> `12/12` artifact consistency must not be described as 12 independent model replications.

### Evidence for a generic joint-training effect

The generic-effect interpretation becomes stronger if the matched random-direction control:

- preserves positive history-only excess sensitivity;
- preserves positive block-2 amplification;
- reproduces the effect across all three clean model seeds;
- produces a block-2 effect of similar qualitative magnitude to the CSDPC condition.

This would indicate that the Group 4E effect is not strongly dependent on the specific CSDPC perturbation direction.

### Evidence for CSDPC-direction sensitivity

The CSDPC-specific interpretation becomes stronger if:

- the original CSDPC perturbations retain strong positive block-2 amplification;
- the matched random-direction control substantially weakens that amplification;
- or the control changes its sign consistently across clean model seeds.

### Inconclusive result

If the matched control produces mixed or strongly seed-dependent behavior, E1 is classified as:

`INCONCLUSIVE`

---

## 10. Relationship to Return-Level Results

E1 does not test policy return.

It does not use Group 4D stress-response gap `G` as an outcome.

Even if E1 identifies a perturbation-specific mechanism, it does not establish that the mechanism explains Group 4D return behavior.

The finalized study already found weak relationships between the mechanism probes and return-level `G`.

Therefore the mechanistic and behavioral findings remain separate unless a later experiment directly establishes a link.

---

## 11. Claim Boundary

E1 may support a statement about whether the previously observed historical-state mechanism is:

- generic to matched state perturbations; or
- sensitive to CSDPC perturbation direction.

E1 cannot establish:

- that CSDPC is an effective poisoning attack;
- that DT+MTM is a poisoning defense;
- that block-2 amplification causes return degradation;
- that block-2 amplification explains Group 4D `G`;
- generalization beyond the frozen Walker2d models.

---

## 12. Output

The analysis writes:

```text
experiments/postfinal_controls/e1_matched_random_control.json
```

The output must contain:

- control-construction invariants;
- per-artifact CSDPC measurements;
- per-artifact matched-control measurements;
- aggregate summaries;
- seed-level block-2 summaries;
- paired block-2 comparison statistics.

No result interpretation is frozen until all control invariants pass.
