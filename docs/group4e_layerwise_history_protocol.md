# Group 4E — Layerwise historical-state propagation

## Question

DT+MTM contracts state perturbations at the input representation but is
more sensitive than vanilla DT to those perturbations when they occur in
causal history.

This analysis localizes where that reversal emerges inside the
three-layer causal GPT-2 backbone.

## Population

Use history-only endpoints:

- prediction endpoint itself is clean;
- at least one previous observation within K=20 is modified.

Only observation/state corruption is applied. Historical actions remain
clean.

At most 5,000 endpoints are deterministically sampled per artifact.

## Measurement

For each clean/poisoned context pair, measure the L2 change at the
CURRENT ENDPOINT STATE TOKEN.

Because the endpoint itself is clean, its pre-Transformer input is
identical between clean and poisoned contexts.

Measure endpoint-state hidden shift:

- transformer input;
- after GPT block 1;
- after GPT block 2;
- after GPT block 3 / final transformer representation;
- final predicted action.

For each level l:

C_l =
    mean shift_joint(l)
    - mean shift_DT(l)

Positive C_l means DT+MTM propagates more historical corruption into the
current representation than vanilla DT.

The key outcome is the earliest Transformer layer at which C_l becomes
consistently positive.
