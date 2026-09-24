# Group 4E — State-input representation amplification

## Question

Historical modality analysis showed that DT+MTM's excess sensitivity
comes almost entirely from observation/state corruption.

This analysis tests whether that excess sensitivity is already present
before causal Transformer mixing.

## Models

For each seed compare:

- frozen clean vanilla DT;
- frozen clean DT branch of DT+MTM.

No retraining is performed.

## Population

Use every transition whose observation differs between the clean and
corresponding poisoned artifact.

States are normalized using the frozen clean state_mean/state_std.

## Representation levels

For every changed state s -> s':

1. raw state embedding

   z = embed_state(s)

2. actual pre-Transformer state token

   h = embed_ln(embed_state(s) + embed_timestep(t))

The second quantity matches the state-token preprocessing used by
DecisionTransformer.forward before GPT-2 temporal mixing.

## Metrics

For each model:

E_raw =
    ||embed_state(s') - embed_state(s)||_2

E_token =
    ||token(s',t) - token(s,t)||_2

Report:

B_raw =
    mean(E_raw_joint) - mean(E_raw_DT)

B_token =
    mean(E_token_joint) - mean(E_token_DT)

and the corresponding joint/DT ratios and fractions where joint > DT.

Finally compare B_raw and B_token with the previously observed
history-only policy sensitivity A_history.

Interpretation:

- positive B_raw/B_token suggests MTM changes state-input sensitivity
  before causal temporal processing;

- near-zero or negative B with positive A_history suggests the extra
  history sensitivity emerges downstream in the causal Transformer.
