# Group 4E — Historical-state attention routing

## Question

Layerwise analysis localized systematic DT+MTM amplification of
historical state corruption to the causal Transformer, becoming clear
by block 2.

This final mechanism probe tests whether that amplification is
associated with increased causal attention routing from the current
endpoint state token toward corrupted historical state tokens.

## Population

Use history-only state-corruption contexts:

- current endpoint observation is clean;
- one or more earlier observations inside K=20 are modified;
- actions remain clean.

At most 5,000 endpoints are deterministically sampled per artifact.

## Measurement

For every selected context, mark the historical STATE-token positions
whose observations are modified.

For each Transformer layer and each frozen clean model, measure the
attention mass from the current endpoint STATE query token to those
marked historical state tokens.

Evaluate both:

1. clean context;
2. state-poisoned context.

Define perturbation-induced attention rerouting:

R_model =
    attention_mass(poisoned_context)
    - attention_mass(clean_context)

Normalize by the number of poisoned historical state tokens.

Then define:

D_layer =
    mean(R_joint)
    - mean(R_DT)

Positive D means the MTM-trained policy reroutes more causal attention
toward attacked historical state positions when their observations are
perturbed.

This analysis is exploratory and is the final new Group-4E mechanism
probe.
