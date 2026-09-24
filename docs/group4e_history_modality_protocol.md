# Group 4E — Historical modality sensitivity

## Question

DT+MTM shows consistently greater sensitivity than vanilla DT to
corruption appearing only in preceding causal-history tokens.

The frozen poison artifacts modify both observations and actions.

This analysis separates those two channels.

## Population

Use history-only endpoints:

- endpoint itself is clean;
- at least one earlier token in its K=20 causal context is modified.

Use the same deterministic sampling rule as the previous frozen
clean-policy sensitivity audit, with at most 10,000 history-only
endpoints per artifact.

## Counterfactual contexts

For every selected endpoint construct:

1. clean:
   clean observations + clean actions;

2. state_only:
   poisoned observations + clean actions;

3. action_only:
   clean observations + poisoned actions;

4. both:
   poisoned observations + poisoned actions.

Rewards, RTG, timesteps, normalization, and attention masks remain
unchanged.

## Metrics

For each model and counterfactual:

S = || predicted_action(counterfactual)
       - predicted_action(clean) ||_2

For each modality define:

A_state  = mean(S_joint_state)  - mean(S_DT_state)
A_action = mean(S_joint_action) - mean(S_DT_action)
A_both   = mean(S_joint_both)   - mean(S_DT_both)

Positive A means DT+MTM is more sensitive than vanilla DT.

This analysis is frozen before viewing results.
