# Group 4 clean DT+MTM protocol

## Question

Can the validated causal Decision Transformer be trained jointly with the validated bidirectional MTM objective without a practically meaningful clean-performance collapse, before any poisoning result is examined?

## Frozen primary architecture

- DT path: unchanged Group-1 Walker2d Decision Transformer.
- MTM path: unchanged Group-3 MTM architecture, initialized from scratch.
- Shared trainable parameters: DT state and action input embeddings only.
- DT return-to-go and MTM future-value return target remain separate.
- No Group-3 MTM checkpoint is loaded in the primary experiment.

## Sampling

The two objectives use independent sampling streams from the same clean Walker2d dataset.

- DT keeps the Group-1 trajectory-length-proportional sampler, batch size 64, context length 20.
- MTM keeps the Group-3 trajectory-level 95/5 split, fixed trajectory length 4, train-window shuffling, reference tokenizers, and AUTO_MASK semantics.
- Joint MTM batch size is fixed to 512 for Group 4 to control joint-training compute.

The streams are not artificially aligned to the same window because doing so would change the validated DT sampling distribution.

## Optimization

The primary joint objective is

`L_total = L_DT + lambda_mtm * L_MTM`.

A single AdamW optimizer is used so the shared parameters receive the mathematically defined combined gradient. The outer optimizer contract follows Group 1:

- learning rate: 1e-4
- weight decay: 1e-4
- linear warmup: 10,000 updates
- DT-parameter gradient clipping: 0.25 (same complete DT parameter set as Group 1)
- auxiliary-only MTM/bridge gradient clipping: 0.25, applied separately so MTM-only gradient magnitude cannot rescale the entire DT update
- full clean horizon: 100,000 updates

The standalone Group-3 optimizer/schedule is not simultaneously applied because two optimizers would update the shared parameters twice and would no longer optimize the stated joint objective.

## Lambda freeze rule

Lambda is selected once, on clean seed 0 only, before any poisoned DT+MTM result exists.

Candidate grid:

`0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0`

Using 16 clean initialization batches, compute the shared-parameter gradient norms for DT and unscaled MTM. Let

`r = median(||g_MTM|| / ||g_DT||)`.

Reference AUTO_MASK can legitimately produce a batch with zero MTM gradient on
the shared DT state/action embeddings (for example, when all state/action input
tokens are hidden). Such a batch is part of the frozen mask distribution and is
recorded as ratio `0.0`; it is not resampled or discarded. Its gradient cosine
is undefined and is omitted only from the cosine-summary statistic. A zero or
non-finite DT shared gradient remains a calibration error.

Choose the largest candidate satisfying

`lambda_mtm * r <= 0.25`.

This predeclared cap makes the auxiliary gradient non-dominant on the exact parameters shared with the policy. If no candidate satisfies the rule, Group 4B stops for clean-integration diagnosis; the grid is not extended after seeing poisoned results.

## Diagnostics

During joint training, record:

- DT action loss
- MTM total/full state/action/return losses
- total joint loss
- total pre-clip gradient norm
- shared DT gradient norm
- shared MTM gradient norm
- lambda-scaled shared MTM gradient norm
- shared-gradient cosine similarity

Shared-gradient diagnostics are sampled periodically rather than every update to limit compute.

## Clean compatibility gate

The clean joint model must complete correctness/smoke checks and then the same seed set used by Group 1. Policy evaluation uses the DT branch only and the frozen Group-1 Walker2d evaluation protocol.

A poisoning-defense claim is not permitted at this stage. Group 4B only determines whether the joint learning path is a viable clean baseline.

## Predeclared policy-return compatibility thresholds

These thresholds are frozen before the real Group-4 clean policy is evaluated.

- Seed-0 compute screen: after the full 100,000-update seed-0 run, its 100-episode normalized-return mean at target return 5000 must be at least 80% of the frozen Group-1 seed-0 mean (69.32201154593943), i.e. at least 55.45760923675155. Falling below this is treated as a severe clean collapse and the remaining joint seeds are not launched until the integration is diagnosed. Passing this screen is not the final clean gate.
- Final clean compatibility gate: after matched clean seeds 0/1/2, the joint three-seed mean must be at least 90% of the frozen Group-1 three-seed mean (69.80938768165248), i.e. at least 62.82844891348723. This is a compatibility criterion, not evidence of poisoning robustness.
