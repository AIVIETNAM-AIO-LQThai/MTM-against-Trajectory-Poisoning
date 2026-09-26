# Group 3 — Official MTM Source Audit

## Reference

Repository:
`facebookresearch/mtm`

Frozen commit:
`3547c23bf1daacb41db3332950226b3b12f00ab4`

Paper:
Masked Trajectory Models for Prediction, Representation, and Control
Wu et al., ICML 2023

## Reference architecture

- n_embd: 512
- n_enc_layer: 2
- n_dec_layer: 1
- n_head: 4
- activation: GELU
- dropout: 0.1
- architecture: bidirectional Transformer encoder + decoder
- modality-specific input embeddings
- modality-specific output heads

## Reference training configuration

- batch_size: 2048
- traj_length: 4
- learning_rate: 1e-4
- weight_decay: 0.005
- warmup_steps: 40000
- num_train_steps: 140010
- optimizer: AdamW
- scheduler: warmup followed by cosine decay

## Reference dataset fields and modeled modalities

The official continuous D4RL SequenceDataset uses
`use_reward: True`, so dataset samples may contain:

- states
- actions
- rewards
- returns

However, the official `d4rl_cont` tokenizer configuration creates
tokenizers only for:

- states
- actions
- returns

Therefore, for the reference continuous D4RL MTM:

- rewards are source data used to construct the return/value target;
- rewards are not a modeled/tokenized MTM modality;
- the modeled modalities are states, actions, and returns.

This distinction must be preserved in our reproduction.

## Important return-semantics finding

Official MTM `returns` must NOT be assumed equivalent to
Decision Transformer RTG.

The official SequenceDataset computes its own future-value signal.

Group-1 DT RTG semantics remain frozen and must not be changed.

## Continuous-tokenizer behavior

Official MTM standardizes continuous modalities using
training-data feature-wise mean and standard deviation.

Very small standard deviations are replaced according to the
official implementation.

MTM normalization must not modify the frozen Group-1 DT input
normalization.

Important implementation detail:

`SequenceDataset.trajectory_statistics()` computes modality
statistics over the segmented trajectory tensors, which are padded
to `max_path_length`.

Therefore reference tokenizer mean/std statistics include the
zero-padded portions of those tensors.

This behavior must be reproduced during the faithful reference
stage before testing cleaner project-specific alternatives.

## Reference masking

Primary reference mask:
AUTO_MASK

Mask convention:

- 1 = visible / retained
- 0 = hidden / reconstructed

Reference configured ratios:

- 0.50
- 0.60
- 0.70
- 0.80
- 0.85
- 0.90
- 0.95
- 1.00

AUTO_MASK modality weights:

- states: 0.2
- returns: 0.1
- actions: 0.7

To avoid ambiguity, project logging should prefer:

- visible_fraction
- masked_fraction

rather than an ambiguous `mask_ratio`.

### AUTO_MASK implementation semantics

The official naming is potentially misleading.

Mask values use:

- `1` = visible / conditioned
- `0` = hidden / reconstructed

`create_full_random_mask()` computes:

`int(traj_length * tokens_per_timestep * mask_ratio)`

entries equal to one.

Therefore the configured `mask_ratio` controls the initial
visible-token fraction before AUTO_MASK applies its additional
future frontier.

For the continuous D4RL tokenizers, each modality has one token
per timestep, so masks have shape `[T, 1]`.

AUTO_MASK samples:

1. a target modality from the order
   `states -> returns -> actions`,
2. a timestep,
3. independent initial random masks for each modeled modality,
4. then hides future tokens according to the sampled
   modality/timestep frontier.

This is a data-visibility mask, NOT a causal Transformer
attention mask. The MTM encoder/decoder remains bidirectional
over visible tokens.

The d4rl_cont tokenizer declaration order is:

1. states
2. actions
3. returns

Because the implementation samples modality masks while iterating
through that dictionary, this insertion order also affects the
exact NumPy RNG stream.

## Reference reconstruction objective

Critical finding:

The official implementation computes:

- full reconstruction loss
- masked reconstruction loss
- visible/conditioned reconstruction loss

but the default optimization objective uses FULL reconstruction
loss.

Therefore masked-only training is NOT part of the faithful
reference reproduction.

A masked-only objective may later be tested only as a separately
named project variant.

### Verified reconstruction-loss behavior

The official continuous D4RL configuration uses:

`norm: "none"`

Therefore continuous targets are reconstructed using raw
elementwise squared error in encoded/tokenized space.

For each modality, the optimized reference loss is the full
reconstruction loss:

`raw_loss.mean(dim=(2, 3)).mean()`

The total training objective is the sum of the selected
per-modality full reconstruction losses.

The implementation also reports:

- masked reconstruction loss;
- visible/conditioned reconstruction loss.

These are diagnostics only.

The official implementation contains a disabled masked-only branch
(`if False`) and therefore does not train on masked-only loss.

Another important scale detail is that masked and visible diagnostic
losses divide by the number of selected tokens but not by the
continuous feature dimension.

Consequently, for a modality with feature dimension D > 1,
masked/visible diagnostic magnitudes are not directly comparable
to the ordinary full-element MSE without accounting for this
reduction difference.

If a realization contains no hidden tokens or no visible tokens,
the corresponding diagnostic is undefined. Our implementation
reports this explicitly as NaN while keeping the full training
objective finite.

## Project adaptations

The following are project adaptations and must not be described
as exact reference behavior:

- walker2d-medium-v2 project dataset
- Group-1 trajectory segmentation contract
- Group-1 exact raw HDF5 identity
- transition-group masks
- contiguous-block masks
- masked-only reconstruction objective
- DT+MTM joint training
- CSDPC experiments

## Reproduction rule

First reproduce reference MTM behavior.

Only after standalone MTM is credible may project-specific
masking or DT integration begin.

## Project attack-status boundary

The independent CSDPC reproduction has been frozen with
`GATE B: INCONCLUSIVE`.

The standalone MTM reproduction remains scientifically independent
of that result and should continue unchanged.

Later CSDPC-derived experiments use the frozen datasets only as
audited trajectory-poisoning stress conditions.

See:

`docs/mtm_experiment_plan.md`

### Verified model execution path

The continuous reference-style MTM architecture is implemented as:

1. modality-specific linear encoder projection;
2. learned per-modality/per-token encoder encoding;
3. fixed 1D sin/cos timestep encoding divided by 2;
4. flatten time/token dimensions within each modality;
5. retain only `mask == 1` visible tokens;
6. concatenate visible tokens across modalities;
7. process visible tokens jointly with a bidirectional Transformer encoder;
8. split the encoded visible representations back by modality;
9. append learned mask tokens for hidden positions;
10. restore each modality to its original token ordering;
11. apply modality-specific decoder projection;
12. add decoder modality/token encoding and timestep encoding;
13. concatenate modalities and process them jointly with a
    bidirectional Transformer decoder;
14. split decoder output back by modality;
15. reconstruct each modality with an independent output head.

The output head is:

`LayerNorm -> Linear -> GELU -> Linear`

No causal Transformer attention mask is used in the standalone MTM
path.

A dedicated test verifies that altering a visible future token can
change reconstruction of an earlier hidden token.

A separate test verifies that altering the raw value of a hidden
token cannot affect model output because the hidden token is removed
before the encoder and replaced by a learned mask token before the
decoder.

The implementation also explicitly handles the valid AUTO_MASK case
where no tokens remain visible.

## Real-data integration smoke

The standalone MTM pipeline has been exercised end-to-end on the
frozen `walker2d-medium-v2` dataset.

The smoke pipeline uses:

- frozen Group-1 raw dataset identity;
- Group-1 trajectory segmentation;
- trajectory length 4;
- deterministic 95/5 trajectory split;
- training-only padded tokenizer statistics;
- official-style MTM future-value target;
- states/actions/returns continuous tokenizers;
- stochastic reference AUTO_MASK;
- bidirectional MTM encoder-decoder;
- full reconstruction training objective.

Validation trajectories are excluded from:

- training-window sampling;
- tokenizer-statistic estimation.

A fixed validation-window/mask bank is evaluated before and after
training so that reconstruction improvement is not attributable to
evaluation resampling.

This smoke test uses a reduced 64-dimensional model and is an
integration check, not the final reference-scale training run.
