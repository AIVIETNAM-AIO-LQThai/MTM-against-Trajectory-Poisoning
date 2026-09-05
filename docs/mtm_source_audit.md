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

## Reference modalities

For the official continuous D4RL MTM configuration,
`use_reward: True`.

Therefore the reference SequenceDataset exposes:

- states
- actions
- rewards
- returns

The `returns` modality is internally computed from the
SequenceDataset future-value construction and is NOT the
same object as Group-1 Decision Transformer RTG.

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
