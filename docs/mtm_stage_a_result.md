# Standalone MTM Reproduction Result

## Status

`MTM REPRODUCTION: PASS`

This verdict applies to the project-aligned reference-style standalone
MTM implementation.

It establishes that the implementation can learn the clean
Walker2d trajectory reconstruction task under the audited reference
architecture, masking, optimization, and reconstruction semantics.

It does not claim exact reproduction of every preprocessing choice
or headline result from the original MTM paper.

## Frozen code identity

Branch:

`exp/mtm-reproduction`

Code commit:

`c3b39363f1887540ffe3330dd03352f529f3a00a`

Pinned upstream MTM source:

`facebookresearch/mtm@3547c23bf1daacb41db3332950226b3b12f00ab4`

## Dataset identity

Dataset:

`walker2d-medium-v2`

Project dataset path:

`data/raw/walker2d-medium-v2/walker2d_medium-v2.hdf5`

SHA256:

`CF00F43ADD04C17FDFC2958DD581DEA0851B2E5BEDBE6FDA073758A8F841AEDA`

Completed trajectories:

`1190`

Training trajectories:

`1130`

Validation trajectories:

`60`

Used transitions:

`999995`

Trailing transitions excluded:

`5`

## Training configuration

Seed:

`0`

Updates:

`140010`

Batch size:

`2048`

Learning rate:

`1e-4`

Weight decay:

`0.005`

Warmup steps:

`40000`

MTM embedding dimension:

`512`

Attention heads:

`4`

Encoder layers:

`2`

Decoder layers:

`1`

Dropout:

`0.1`

Trainable parameters:

`11068952`

Runtime:

- Python 3.10
- PyTorch 2.7.1+cu128
- NVIDIA GeForce RTX 5050 Laptop GPU

Elapsed training time:

`43377.2361 s` (~12.05 h)

## Validation result

| Metric         |   Initial |    Final | Reduction |
| -------------- | --------: | -------: | --------: |
| total full     |  2.709722 | 0.605839 |    77.64% |
| full states    |  1.018131 | 0.226296 |    77.77% |
| full actions   |  1.123692 | 0.279098 |    75.16% |
| full returns   |  0.567898 | 0.100446 |    82.31% |
| masked states  | 17.656144 | 4.412306 |    75.01% |
| masked actions |  6.829338 | 1.971133 |    71.14% |
| masked returns |  0.573674 | 0.101053 |    82.39% |

Masked diagnostics use the reference reduction convention and are
not directly comparable in absolute scale with full-element MSE.

The step-140000 and step-140010 validation results were effectively
identical, consistent with the cosine learning rate having decayed
to approximately zero.

No non-finite training or validation loss was observed.

## Frozen output artifacts

Final checkpoint:

`experiments/mtm/walker2d_medium_reference/seed_0/checkpoints/step_140010.pt`

SHA256:

`0D27F7A2B5E876E10E017544D965A33CE2EEC6C08E8DBD4CB0F6B340BBD3C187`

Evaluation summary:

`experiments/mtm/walker2d_medium_reference/seed_0/evaluation_summary.json`

SHA256:

`AA2B69D949676AAAB343A64B8AFB48EA1E585F185B5E27E8C4DA7B10B8076D41`

The large checkpoint remains a local experiment artifact and is not
committed to Git.

## Checkpoint-resume audit

Interrupted-versus-uninterrupted deterministic comparison:

`step: 12 12`

`model_exact_match: True`

`max_abs_diff: 0.0`

The checkpoint restores model, optimizer, scheduler, shuffled-window
sampler state, mask RNG state, and the relevant global RNG states.

## Interpretation

The standalone MTM implementation satisfies the Stage-A credibility
gate:

`MTM REPRODUCTION: PASS`

The result supports proceeding to clean DT+MTM integration.

The standalone MTM checkpoint is a validation artifact for the
reference-style implementation. It is not automatically assumed to
be the initialization for the later joint DT+MTM model.
