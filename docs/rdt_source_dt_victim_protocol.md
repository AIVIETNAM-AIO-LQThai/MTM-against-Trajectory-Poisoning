# Source-Compatible RDT Vanilla-DT Victim — Predeclared Protocol

## Purpose

The repository's frozen Hugging-Face-GPT2 Decision Transformer failed to complete the clean 2% Walker2d-medium-v2 control. Two ordinary continuations from the same 60,000-update checkpoint independently re-entered a late-training exploding-gradient regime and terminated before 100,000 updates.

The data construction and corruption artifacts already passed structural preflight. This experiment therefore reuses those artifacts unchanged and changes only the victim implementation, using the public RDT repository's vanilla-DT training/model semantics as the external reference.

## Frozen input artifacts

Reuse without modification:

- `data/derived/rdt_source_random_state_corruption/walker2d-medium-v2/clean_ratio_0p02.hdf5`
- clean SHA256 `a2b0eedb6083b2d28f4ea0b5ce220dd6763df8c8152ce8209e98292a359ad401`
- 23 whole trajectories
- 20,147 transitions
- no trailing fragment
- downsampling seed 1234
- corruption seeds 2023, 2024, 2025
- random observation corruption rate 0.30
- corruption scale 1.0

## Source-compatible victim

Reference repository: `jiawei415/RobustDecisionTransformer`

Reference commit: `865fb60632153ed7d8a49e7941b675f026480006`

Port the public vanilla `DecisionTransformer`, `TransformerBlock`, `MLPBlock`, sequence batching, initialization, loss, optimizer, and seed semantics needed for this experiment.

Model:
- embedding dimension 128
- 3 transformer blocks
- 1 attention head
- sequence length 20
- episode length 1000
- timestep embedding table length 1020
- attention dropout 0.0
- residual dropout 0.1
- embedding dropout disabled
- prediction dropout 0.1
- GELU feed-forward activation
- final action head: Linear + Tanh
- state normalization disabled

Initialization:
- Linear and Embedding weights: Normal(0, 0.02)
- Linear biases: 0
- LayerNorm bias: 0
- LayerNorm weight: 1

## Sequence sampling

Match source semantics:
1. sample trajectory IDs with NumPy proportional to trajectory length;
2. choose each start with `np.random.randint(0, trajectory_length)`;
3. take up to 20 consecutive transitions;
4. right-pad states/actions/returns with zero;
5. mask = valid ones followed by padded zeros;
6. use `np.arange(start_idx, start_idx + 20)` for timesteps, including padded positions;
7. no state normalization;
8. multiply RTG by 0.001.

## Loss

Match the source reduction exactly:

```python
loss = F.mse_loss(
    predicted_actions,
    actions.detach(),
    reduction="none",
)
loss = (
    loss
    * mask.unsqueeze(-1)
).mean()
```

## Optimization

- AdamW
- lr 1e-4
- betas (0.9, 0.999)
- weight decay 1e-4
- batch 64
- 100 epochs
- 1000 updates/epoch
- 100,000 total updates
- 10,000-step warmup
- gradient clip 0.25

Source-style seeding:
- Python `random`
- NumPy
- PyTorch CPU/CUDA
- cuDNN deterministic = true
- cuDNN benchmark = false

## Clean viability gate first

Before any corrupted training, train clean seeds 0, 1, and 2.

The victim is viable only if all three:
- reach 100,000 updates;
- keep finite losses;
- encounter no non-finite gradient norm;
- finish with finite parameters;
- finish with finite Adam first/second-moment states.

If any clean seed fails, stop before corrupted training.

## Corruption qualification

Only if clean viability passes, train the 3 x 3 corrupted matrix.

For corruption seed `c` and model seed `s`:

`Delta(c,s) = J_clean(s) - J_corrupt(c,s)`

Pass only if all hold:
1. mean degradation >= 5% of fresh clean mean;
2. at least 7/9 cells positive;
3. every model seed mean degradation positive;
4. every corruption seed mean degradation positive.

## Evaluation boundary

For continuity with this repository:
- Walker2d-v3
- target return 5000
- 100 episodes
- evaluation seeds 30000–30099

Therefore this is source-compatible victim training, not an exact end-to-end reproduction of the public RDT benchmark.

## Anti-tuning

Do not change data, corruption, architecture, padding, timestep semantics, normalization choice, loss reduction, optimizer settings, clean viability gate, or attack qualification gate after victim outcomes are observed.
