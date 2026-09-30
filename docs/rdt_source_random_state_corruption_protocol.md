# RDT-Source Random State Corruption — Predeclared Protocol

## Purpose

Previous custom corruption regimes did not reliably degrade this repository's vanilla Decision Transformer.
The next experiment therefore adopts the downsampling and random state-corruption semantics from the public
implementation accompanying *Tackling Data Corruption in Offline Reinforcement Learning via Sequence Modeling*
(ICLR 2025).

This is not claimed as an exact reproduction of the paper's DT baseline. The corruption/data-scarcity mechanics
are source-derived, while this repository's frozen vanilla-DT architecture, optimizer, and evaluation pipeline
remain unchanged.

## Data-scarce dataset

Start from the frozen `walker2d-medium-v2` HDF5.

1. Parse the 1190 completed trajectories using terminal/timeout boundaries.
2. Exclude the trailing 5-transition incomplete fragment.
3. Set Python `random.seed(1234)`.
4. Select `int(1190 * 0.02) = 23` completed trajectories with `random.sample`.
5. Concatenate those whole trajectories in the sampled order.

This mirrors the public RDT repository's `utils/ratio_dataset.py` semantics for `medium-v2`.

## Random state corruption

For each corruption seed in `{2023, 2024, 2025}`:

1. Construct `np.random.RandomState(seed)`.
2. Draw one uniform random value per transition.
3. Corrupt indices where `u < 0.30`.
4. Compute per-dimension observation std from the clean downsampled dataset.
5. Continue using the same RNG instance and draw uniform noise in `[-1, 1]`.
6. Replace each attacked observation with:
   `s'_t = s_t + noise_t * std(clean_downsampled_observations)`
7. Do not clip.
8. Leave actions, rewards, terminals, and timeouts unchanged.

The attacked transition count is Bernoulli-sampled and is not forced to exactly 30%.

## Repository-specific normalization

The public RDT DT baseline disables state normalization by default. This repository's frozen DT pipeline uses
state normalization. Therefore:

- compute `state_mean` and `state_std` from the clean downsampled dataset once;
- reuse those exact statistics for all clean and corrupted runs.

This keeps normalization fixed across the comparison.

## Victim matrix

Fresh clean controls:
- model seeds 0, 1, 2

Corrupted runs:
- corruption seeds 2023, 2024, 2025
- model seeds 0, 1, 2

Total:
- 3 clean
- 9 corrupted
- 12 training runs

Training:
- 100,000 updates
- batch size 64
- learning rate 1e-4
- weight decay 1e-4
- warmup 10,000
- gradient clip 0.25

Evaluation:
- Walker2d-v3
- target return 5000
- 100 episodes
- seeds 30000–30099

## Frozen qualification gate

For corruption seed `c` and model seed `s`:

`Delta(c,s) = J_clean(s) - J_corrupt(c,s)`

Pass only if all hold:

1. overall mean degradation >= 5% of the fresh clean mean;
2. at least 7/9 cells have positive degradation;
3. every model seed has positive mean degradation across corruption seeds;
4. every corruption seed has positive mean degradation across model seeds.

No full-data clean floor is imposed because the experiment intentionally changes to a 2% data-scarce regime.

## Anti-tuning

Do not change the dataset ratio, downsampling seed, corruption field, corruption rate, corruption scale, corruption
seeds, normalization rule, or gate after victim outcomes are observed.
