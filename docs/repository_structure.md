# Repository Structure

- `configs/` — human-readable experiment configurations.
- `data/` — raw, clean-derived, poisoned-derived, and metadata files.
- `src/methods/` — learning algorithms.
- `src/attacks/` — poisoning attacks.
- `src/evaluation/` — shared evaluation logic.
- `src/data/` — shared dataset and trajectory processing.
- `src/utils/` — reproducibility, hashing, seeding, logging.
- `scripts/` — user-facing entry points.
- `tests/` — correctness and causality tests.
- `experiments/` — immutable per-run artifacts.
- `results/` — cross-run summaries and figures.
- `schemas/` — common output contract.

Method folders are deliberately simple: `dt`, `bc`, `cql`, `rdt`, `dt_mtm`.
The mother folder already tells you whether it is a config, method implementation, or experiment.
