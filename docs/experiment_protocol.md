# Experiment Protocol

## Group 1 goal
`D_clean -> DT -> J_clean`

Gate A asks whether the clean Decision Transformer baseline is correct, causal, reproducible, stable across the three development seeds, and reasonably compatible with the reference Walker2d-medium result.

No poisoning or MTM code is active during Group 1.

## Run immutability
Each completed run should preserve its resolved config, config hash, dataset hash, git commit, dependencies, seed, training metrics, every evaluation episode, checkpoints, and final summary. Do not overwrite completed run directories.
