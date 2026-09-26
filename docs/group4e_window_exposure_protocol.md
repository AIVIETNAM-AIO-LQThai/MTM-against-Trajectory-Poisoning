# Group 4E — Poison sequence and window exposure

## Purpose

The raw localization audit established that the frozen perturbations
modify observations and actions, not rewards.

This analysis measures how those modified transitions are organized
inside trajectories and how broadly they can enter the sequence context
used by DT and MTM.

## Structural exposure definitions

Completed trajectories are reconstructed from clean terminals/timeouts.
The known trailing incomplete transitions are excluded.

DT:
- causal context length K=20;
- one structural context is defined for every transition endpoint;
- context consists of up to the previous 20 transitions including the
  endpoint.

MTM:
- window length K=4;
- all complete contiguous length-4 windows are enumerated.

For each poisoned dataset report:

1. realized changed transitions;
2. affected trajectories;
3. changed transitions per affected trajectory;
4. contiguous poison-run statistics;
5. fraction of DT causal contexts containing poison;
6. fraction of clean DT endpoints whose history contains poison;
7. fraction of MTM length-4 windows containing poison;
8. poison density inside exposed DT/MTM windows.

These are structural dataset-exposure quantities, not empirical
sampler-frequency estimates.
