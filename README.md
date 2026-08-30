# MTM-Enhanced Decision Transformer Against Trajectory-Level Poisoning

Research repository scaffold for studying whether Masked Trajectory Modeling (MTM) can improve the robustness of a causal Decision Transformer (DT) against trajectory-level / coverage-targeted poisoning in offline reinforcement learning.

## Current active scope
Only **Group 1 — Foundation + Clean Baseline** is active:
- dataset: `walker2d-medium-v2`
- method: vanilla Decision Transformer
- poisoning: disabled
- MTM: disabled
- development seeds: `0, 1, 2`

Do not implement CSDPC, BC, CQL, RDT, or DT+MTM until the clean DT baseline passes Gate A.

## Naming
Folders are named by what they contain: `dt`, `bc`, `cql`, `rdt`, and `dt_mtm`.
`dt_mtm` is the filesystem-safe form of “DT + MTM”; using spaces or `&` would make Python imports and shell commands unnecessarily awkward.

See `docs/repository_structure.md` and `docs/research_matrix.md`.
