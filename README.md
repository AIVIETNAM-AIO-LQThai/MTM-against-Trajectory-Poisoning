# MTM-Enhanced Decision Transformer Against Trajectory-Level Poisoning

Research repository scaffold for studying whether Masked Trajectory Modeling (MTM) can improve the robustness of a causal Decision Transformer (DT) against trajectory-level / coverage-targeted poisoning in offline reinforcement learning.

## Current active scope

**Group 2 — Attack Reproduction + Vulnerability** is active.

Frozen baseline:
- dataset: `walker2d-medium-v2`
- clean method: vanilla Decision Transformer
- Gate A baseline tag: `clean-dt`
- development seeds: `0, 1, 2`

Current Group-2 workflow:

1. reproduce CSDPC as an independent dataset transformation;
2. validate CSDPC structurally and mechanistically;
3. validate the attack on CQL (Gate B);
4. test transfer to the frozen Decision Transformer (Gate C);
5. compare against Behavior Cloning.

MTM and DT+MTM are not implemented in Group 2.
No defense may be added until the attack reproduction and
DT vulnerability experiments are completed.

## Naming
Folders are named by what they contain: `dt`, `bc`, `cql`, `rdt`, and `dt_mtm`.
`dt_mtm` is the filesystem-safe form of “DT + MTM”; using spaces or `&` would make Python imports and shell commands unnecessarily awkward.

See `docs/repository_structure.md` and `docs/research_matrix.md`.
