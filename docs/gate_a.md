# Gate A — Clean DT Acceptance

Required checks:
1. Correct `walker2d-medium-v2` dataset and fingerprint.
2. Correct terminal/timeout segmentation.
3. Correct RTG calculation.
4. Train/eval normalization consistency.
5. Correct sequence alignment and padding.
6. Future-token causality test passes.
7. Tiny-overfit test passes.
8. Three clean DT development seeds finish with equal budget.
9. Environment-interaction performance is credible.
10. Config, code revision, dataset hash, dependencies, and results are saved.

Verdict must be exactly one of:
- `GATE A: PASS`
- `GATE A: FAIL`
- `GATE A: INCONCLUSIVE`
