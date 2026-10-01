# Attention execution-path diagnostic

Victim:
Source-compatible RDT vanilla Decision Transformer.

Runtime:
PyTorch 2.7.1 + CUDA 12.8, RTX 5050 Laptop GPU.

Starting state:
The exact same clean seed-0 checkpoint at update 10,000 was replayed
under two attention execution modes.

Comparison:

1. Current PyTorch-2.7 path
   - MultiheadAttention called with need_weights=False
   - enters rapidly exploding gradients
   - actual non-finite gradient elements appear at update 13,361
   - result: FAIL

2. Legacy-like explicit MHA path
   - MultiheadAttention called with need_weights=True
   - returned attention weights are discarded
   - remains numerically stable through update 15,000
   - loss stays approximately 0.012–0.017 late in the diagnostic
   - gradient norm stays approximately 0.25–0.45
   - result: PASS_TO_15000

Interpretation:
The experiment strongly implicates the modern attention execution path
as the source of the observed numerical instability.

This does not establish that need_weights=True is bit-for-bit equivalent
to PyTorch 1.8.1. It motivates a separately predeclared legacy-MHA
runtime-compatibility victim.

No attack/corruption outcome was observed in making this decision.
