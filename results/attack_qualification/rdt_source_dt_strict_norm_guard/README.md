# Strict total-gradient-norm guard result

Victim:
Source-compatible vanilla Decision Transformer ported from
jiawei415/RobustDecisionTransformer at commit
865fb60632153ed7d8a49e7941b675f026480006.

Runtime:
PyTorch 2.7.1 + CUDA 12.8 on RTX 5050 Laptop GPU.

Observed run:
- clean model seed 0
- 2% Walker2d-medium-v2 dataset
- 23 trajectories
- 20,147 transitions
- training stopped at update 12,996
- reported clip_grad_norm_ value was Infinity
- loss remained finite

Diagnostic replay from the valid update-10,000 checkpoint:
- first reproduced float32 total-norm overflow at update 12,614
- all individual gradient elements were finite
- manual float64 global gradient norm:
  2.546820047463574e19
- manual float32 global gradient norm:
  Infinity
- torch.nn.utils.clip_grad_norm_ returned:
  Infinity

Interpretation:
The strict viability guard treated a non-finite reported total norm as
training failure. This is stricter than the public RDT trainer, which
calls clip_grad_norm_ and does not inspect its return value.

Therefore this result is frozen as a guard-triggered stop, not as evidence
that the public source-compatible DT itself necessarily failed.

The next experiment uses source-faithful clipping semantics:
- individual gradient elements must remain finite before clipping;
- the reported total norm may overflow to Infinity and is telemetry only;
- optimizer execution follows the public source behavior;
- model parameters and Adam states must remain finite.
