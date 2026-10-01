# Frozen clean-control instability result

Experiment:
RDT-source 2% Walker2d-medium-v2 random-state-corruption qualification.

Status:
CLEAN VICTIM NUMERICAL STABILITY FAILURE.
Attack qualification was not evaluated.

Observed behavior:
- clean model seed 0 trained normally through approximately 61k updates;
- the gradient norm then entered a rapidly exploding regime;
- the first training attempt terminated with a non-finite gradient norm before 100k;
- one ordinary resume from the frozen 60k checkpoint independently re-entered the same late-training instability region and again terminated with a non-finite gradient norm;
- no corrupted victim was trained before this failure.

Diagnostic findings:
- 2% state normalization was not more extreme than the full-data normalization;
- sampled state and RTG magnitudes remained ordinary during replay;
- large gradients were finite before failure and float32/float64 norm calculations agreed;
- gradient amplification concentrated in early/middle Transformer attention, MLP and LayerNorm parameters;
- the instability was therefore not attributed to a pathological corruption artifact or a single abnormal batch.

Scientific consequence:
The repository's frozen GPT2-based DT is not a viable clean victim for this 2% data-scarce regime under the predeclared 100k-update protocol. The corruption qualification is therefore not evaluable with that victim.
