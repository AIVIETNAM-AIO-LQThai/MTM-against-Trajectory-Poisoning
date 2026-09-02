# CQL Gate-B Source Audit

## Purpose

This document freezes the CQL implementation used to validate the
canonical CSDPC reproduction before any poisoned CQL result is observed.

CQL is a validation target for Gate B. Its configuration must not be
changed in response to observed attack strength.

## Source priority

1. CSDPC paper and supplementary material
2. Official CQL implementation by Kumar et al.
3. D4RL reference behavior
4. Project reproduction choices only where the above are underspecified

## Development setting

Environment / dataset:
walker2d-medium-v2

Attack conditions:
rho = 0.00, 0.01, 0.05

Project model seeds:
0, 1, 2

Paired development convention:
attack seed 0 -> model seed 0
attack seed 1 -> model seed 1
attack seed 2 -> model seed 2

The original CSDPC MuJoCo experiment reports seed 0.
The additional seeds 1 and 2 are project robustness checks.

## CQL source-specified parameters

| Parameter | Value | Source status |
|---|---:|---|
| Optimizer | Adam | VERIFIED_CSDPC_SUPPLEMENT |
| Critic learning rate | 0.003 | VERIFIED_CSDPC_SUPPLEMENT |
| Actor learning rate | 0.003 | VERIFIED_CSDPC_SUPPLEMENT |
| Batch size | 256 | VERIFIED_CSDPC_SUPPLEMENT |
| Critic hidden units | [256, 256, 256] | VERIFIED_CSDPC_SUPPLEMENT |
| Actor hidden units | [256, 256, 256] | VERIFIED_CSDPC_SUPPLEMENT |
| Discount gamma | 0.99 | VERIFIED_CSDPC_SUPPLEMENT |

## CQL-specific unresolved parameters

The following are not sufficiently specified by the CSDPC Table-8
configuration and must be resolved from the official CQL D4RL
implementation before training:

- min_q_weight
- min_q_version
- Lagrange configuration
- entropy / temperature configuration
- target update configuration
- reward scaling
- observation normalization
- number of training gradient steps
- evaluation frequency
- number of evaluation episodes
- replay-buffer construction
- handling of timeout transitions
- construction of next_observations

No unresolved parameter may be selected based on poisoned-agent
performance.

## Precedence rule

For parameters explicitly reported by the CSDPC supplement, the CSDPC
value is used.

For CQL-specific parameters omitted by the CSDPC supplement, the frozen
official CQL D4RL implementation value is used and recorded as
SOURCE_RESOLUTION_OFFICIAL_CQL.

If neither source operationally specifies a parameter, the project must
declare a REPRODUCTION_CHOICE before poisoned training begins.

## Anti-tuning rule

Clean CQL must be validated before any poisoned CQL result is used.

Once the CQL configuration is frozen, no hyperparameter may be changed
because it strengthens or weakens CSDPC.