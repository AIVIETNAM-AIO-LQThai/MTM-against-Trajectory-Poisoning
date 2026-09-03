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

## Official CQL reference snapshot

Repository:
https://github.com/aviralkumar2907/CQL

Commit:
d67dbe9cf5d2b96e3b462b6146f249b3d6569796

The official CQL repository is treated as a read-only reference snapshot.
CQL-specific implementation details not specified by the CSDPC paper are
resolved against this exact commit.

Local modifications to the reference repository:
none

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

## Official CQL implementation ambiguities

### MuJoCo runtime configuration

The frozen official CQL source contains configuration defaults in
`d4rl/examples/cql_mujoco_new.py`, while the official repository README
recommends different command-line values for D4RL MuJoCo experiments.

This is treated as a SOURCE_CONFLICT rather than silently selecting one
configuration.

In particular:

- `min_q_version=3` is consistently recommended.
- the example configuration contains `min_q_weight=1.0`.
- the official MuJoCo README recommends `min_q_weight=5.0 or 10.0`.
- the example configuration enables Lagrange CQL.
- the official MuJoCo README example passes `lagrange_thresh=-1.0`,
  which disables the Lagrange variant.
- constructor defaults in `rlkit/torch/sac/cql.py` are not treated as
  experimental settings when the example/CLI explicitly overrides them.

CSDPC-specified CQL parameters retain higher precedence where the CSDPC
supplement explicitly gives a value.

Any remaining CQL-specific conflict must be frozen before poisoned CQL
training.

### Random-seed handling

The official CQL MuJoCo script exposes a `--seed` argument.

The frozen source must be checked for actual calls that seed Python,
NumPy, PyTorch, CUDA, and the evaluation environment.

If the argument is not operationally applied to those random generators,
the project will add explicit deterministic seed initialization in its
own CQL wrapper.

The official reference clone itself will remain unmodified.

This is a PROJECT_REPRODUCIBILITY_FIX and not a CQL algorithm change.

### Seed resolution

The frozen `cql_mujoco_new.py` exposes `--seed` and stores the value in
`variant["seed"]`, but the script executes the experiment without an
observed call to the RLKit `set_seed()` path.

Status:
PROJECT_REPRODUCIBILITY_FIX_REQUIRED

The project CQL runner will explicitly seed:

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA
- evaluation environment where supported

The official CQL reference clone remains unmodified.

### D4RL dataset transformation dependency

The frozen official CQL repository does not define
`qlearning_dataset()` internally.

The MuJoCo entry point calls:

    d4rl.qlearning_dataset(eval_env)

The search of the frozen CQL repository found calls to
`qlearning_dataset()` in the MuJoCo and AntMaze entry points, but no
local definition of that function.

Therefore, the exact offline-transition transformation is supplied by
the external D4RL dependency installed in the CQL environment.

Status:
SOURCE_DEPENDENCY_EXTERNAL_D4RL

The project CQL training-view adapter must be verified against the exact
installed D4RL version before Gate-B training.

The equivalence check must cover at least:

- whether the transformation iterates over N or N-1 raw transitions;
- construction of `next_observations`;
- handling of timeout transitions;
- handling of true terminal transitions;
- the default value and behavior of `terminate_on_end`;
- output dtypes;
- any transition filtering performed by D4RL.

No poisoned CQL run may begin until this equivalence check passes.

The project-owned `raw_indices` field is audit metadata only and must
never be supplied to CQL as a learning feature.

### CQL dependency specification

The frozen official CQL repository does not provide a dedicated
requirements file for the D4RL CQL experiments.

The same frozen repository snapshot contains conflicting dependency
guidance.

`d4rl/environment/linux-gpu-env.yml` specifies:

- Python 3.5.2
- NumPy 1.11.3
- PyTorch 0.4.1
- Gym 0.10.5
- gtimer 1.0.0b5

However, the frozen CQL README explicitly instructs users of the D4RL
experiments to install the RLKit environment while ensuring:

    torch >= 1.1.0

The environment file and `cql_mujoco_new.py` were introduced by the
same frozen repository commit, so their discrepancy cannot be resolved
by repository age alone.

The CQL README also instructs users to install D4RL externally but does
not pin an exact D4RL commit or package version.

Status:
SOURCE_CONFLICT

For Gate B, the project will therefore use a Python-3.8-compatible
software environment that preserves the APIs and numerical behavior
required by the frozen CQL implementation and the Walker2d-medium-v2
D4RL task.

Exact package versions will be frozen in a project environment manifest
before any CQL performance result is observed.

Software compatibility changes must not alter the CQL objective,
architecture, attack artifacts, or frozen hyperparameters.

## D4RL Training-View Resolution

The frozen official CQL repository delegates MuJoCo offline-dataset
conversion to the external:

    d4rl.qlearning_dataset(...)

function.

For Gate B, the exact installed D4RL distribution and runtime software
stack are recorded in:

- `configs/environments/g2-cql.yml`
- `configs/environments/g2-cql-pip-freeze.txt`
- `configs/environments/g2-cql-runtime.txt`
- `docs/d4rl_qlearning_dataset_snapshot.txt`

The project-owned CQL dataset adapter was tested field-by-field against
the installed D4RL `qlearning_dataset()` implementation with
`terminate_on_end=False`.

Equivalence covers:

- observations;
- actions;
- next observations;
- rewards;
- terminals;
- timeout filtering;
- output dtypes.

Status:
VERIFIED_AGAINST_FROZEN_D4RL

The project-only `raw_indices` field is retained exclusively for
attack-exposure auditing and is never supplied to the CQL learner.