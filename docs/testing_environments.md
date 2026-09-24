# Environments and Testing

## Principle

This repository records a multi-stage offline-reinforcement-learning research project.

The experimental stages were developed under different dependency stacks. In particular, legacy D4RL/Gym/MuJoCo requirements used by the clean-DT and CQL work are not identical to the dependencies used by the standalone MTM and joint DT+MTM experiments.

For reproducibility, the repository therefore preserves stage-specific environments rather than defining one universal environment for all experiments.

## Group 1 — Clean Decision Transformer

Historical reference specification:

`environment.reference.yml`

Recorded evaluation runtime:

`data/metadata/reference_evaluation_environment.txt`

The recorded evaluation runtime includes the legacy Walker2d stack used by the clean DT evaluation.

The environment specification and runtime record should be treated as historical provenance rather than silently reconciled if individual dependency versions differ.

Some integration tests require experiment artifacts such as trained checkpoints that are intentionally excluded from the final Git tree.

## Group 2 — CSDPC and CQL

The source-level CSDPC environment record is:

`environment.group2.yml`

The complete frozen CQL runtime is separately recorded under:

- `configs/environments/g2-cql.yml`
- `configs/environments/g2-cql-runtime.txt`
- `configs/environments/g2-cql-pip-freeze.txt`

The CQL environment contains the D4RL, Gym, MuJoCo, MJRL, NumPy, PyTorch, and CUDA stack used for the Group-2 learner-response experiments.

The Group-2 CQL-summary analysis code also imports pandas.

Pandas is not present in the frozen CQL pip-freeze snapshot. It should therefore be treated as an analysis/test dependency rather than retroactively described as part of the original frozen CQL training environment.

## Group 3 — Standalone MTM

The standalone MTM reference environment is:

`environment.group3-mtm.yml`

The corresponding package lock is:

`requirements.group3-mtm-lock.txt`

This environment intentionally contains only the dependencies needed for the project-aligned MTM reproduction.

It is not intended to execute legacy D4RL, CQL, or Walker2d evaluation integration tests.

## Group 4 — DT + MTM

The finalized Group-4 environment should be frozen under:

- `configs/environments/g4-dt-mtm.yml`
- `configs/environments/g4-dt-mtm-runtime.txt`
- `configs/environments/g4-dt-mtm-pip-freeze.txt`

This environment corresponds to the joint DT+MTM training, stress-transfer comparison, and mechanism-analysis work.

The environment should be exported from the actual runtime used to produce the Group-4 results rather than reconstructed later from guessed package versions.

## Test categories

Group-1 DT tests cover the clean causal DT foundation and legacy evaluation path.

Group-2 tests are identified primarily by the `test_csdpc_` and `test_cql_` prefixes.

Group-3 tests are identified by the `test_mtm_` prefix.

Group-4 integration tests are identified by the `test_dt_mtm_` prefix together with the DT stress-pipeline tests.

The artifact-dependent clean-DT evaluation-reproducibility test requires a local trained checkpoint that is intentionally excluded from Git.

## Recommended validation strategy

Validate each experimental stage inside its corresponding environment.

A failure to collect the entire repository-wide test suite from a stage-specific environment is not, by itself, evidence of a code regression.

For example, running the complete test suite from the MTM environment will fail to import Group-2 D4RL dependencies unless those unrelated legacy dependencies have also been installed.

The final repository should therefore report stage-specific test results and the environment used to obtain them.

## Scientific claim boundary

Environment documentation is provenance.

It should describe the software stack that actually produced an experiment rather than alter historical environment records merely to make unrelated tests executable from one modern environment.

Where a later analysis introduces an additional dependency, that dependency should be identified explicitly rather than silently inserted into the historical training environment.
