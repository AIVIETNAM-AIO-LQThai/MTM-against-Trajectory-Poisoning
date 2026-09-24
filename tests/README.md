# Test Suite

The repository contains tests accumulated across the four experimental stages of the project.

There is intentionally no single historical environment that is expected to execute every test in this repository.

Each experiment group should be validated using the software environment associated with that group.

## Group 1 — Decision Transformer

Group-1 tests validate the clean Decision Transformer foundation, including:

- dataset validity;
- trajectory boundaries;
- return-to-go construction;
- normalization;
- sequence alignment;
- padding and masked losses;
- causal behavior and future leakage;
- DT model and inference behavior;
- training behavior;
- evaluation/scoring;
- checkpoint compatibility;
- seed reproducibility.

The historical reference environment is documented in:

`environment.reference.yml`

The actual frozen evaluation runtime is additionally recorded in:

`data/metadata/reference_evaluation_environment.txt`

The evaluation-reproducibility integration test additionally requires the local frozen clean-DT checkpoint.

That checkpoint is intentionally not stored in the final Git tree because large `.pt` experiment checkpoints are excluded from source control.

Therefore:

`tests/test_evaluation_reproducibility.py`

is an artifact-dependent integration test and should only be executed when the required local checkpoint is available.

## Group 2 — CSDPC and CQL

Tests whose filenames begin with:

- `test_csdpc_`
- `test_cql_`

belong to the Group-2 attack/audit/CQL work.

The frozen CQL training environment is:

`configs/environments/g2-cql.yml`

Additional frozen runtime information is stored in:

- `configs/environments/g2-cql-runtime.txt`
- `configs/environments/g2-cql-pip-freeze.txt`

This environment contains the legacy D4RL/Gym/MuJoCo stack required by the CQL and D4RL-equivalence tests.

The CSDPC CQL-summary utility additionally uses pandas. Pandas is not present in the frozen CQL training environment snapshot, so this analysis dependency should not be interpreted as part of the original CQL training runtime.

## Group 3 — Masked Trajectory Model

Tests whose filenames begin with:

`test_mtm_`

validate the standalone MTM reproduction.

The project-aligned reference environment is:

`environment.group3-mtm.yml`

with a corresponding lock snapshot:

`requirements.group3-mtm-lock.txt`

The reference environment was intentionally kept minimal and does not include legacy D4RL or Gym dependencies from Groups 1 and 2.

## Group 4 — DT + MTM

The primary Group-4 integration tests are:

- `test_dt_mtm_integration.py`
- `test_dt_mtm_clean_pipeline.py`
- `test_dt_mtm_trainer.py`
- `test_dt_stress_pipeline.py`

The environment used for the finalized joint DT+MTM study should be recorded under:

`configs/environments/g4-dt-mtm.yml`

with corresponding runtime and pip-freeze snapshots.

Group-4E mechanism scripts should additionally be checked with Python bytecode compilation.

## Running tests

Do not use failure of the entire historical test collection from an arbitrary group-specific environment as evidence of a repository regression.

For example, the Group-3 MTM environment is not expected to provide:

- D4RL;
- legacy Gym/MuJoCo;
- Group-2 CQL dependencies.

Likewise, an old DT/CQL environment should not be assumed to satisfy the MTM or joint-model dependency stack.

Tests should instead be executed in the environment corresponding to the experiment stage they validate.

## Full-suite interpretation

A command such as:

`python -m pytest`

collects tests from all historical stages.

It is only expected to succeed when the active Python environment satisfies the union of all historical dependencies and any required local experiment artifacts.

That is not the primary reproducibility contract of this repository.

The primary contract is stage-specific reproducibility using the frozen environment and artifacts associated with each experimental group.
