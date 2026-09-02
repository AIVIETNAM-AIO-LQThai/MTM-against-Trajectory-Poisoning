import json

import numpy as np

from src.attacks.csdpc.attack import (
    apply_csdpc_attack,
    prepare_csdpc_attack,
)
from src.attacks.csdpc.metadata import (
    build_csdpc_metadata,
    logical_dataset_sha256,
    write_metadata_json,
)


def _dataset():
    rng = np.random.default_rng(123)
    n = 100

    observations = rng.normal(size=(n, 4))
    actions = rng.uniform(-0.8, 0.8, size=(n, 2))
    rewards = rng.normal(size=n)
    terminals = np.zeros(n, dtype=bool,)
    timeouts = np.zeros(n, dtype=bool,)

    timeouts[49] = True
    timeouts[99] = True

    infos_qpos = rng.normal(size=(n, 3))
    infos_qvel = rng.normal(size=(n, 2))

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
        "infos/qpos": infos_qpos,
        "infos/qvel": infos_qvel,
    }


def _prepared(dataset):
    return prepare_csdpc_attack(
        dataset,
        attack_seed=0,
        num_clusters=4,
        sequence_length=5,
        eta=0.05,
        num_candidates=20,
    )


def test_logical_hash_is_independent_of_mapping_order():
    dataset = _dataset()

    reversed_dataset = dict(
        reversed(
            list(
                dataset.items()
            )
        )
    )
    assert (
        logical_dataset_sha256(dataset)
        ==
        logical_dataset_sha256(reversed_dataset)
    )


def test_zero_rho_has_same_logical_hash():
    dataset = _dataset()

    prepared = _prepared(
        dataset
    )

    result = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.0,
    )

    metadata = build_csdpc_metadata(
        dataset,
        prepared,
        result,
    )

    assert (
        metadata["dataset"][
            "clean_logical_sha256"
        ]
        == metadata["dataset"][
            "poisoned_logical_sha256"
        ]
    )

    assert metadata[
        "integrity"
    ]["all_checks_passed"]


def test_nonzero_attack_metadata_passes_integrity():
    dataset = _dataset()

    prepared = _prepared(dataset)
    result = apply_csdpc_attack(dataset, prepared, rho=0.10)
    metadata = build_csdpc_metadata(dataset, prepared, result)

    assert metadata["integrity"]["all_checks_passed"]
    assert metadata["integrity"]["all_non_attack_arrays_identical"]

    assert (
        metadata["attack"][
            "actual_transition_budget"
        ]
        <= metadata["attack"][
            "requested_transition_budget"
        ]
    )


def test_metadata_json_roundtrip(
    tmp_path,
):
    dataset = _dataset()

    prepared = _prepared(
        dataset
    )

    result = apply_csdpc_attack(
        dataset,
        prepared,
        rho=0.10,
    )

    metadata = build_csdpc_metadata(
        dataset,
        prepared,
        result,
    )

    path = (
        tmp_path
        / "metadata.json"
    )

    write_metadata_json(
        path,
        metadata,
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        loaded = json.load(
            handle
        )

    assert (
        loaded["schema_version"]
        == "csdpc-artifact-metadata-v1"
    )

    assert loaded[
        "integrity"
    ]["all_checks_passed"]