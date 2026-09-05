from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np

from src.data.trajectories import (
    find_completed_trajectories,
)


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)

EXPECTED_FILE_SIZE = 232_254_996

EXPECTED_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea0"
    "851b2e5bedbe6fda073758a8f841aeda"
)

EXPECTED_RAW_TRANSITIONS = 1_000_000
EXPECTED_TRAINING_TRANSITIONS = 999_995

EXPECTED_NUM_TRAJECTORIES = 1_190
EXPECTED_TRAILING_TRANSITIONS = 5

EXPECTED_STATE_DIM = 17
EXPECTED_ACTION_DIM = 6

EXPECTED_TERMINAL_COUNT = 513
EXPECTED_TIMEOUT_COUNT = 677


def sha256_file(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def test_group1_dataset_file_exists():
    assert DATASET_PATH.exists(), (
        "\nFrozen Group-1 Walker2d dataset is missing.\n"
        f"Expected path:\n{DATASET_PATH}\n\n"
        "The raw dataset is git-ignored, so it must exist "
        "locally on this machine before Group-3 "
        "dataset-dependent tests can pass."
    )


def test_group1_dataset_file_identity():
    assert DATASET_PATH.stat().st_size == EXPECTED_FILE_SIZE

    actual_sha256 = sha256_file(
        DATASET_PATH
    )

    assert actual_sha256 == EXPECTED_SHA256, (
        "Walker2d HDF5 SHA256 differs from the "
        "frozen Group-1 dataset.\n"
        f"Expected: {EXPECTED_SHA256}\n"
        f"Actual:   {actual_sha256}"
    )


def test_group1_dataset_shapes_and_boundaries():
    with h5py.File(
        DATASET_PATH,
        "r",
    ) as handle:
        observations = handle[
            "observations"
        ]

        actions = handle[
            "actions"
        ]

        rewards = handle[
            "rewards"
        ]

        terminals = (
            handle["terminals"][:]
            .astype(bool)
        )

        timeouts = (
            handle["timeouts"][:]
            .astype(bool)
        )

        assert observations.shape == (
            EXPECTED_RAW_TRANSITIONS,
            EXPECTED_STATE_DIM,
        )

        assert actions.shape == (
            EXPECTED_RAW_TRANSITIONS,
            EXPECTED_ACTION_DIM,
        )

        assert rewards.shape[0] == (
            EXPECTED_RAW_TRANSITIONS
        )

        assert terminals.shape == (
            EXPECTED_RAW_TRANSITIONS,
        )

        assert timeouts.shape == (
            EXPECTED_RAW_TRANSITIONS,
        )

    assert int(terminals.sum()) == (
        EXPECTED_TERMINAL_COUNT
    )

    assert int(timeouts.sum()) == (
        EXPECTED_TIMEOUT_COUNT
    )

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    assert len(trajectories) == (
        EXPECTED_NUM_TRAJECTORIES
    )

    assert trailing == (
        EXPECTED_TRAILING_TRANSITIONS
    )

    used_transitions = sum(
        trajectory.length
        for trajectory in trajectories
    )

    assert used_transitions == (
        EXPECTED_TRAINING_TRANSITIONS
    )

    # Final consistency check:
    assert (
        used_transitions
        + trailing
        == EXPECTED_RAW_TRANSITIONS
    )


def test_every_completed_trajectory_ends_at_boundary():
    with h5py.File(
        DATASET_PATH,
        "r",
    ) as handle:
        terminals = (
            handle["terminals"][:]
            .astype(bool)
        )

        timeouts = (
            handle["timeouts"][:]
            .astype(bool)
        )

    trajectories, _ = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    for trajectory in trajectories:
        final_raw_index = (
            trajectory.end - 1
        )

        assert (
            terminals[final_raw_index]
            or timeouts[final_raw_index]
        )


def test_completed_trajectories_are_contiguous():
    with h5py.File(
        DATASET_PATH,
        "r",
    ) as handle:
        terminals = (
            handle["terminals"][:]
            .astype(bool)
        )

        timeouts = (
            handle["timeouts"][:]
            .astype(bool)
        )

    trajectories, _ = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    for current, following in zip(
        trajectories[:-1],
        trajectories[1:],
    ):
        assert current.end == following.start