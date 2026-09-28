from __future__ import annotations

import h5py
import numpy as np

from scripts.generate_a3_action_cyclic_shift import (
    apply_action_shift,
    choose_shift_offset,
    select_trajectories,
    shift_bounds,
    trajectory_returns,
    validate,
    write_action_only,
)
from src.data.trajectories import TrajectorySlice


def synthetic():
    trajectories = [
        TrajectorySlice(0, 4),
        TrajectorySlice(4, 9),
        TrajectorySlice(9, 15),
    ]

    actions = np.arange(
        17 * 3,
        dtype=np.float32,
    ).reshape(17, 3)

    rewards = np.asarray(
        [1.0] * 4
        + [2.0] * 5
        + [3.0] * 6
        + [99.0] * 2,
        dtype=np.float32,
    )

    return actions, rewards, trajectories


def test_shift_bounds():
    assert shift_bounds(4) == (1, 3)
    assert shift_bounds(5) == (2, 3)
    assert shift_bounds(10) == (3, 7)


def test_offset_is_deterministic_and_in_range():
    a = choose_shift_offset(
        length=100,
        attack_seed=20,
        trajectory_index=7,
    )
    b = choose_shift_offset(
        length=100,
        attack_seed=20,
        trajectory_index=7,
    )
    assert a == b
    assert 25 <= a <= 75


def test_selection_is_deterministic_and_high_return_only():
    _, rewards, trajectories = synthetic()
    returns = trajectory_returns(
        rewards,
        trajectories,
    )

    first = select_trajectories(
        trajectories,
        returns,
        threshold=10.0,
        min_length=4,
        requested_budget=11,
        attack_seed=20,
    )

    second = select_trajectories(
        trajectories,
        returns,
        threshold=10.0,
        min_length=4,
        requested_budget=11,
        attack_seed=20,
    )

    assert first == second

    selected, used, _, _ = first
    assert used <= 11
    assert all(
        returns[i] >= 10.0
        for i in selected
    )


def test_action_shift_mapping_and_multiset():
    actions, _, trajectories = synthetic()

    poisoned, records = apply_action_shift(
        actions,
        trajectories,
        [2],
        attack_seed=22,
    )

    record = records[0]
    start = record["start"]
    end = record["end"]
    k = record["shift_offset"]

    expected = np.roll(
        actions[start:end],
        -k,
        axis=0,
    )

    np.testing.assert_array_equal(
        poisoned[start:end],
        expected,
    )

    np.testing.assert_array_equal(
        np.roll(
            poisoned[start:end],
            k,
            axis=0,
        ),
        actions[start:end],
    )

    np.testing.assert_array_equal(
        poisoned[:9],
        actions[:9],
    )

    np.testing.assert_array_equal(
        poisoned[15:],
        actions[15:],
    )


def test_validator_accepts_valid_artifact():
    actions, rewards, trajectories = synthetic()

    returns = trajectory_returns(
        rewards,
        trajectories,
    )

    selected = [1, 2]

    poisoned, _ = apply_action_shift(
        actions,
        trajectories,
        selected,
        attack_seed=20,
    )

    result = validate(
        actions,
        poisoned,
        trajectories,
        selected,
        returns=returns,
        threshold=10.0,
        min_length=4,
        used_transitions=15,
        requested_budget=11,
        actual_budget=11,
        min_util=0.98,
        attack_seed=20,
    )

    assert (
        result[
            "action_multiset_preserved_per_selected_trajectory"
        ]
        is True
    )
    assert (
        result[
            "unselected_actions_identical"
        ]
        is True
    )
    assert (
        result[
            "actual_transition_budget"
        ]
        == 11
    )


def test_writer_changes_only_actions(tmp_path):
    clean = tmp_path / "clean.hdf5"
    poison = tmp_path / "poison.hdf5"

    string_dtype = h5py.string_dtype(
        encoding="utf-8"
    )

    with h5py.File(clean, "w") as h:
        ds = h.create_dataset(
            "actions",
            data=np.asarray(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ],
                dtype=np.float32,
            ),
            compression="gzip",
        )
        ds.attrs["meaning"] = "keep"

        h.create_dataset(
            "rewards",
            data=np.asarray(
                [1.0, 2.0],
                dtype=np.float32,
            ),
        )

        g = h.create_group("metadata")
        g.attrs["group_attr"] = "keep"
        g.create_dataset(
            "strings",
            data=np.asarray(
                ["walker2d", "medium-v2"],
                dtype=object,
            ),
            dtype=string_dtype,
        )
        h.attrs["root_attr"] = "keep"

    poisoned_actions = np.asarray(
        [
            [3.0, 4.0],
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )

    write_action_only(
        clean,
        poison,
        poisoned_actions,
    )

    with h5py.File(clean, "r") as a, h5py.File(poison, "r") as b:
        np.testing.assert_array_equal(
            b["actions"][:],
            poisoned_actions,
        )
        np.testing.assert_array_equal(
            b["rewards"][:],
            a["rewards"][:],
        )
        np.testing.assert_array_equal(
            b["metadata/strings"][:],
            a["metadata/strings"][:],
        )

        assert (
            b["actions"].compression
            == a["actions"].compression
        )
        assert (
            b["actions"].attrs["meaning"]
            == a["actions"].attrs["meaning"]
        )
        assert (
            b["metadata"].attrs["group_attr"]
            == a["metadata"].attrs["group_attr"]
        )
        assert (
            b.attrs["root_attr"]
            == a.attrs["root_attr"]
        )
