import copy

import numpy as np

from scripts.analyze_e1b_relocated_run_control import (
    contiguous_runs,
    relocate_runs_within_trajectory,
    run_lengths_from_mask,
    validate_relocation,
)


def fake_dataset():
    n = 24

    observations = np.arange(
        n * 3,
        dtype=np.float32,
    ).reshape(n, 3)

    actions = np.zeros(
        (n, 2),
        dtype=np.float32,
    )

    rewards = np.ones(
        n,
        dtype=np.float32,
    )

    terminals = np.zeros(
        n,
        dtype=bool,
    )

    terminals[11] = True
    terminals[23] = True

    timeouts = np.zeros(
        n,
        dtype=bool,
    )

    clean = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "timeouts": timeouts,
    }

    poison = copy.deepcopy(
        clean
    )

    poison["observations"] = (
        observations.copy()
    )

    poison["observations"][1] += (
        np.asarray(
            [0.2, -0.4, 0.1],
            dtype=np.float32,
        )
    )

    poison["observations"][2] += (
        np.asarray(
            [-0.1, 0.3, 0.2],
            dtype=np.float32,
        )
    )

    poison["observations"][7] += (
        np.asarray(
            [0.5, 0.1, -0.2],
            dtype=np.float32,
        )
    )

    poison["observations"][14] += (
        np.asarray(
            [-0.25, 0.15, 0.4],
            dtype=np.float32,
        )
    )

    poison["observations"][15] += (
        np.asarray(
            [0.3, -0.2, 0.1],
            dtype=np.float32,
        )
    )

    return clean, poison


def test_contiguous_runs():
    values = np.asarray(
        [1, 2, 5, 8, 9, 10],
        dtype=np.int64,
    )

    runs = contiguous_runs(values)

    assert [
        x.tolist()
        for x in runs
    ] == [
        [1, 2],
        [5],
        [8, 9, 10],
    ]


def test_relocation_is_deterministic():
    clean, poison = fake_dataset()

    first, first_records = (
        relocate_runs_within_trajectory(
            clean,
            poison,
            used_n=24,
            seed=12345,
        )
    )

    second, second_records = (
        relocate_runs_within_trajectory(
            clean,
            poison,
            used_n=24,
            seed=12345,
        )
    )

    np.testing.assert_array_equal(
        first["observations"],
        second["observations"],
    )

    assert (
        first_records
        == second_records
    )


def test_relocation_avoids_source_positions():
    clean, poison = fake_dataset()

    control, _ = (
        relocate_runs_within_trajectory(
            clean,
            poison,
            used_n=24,
            seed=12345,
        )
    )

    source_mask = np.any(
        poison["observations"]
        != clean["observations"],
        axis=1,
    )

    target_mask = np.any(
        control["observations"]
        != clean["observations"],
        axis=1,
    )

    assert not np.any(
        source_mask & target_mask
    )


def test_relocation_preserves_per_trajectory_counts():
    clean, poison = fake_dataset()

    control, _ = (
        relocate_runs_within_trajectory(
            clean,
            poison,
            used_n=24,
            seed=12345,
        )
    )

    source_mask = np.any(
        poison["observations"]
        != clean["observations"],
        axis=1,
    )

    target_mask = np.any(
        control["observations"]
        != clean["observations"],
        axis=1,
    )

    assert (
        source_mask[:12].sum()
        == target_mask[:12].sum()
    )

    assert (
        source_mask[12:].sum()
        == target_mask[12:].sum()
    )


def test_relocation_preserves_run_lengths():
    clean, poison = fake_dataset()

    control, _ = (
        relocate_runs_within_trajectory(
            clean,
            poison,
            used_n=24,
            seed=12345,
        )
    )

    source_mask = np.any(
        poison["observations"]
        != clean["observations"],
        axis=1,
    )

    target_mask = np.any(
        control["observations"]
        != clean["observations"],
        axis=1,
    )

    assert (
        run_lengths_from_mask(
            source_mask,
            0,
            12,
        )
        == run_lengths_from_mask(
            target_mask,
            0,
            12,
        )
    )

    assert (
        run_lengths_from_mask(
            source_mask,
            12,
            24,
        )
        == run_lengths_from_mask(
            target_mask,
            12,
            24,
        )
    )


def test_relocation_transfers_exact_delta_sequences():
    clean, poison = fake_dataset()

    control, records = (
        relocate_runs_within_trajectory(
            clean,
            poison,
            used_n=24,
            seed=12345,
        )
    )

    source_delta = (
        poison["observations"]
        - clean["observations"]
    )

    target_delta = (
        control["observations"]
        - clean["observations"]
    )

    for record in records:
        s0 = record["source_start"]
        s1 = record["source_end"]
        t0 = record["target_start"]
        t1 = record["target_end"]

        expected_observation = (
            clean["observations"][t0:t1]
            + source_delta[s0:s1]
        )

        # This is the actual construction invariant.
        np.testing.assert_array_equal(
            control["observations"][t0:t1],
            expected_observation,
        )

        # Recovering the delta by subtracting float32 values
        # introduces an additional rounding step.
        np.testing.assert_allclose(
            target_delta[t0:t1],
            source_delta[s0:s1],
            atol=1e-5,
            rtol=0.0,
        )


def test_validation_passes():
    clean, poison = fake_dataset()

    control, records = (
        relocate_runs_within_trajectory(
            clean,
            poison,
            used_n=24,
            seed=12345,
        )
    )

    result = validate_relocation(
        clean,
        poison,
        control,
        records,
        used_n=24,
    )

    assert result[
        "source_modified_count"
    ] == 5

    assert result[
        "relocated_modified_count"
    ] == 5

    assert result[
        "source_target_overlap_count"
    ] == 0

    assert result[
        "per_trajectory_count_match"
    ] is True

    assert result[
        "per_trajectory_run_length_match"
    ] is True

    assert result[
        "max_delta_transfer_error"
    ] <= 1e-5
