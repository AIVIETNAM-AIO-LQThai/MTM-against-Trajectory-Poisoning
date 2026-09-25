import copy

import numpy as np

from scripts.analyze_e1b_circular_shift_control import (
    contiguous_runs,
    run_lengths,
    enumerate_valid_shifts,
    build_circular_shift_control,
    validate_control,
)


def fake_dataset():
    n = 40

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

    terminals[19] = True
    terminals[39] = True

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

    poison = copy.deepcopy(clean)
    poison["observations"] = observations.copy()

    modified = {
        2: [0.2, -0.4, 0.1],
        3: [-0.1, 0.3, 0.2],
        8: [0.5, 0.1, -0.2],
        24: [-0.25, 0.15, 0.4],
        25: [0.3, -0.2, 0.1],
        31: [0.1, 0.2, -0.3],
    }

    for idx, delta in modified.items():
        poison["observations"][idx] += (
            np.asarray(
                delta,
                dtype=np.float32,
            )
        )

    return clean, poison


def test_contiguous_runs():
    x = np.asarray(
        [1, 2, 5, 8, 9, 10],
        dtype=np.int64,
    )

    assert [
        run.tolist()
        for run in contiguous_runs(x)
    ] == [
        [1, 2],
        [5],
        [8, 9, 10],
    ]


def test_run_lengths():
    mask = np.zeros(
        12,
        dtype=bool,
    )

    mask[[1, 2, 5, 8, 9, 10]] = True

    assert run_lengths(mask) == [1, 2, 3]


def test_valid_shifts_preserve_run_lengths():
    mask = np.zeros(
        20,
        dtype=bool,
    )

    mask[[2, 3, 8]] = True

    shifts = enumerate_valid_shifts(
        mask,
        tie_seed=123,
    )

    assert len(shifts) >= 5

    source_lengths = run_lengths(mask)
    source_idx = np.flatnonzero(mask)

    for row in shifts:
        target = (
            source_idx
            + row["shift"]
        ) % len(mask)

        target_mask = np.zeros_like(mask)
        target_mask[target] = True

        assert run_lengths(
            target_mask
        ) == source_lengths


def test_shifts_are_ranked_by_overlap():
    mask = np.zeros(
        20,
        dtype=bool,
    )

    mask[[2, 3, 8]] = True

    shifts = enumerate_valid_shifts(
        mask,
        tie_seed=123,
    )

    overlaps = [
        row["overlap_count"]
        for row in shifts
    ]

    assert overlaps == sorted(overlaps)


def test_control_is_deterministic():
    clean, poison = fake_dataset()

    first, first_records = (
        build_circular_shift_control(
            clean,
            poison,
            used_n=40,
            replicate=0,
            artifact_seed=7,
        )
    )

    second, second_records = (
        build_circular_shift_control(
            clean,
            poison,
            used_n=40,
            replicate=0,
            artifact_seed=7,
        )
    )

    np.testing.assert_array_equal(
        first["observations"],
        second["observations"],
    )

    assert first_records == second_records


def test_control_preserves_other_modalities():
    clean, poison = fake_dataset()

    control, _ = (
        build_circular_shift_control(
            clean,
            poison,
            used_n=40,
            replicate=0,
            artifact_seed=7,
        )
    )

    np.testing.assert_array_equal(
        control["actions"],
        clean["actions"],
    )

    np.testing.assert_array_equal(
        control["rewards"],
        clean["rewards"],
    )

    np.testing.assert_array_equal(
        control["terminals"],
        clean["terminals"],
    )

    np.testing.assert_array_equal(
        control["timeouts"],
        clean["timeouts"],
    )


def test_validation_passes_all_five_replicates():
    clean, poison = fake_dataset()

    for replicate in range(5):
        control, records = (
            build_circular_shift_control(
                clean,
                poison,
                used_n=40,
                replicate=replicate,
                artifact_seed=7,
            )
        )

        result = validate_control(
            clean,
            poison,
            control,
            records,
            used_n=40,
        )

        assert (
            result["source_modified_count"]
            == 6
        )

        assert (
            result["shifted_modified_count"]
            == 6
        )

        assert (
            0.0
            <= result[
                "source_target_overlap_fraction"
            ]
            <= 1.0
        )

        assert (
            result["max_recovered_delta_error"]
            <= 1e-5
        )
