import copy

import numpy as np

from scripts.e1b_min_overlap_core import (
    contiguous_runs,
    ordered_run_lengths,
    minimum_overlap_layout,
    build_min_overlap_control,
    validate_min_overlap_control,
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
    values = np.asarray(
        [1, 2, 5, 8, 9, 10],
        dtype=np.int64,
    )

    assert [
        x.tolist()
        for x in contiguous_runs(values)
    ] == [
        [1, 2],
        [5],
        [8, 9, 10],
    ]


def test_ordered_run_lengths():
    mask = np.zeros(
        12,
        dtype=bool,
    )
    mask[[1, 2, 5, 8, 9, 10]] = True

    assert ordered_run_lengths(
        mask
    ) == [2, 1, 3]


def test_dp_finds_zero_overlap_when_possible():
    source = np.zeros(
        20,
        dtype=bool,
    )
    source[2:4] = True
    source[8:9] = True

    result = minimum_overlap_layout(
        source,
        [2, 1],
        tie_seed=123,
    )

    assert result[
        "overlap_count"
    ] == 0


def test_dp_handles_dense_layout():
    source = np.zeros(
        30,
        dtype=bool,
    )

    source[0:5] = True
    source[7:12] = True
    source[14:19] = True
    source[21:26] = True

    result = minimum_overlap_layout(
        source,
        [5, 5, 5, 5],
        tie_seed=321,
    )

    starts = result["starts"]

    assert len(starts) == 4

    for j in range(1, len(starts)):
        assert (
            starts[j]
            >= starts[j - 1] + 6
        )


def test_control_is_deterministic():
    clean, poison = fake_dataset()

    first, first_records = (
        build_min_overlap_control(
            clean,
            poison,
            used_n=40,
            replicate=0,
            artifact_seed=7,
        )
    )

    second, second_records = (
        build_min_overlap_control(
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

    assert (
        first_records
        == second_records
    )


def test_all_five_replicates_validate():
    clean, poison = fake_dataset()

    for replicate in range(5):
        control, records = (
            build_min_overlap_control(
                clean,
                poison,
                used_n=40,
                replicate=replicate,
                artifact_seed=7,
            )
        )

        result = (
            validate_min_overlap_control(
                clean,
                poison,
                control,
                records,
                used_n=40,
            )
        )

        assert (
            result[
                "source_modified_count"
            ]
            == 6
        )

        assert (
            result[
                "target_modified_count"
            ]
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
            result[
                "max_recovered_delta_error"
            ]
            <= 1e-5
        )
