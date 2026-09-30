from __future__ import annotations

import random

import numpy as np
import h5py

from scripts.generate_rdt_source_random_state_corruption import (
    create_downsampled_hdf5,
    generate_random_state_corruption,
    select_trajectory_indices,
)


def test_source_downsampling_matches_python_random_sample():
    actual = select_trajectory_indices(
        1190,
        ratio=0.02,
        seed=1234,
    )

    random.seed(
        1234
    )

    expected = random.sample(
        list(
            range(
                1190
            )
        ),
        int(
            1190
            * 0.02
        ),
    )

    assert actual == expected
    assert len(actual) == 23


def test_random_state_corruption_uses_one_continuous_rng_stream():
    observations = np.asarray(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 8.0],
            [4.0, 16.0],
            [5.0, 32.0],
            [6.0, 64.0],
        ],
        dtype=np.float32,
    )

    seed = 2023
    rate = 0.30
    scale = 1.0

    (
        poisoned,
        attacked,
        observed_std,
    ) = generate_random_state_corruption(
        clean_observations=(
            observations
        ),
        corruption_seed=seed,
        corruption_rate=rate,
        corruption_scale=scale,
    )

    rng = np.random.RandomState(
        seed
    )

    random_num = rng.random(
        len(
            observations
        )
    )

    expected_attacked = np.where(
        random_num < rate
    )[0].astype(
        np.int64
    )

    expected_std = np.std(
        observations,
        axis=0,
        keepdims=True,
    )

    original = observations[
        expected_attacked
    ].copy()

    noise = rng.uniform(
        -scale,
        scale,
        size=original.shape,
    )

    expected_values = (
        original
        + noise
        * expected_std
    )

    expected_poisoned = (
        observations.copy()
    )

    expected_poisoned[
        expected_attacked
    ] = expected_values

    np.testing.assert_array_equal(
        attacked,
        expected_attacked,
    )

    np.testing.assert_array_equal(
        observed_std,
        expected_std.reshape(
            -1
        ),
    )

    np.testing.assert_array_equal(
        poisoned,
        expected_poisoned,
    )


def test_unattacked_rows_are_unchanged():
    observations = np.arange(
        200,
        dtype=np.float32,
    ).reshape(
        100,
        2,
    )

    (
        poisoned,
        attacked,
        _,
    ) = generate_random_state_corruption(
        clean_observations=(
            observations
        ),
        corruption_seed=2024,
        corruption_rate=0.30,
        corruption_scale=1.0,
    )

    mask = np.zeros(
        len(
            observations
        ),
        dtype=bool,
    )

    mask[
        attacked
    ] = True

    np.testing.assert_array_equal(
        poisoned[
            ~mask
        ],
        observations[
            ~mask
        ],
    )


def test_downsample_hdf5_preserves_nonmonotonic_sample_order(tmp_path):
    source = tmp_path / "source.hdf5"
    destination = tmp_path / "downsampled.hdf5"

    observations = np.arange(
        12,
        dtype=np.float32,
    ).reshape(
        6,
        2,
    )

    actions = np.arange(
        6,
        dtype=np.float32,
    ).reshape(
        6,
        1,
    )

    with h5py.File(
        source,
        "w",
    ) as handle:
        handle.create_dataset(
            "observations",
            data=observations,
        )
        handle.create_dataset(
            "actions",
            data=actions,
        )
        handle.create_dataset(
            "rewards",
            data=np.arange(
                6,
                dtype=np.float32,
            ),
        )

    row_indices = np.asarray(
        [4, 5, 0, 1],
        dtype=np.int64,
    )

    create_downsampled_hdf5(
        source_path=source,
        destination_path=destination,
        row_indices=row_indices,
        source_transition_count=6,
    )

    with h5py.File(
        destination,
        "r",
    ) as handle:
        np.testing.assert_array_equal(
            handle[
                "observations"
            ][:],
            observations[
                row_indices
            ],
        )

        np.testing.assert_array_equal(
            handle[
                "actions"
            ][:],
            actions[
                row_indices
            ],
        )

        np.testing.assert_array_equal(
            handle[
                "rewards"
            ][:],
            np.arange(
                6,
                dtype=np.float32,
            )[
                row_indices
            ],
        )
