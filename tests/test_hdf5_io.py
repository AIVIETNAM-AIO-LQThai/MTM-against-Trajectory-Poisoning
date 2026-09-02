import numpy as np

from src.data.hdf5_io import (
    load_hdf5_dataset,
    write_hdf5_dataset,
)


def test_hdf5_roundtrip(
    tmp_path,
):
    dataset = {
        "observations": np.arange(
            20,
            dtype=np.float32,
        ).reshape(10, 2),
        "actions": np.arange(
            10,
            dtype=np.float32,
        ).reshape(10, 1),
        "rewards": np.arange(
            10,
            dtype=np.float32,
        ),
        "terminals": np.zeros(
            10,
            dtype=bool,
        ),
        "timeouts": np.zeros(
            10,
            dtype=bool,
        ),
    }

    path = (
        tmp_path
        / "dataset.hdf5"
    )

    write_hdf5_dataset(
        path,
        dataset,
    )

    loaded = load_hdf5_dataset(
        path
    )

    assert set(loaded) == set(
        dataset
    )

    for key in dataset:
        np.testing.assert_array_equal(
            loaded[key],
            dataset[key],
        )

        assert (
            loaded[key].dtype
            == dataset[key].dtype
        )