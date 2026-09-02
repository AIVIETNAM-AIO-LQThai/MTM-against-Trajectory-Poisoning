import h5py
import numpy as np

from src.data.hdf5_io import (
    load_hdf5_dataset,
    write_hdf5_dataset,
)


def _dataset():
    return {
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

        "infos/qpos": np.arange(
            30,
            dtype=np.float64,
        ).reshape(10, 3),

        "infos/qvel": np.arange(
            20,
            dtype=np.float64,
        ).reshape(10, 2),
    }


def test_hdf5_roundtrip(
    tmp_path,
):
    dataset = _dataset()

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


def test_nested_paths_are_written_as_groups(
    tmp_path,
):
    dataset = _dataset()

    path = (
        tmp_path
        / "dataset.hdf5"
    )

    write_hdf5_dataset(
        path,
        dataset,
    )

    with h5py.File(
        path,
        "r",
    ) as handle:

        assert "infos" in handle

        assert isinstance(
            handle["infos"],
            h5py.Group,
        )

        assert (
            "infos/qpos"
            in handle
        )

        assert (
            "infos/qvel"
            in handle
        )


def test_nested_arrays_survive_exactly(
    tmp_path,
):
    dataset = _dataset()

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

    np.testing.assert_array_equal(
        loaded["infos/qpos"],
        dataset["infos/qpos"],
    )

    np.testing.assert_array_equal(
        loaded["infos/qvel"],
        dataset["infos/qvel"],
    )