from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.preflight_a2_dt_attack_qualification import (
    _assert_reward_only_hdf5_difference,
)


def _make_clean(path: Path):
    with h5py.File(
        path,
        "w",
    ) as handle:
        handle.attrs[
            "root_name"
        ] = "walker2d"

        handle.create_dataset(
            "observations",
            data=np.arange(
                12,
                dtype=np.float32,
            ).reshape(6, 2),
            compression="gzip",
        )

        handle.create_dataset(
            "rewards",
            data=np.arange(
                6,
                dtype=np.float32,
            ),
        )

        group = handle.create_group(
            "metadata"
        )

        group.attrs[
            "kind"
        ] = "fixture"

        string_dtype = (
            h5py.string_dtype(
                encoding="utf-8"
            )
        )

        group.create_dataset(
            "names",
            data=np.asarray(
                [
                    "alpha",
                    "beta",
                ],
                dtype=object,
            ),
            dtype=string_dtype,
        )


def _copy_as_poison(
    clean_path,
    poison_path,
):
    import shutil

    shutil.copy2(
        clean_path,
        poison_path,
    )

    with h5py.File(
        poison_path,
        "r+",
    ) as handle:
        rewards = handle[
            "rewards"
        ]

        rewards[...] = (
            rewards[...]
            + 1.0
        )


def test_recursive_hdf5_preflight_accepts_groups_and_strings(
    tmp_path,
):
    clean_path = (
        tmp_path
        / "clean.hdf5"
    )

    poison_path = (
        tmp_path
        / "poison.hdf5"
    )

    _make_clean(
        clean_path
    )

    _copy_as_poison(
        clean_path,
        poison_path,
    )

    _assert_reward_only_hdf5_difference(
        clean_path,
        poison_path,
    )


def test_recursive_hdf5_preflight_rejects_nonreward_change(
    tmp_path,
):
    clean_path = (
        tmp_path
        / "clean.hdf5"
    )

    poison_path = (
        tmp_path
        / "poison.hdf5"
    )

    _make_clean(
        clean_path
    )

    _copy_as_poison(
        clean_path,
        poison_path,
    )

    with h5py.File(
        poison_path,
        "r+",
    ) as handle:
        handle[
            "observations"
        ][0, 0] += 1.0

    with pytest.raises(
        RuntimeError,
        match="non-reward dataset differs",
    ):
        _assert_reward_only_hdf5_difference(
            clean_path,
            poison_path,
        )


def test_recursive_hdf5_preflight_rejects_group_attribute_change(
    tmp_path,
):
    clean_path = (
        tmp_path
        / "clean.hdf5"
    )

    poison_path = (
        tmp_path
        / "poison.hdf5"
    )

    _make_clean(
        clean_path
    )

    _copy_as_poison(
        clean_path,
        poison_path,
    )

    with h5py.File(
        poison_path,
        "r+",
    ) as handle:
        handle[
            "metadata"
        ].attrs[
            "kind"
        ] = "changed"

    with pytest.raises(
        RuntimeError,
        match="attribute value changed",
    ):
        _assert_reward_only_hdf5_difference(
            clean_path,
            poison_path,
        )


def test_recursive_hdf5_preflight_rejects_hierarchy_change(
    tmp_path,
):
    clean_path = (
        tmp_path
        / "clean.hdf5"
    )

    poison_path = (
        tmp_path
        / "poison.hdf5"
    )

    _make_clean(
        clean_path
    )

    _copy_as_poison(
        clean_path,
        poison_path,
    )

    with h5py.File(
        poison_path,
        "r+",
    ) as handle:
        handle.create_group(
            "extra"
        )

    with pytest.raises(
        RuntimeError,
        match="HDF5 hierarchy changed",
    ):
        _assert_reward_only_hdf5_difference(
            clean_path,
            poison_path,
        )
