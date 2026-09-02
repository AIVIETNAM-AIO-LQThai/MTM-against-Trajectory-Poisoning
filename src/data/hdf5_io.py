from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


def load_hdf5_dataset(
    path: str | Path,
) -> dict[str, np.ndarray]:
    """
    Load every HDF5 dataset recursively.

    Nested HDF5 datasets are represented internally using their
    slash-separated HDF5 paths, for example:

        infos/qpos
        infos/qvel

    HDF5 groups themselves are not returned as dataset entries.
    """

    path = Path(path)

    loaded = {}

    with h5py.File(
        path,
        "r",
    ) as handle:

        def collect(
            name,
            obj,
        ):
            if isinstance(
                obj,
                h5py.Dataset,
            ):
                loaded[name] = obj[()]

        handle.visititems(
            collect
        )

    return {
        key: loaded[key]
        for key in sorted(
            loaded
        )
    }


def write_hdf5_dataset(
    path: str | Path,
    dataset: Mapping[
        str,
        np.ndarray,
    ],
) -> None:
    """
    Write a flat mapping of arrays to HDF5.

    Keys may contain slash-separated paths such as:

        infos/qpos

    h5py recreates the corresponding HDF5 group hierarchy.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = (
        path.parent
        / f".{path.name}.tmp"
    )

    with h5py.File(
        temporary_path,
        "w",
    ) as handle:

        for key in sorted(
            dataset
        ):
            if not key:
                raise ValueError(
                    "HDF5 dataset key cannot be empty"
                )

            if key.startswith("/"):
                raise ValueError(
                    "HDF5 dataset keys must be relative"
                )

            handle.create_dataset(
                key,
                data=np.asarray(
                    dataset[key]
                ),
            )

    os.replace(
        temporary_path,
        path,
    )