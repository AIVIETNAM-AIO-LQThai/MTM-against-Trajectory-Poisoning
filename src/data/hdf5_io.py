from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np


def load_hdf5_dataset(
    path: str | Path,
) -> dict[str, np.ndarray]:
    path = Path(path)

    dataset = {}

    with h5py.File(
        path,
        "r",
    ) as handle:
        for key in sorted(
            handle.keys()
        ):
            value = handle[key]

            if not isinstance(
                value,
                h5py.Dataset,
            ):
                raise TypeError(
                    "nested HDF5 groups are "
                    f"not supported: {key}"
                )

            dataset[key] = (
                value[()]
            )

    return dataset


def write_hdf5_dataset(
    path: str | Path,
    dataset: Mapping[
        str,
        np.ndarray,
    ],
) -> None:
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