from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

import h5py
import numpy as np

from src.data.normalization import compute_state_statistics
from src.data.trajectories import find_completed_trajectories


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "attack_qualification"
    / "rdt_source_random_state_corruption.json"
)

DEFAULT_DATA_ROOT = (
    ROOT
    / "data"
    / "derived"
    / "rdt_source_random_state_corruption"
    / "walker2d-medium-v2"
)

DEFAULT_METADATA_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "rdt_source_random_state_corruption"
    / "walker2d-medium-v2"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    cfg = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if (
        cfg.get("schema_version")
        != "rdt-source-random-state-corruption-v1"
    ):
        raise RuntimeError(
            "unexpected protocol schema"
        )

    if cfg.get("status") != "predeclared":
        raise RuntimeError(
            "protocol is not predeclared"
        )

    if (
        cfg.get("name")
        != "rdt_source_random_state_corruption_data_scarce"
    ):
        raise RuntimeError(
            "unexpected experiment name"
        )

    down = cfg["downsampling"]
    corruption = cfg["corruption"]

    frozen = [
        (
            float(down["ratio"]),
            0.02,
            "downsampling.ratio",
        ),
        (
            int(down["seed"]),
            1234,
            "downsampling.seed",
        ),
        (
            int(
                down[
                    "expected_selected_trajectory_count"
                ]
            ),
            23,
            "downsampling.expected_selected_trajectory_count",
        ),
        (
            corruption["field"],
            "observations",
            "corruption.field",
        ),
        (
            corruption["mode"],
            "random",
            "corruption.mode",
        ),
        (
            float(
                corruption[
                    "corruption_rate"
                ]
            ),
            0.30,
            "corruption.corruption_rate",
        ),
        (
            float(
                corruption[
                    "corruption_scale"
                ]
            ),
            1.0,
            "corruption.corruption_scale",
        ),
        (
            [
                int(x)
                for x in corruption[
                    "corruption_seeds"
                ]
            ],
            [2023, 2024, 2025],
            "corruption.corruption_seeds",
        ),
    ]

    for actual, expected, name in frozen:
        if actual != expected:
            raise RuntimeError(
                f"frozen field changed: {name}: "
                f"{actual!r} != {expected!r}"
            )

    return cfg


def select_trajectory_indices(
    num_trajectories: int,
    *,
    ratio: float,
    seed: int,
) -> list[int]:
    count = int(
        num_trajectories
        * ratio
    )

    rng = random.Random(
        int(seed)
    )

    # This exactly mirrors:
    # random.seed(seed)
    # random.sample(traj, int(len(traj) * ratio))
    # while keeping trajectory indices as the representation.
    return rng.sample(
        list(
            range(
                num_trajectories
            )
        ),
        count,
    )


def selected_global_indices(
    trajectories,
    selected_trajectory_indices,
) -> np.ndarray:
    chunks = []

    for trajectory_index in (
        selected_trajectory_indices
    ):
        trajectory = trajectories[
            int(
                trajectory_index
            )
        ]

        chunks.append(
            np.arange(
                trajectory.start,
                trajectory.end,
                dtype=np.int64,
            )
        )

    if not chunks:
        raise RuntimeError(
            "downsample selected zero trajectories"
        )

    return np.concatenate(
        chunks
    )


def copy_attributes(
    source,
    destination,
):
    for key, value in source.attrs.items():
        destination.attrs[
            key
        ] = value


def create_downsampled_hdf5(
    *,
    source_path: Path,
    destination_path: Path,
    row_indices: np.ndarray,
    source_transition_count: int,
):
    if destination_path.exists():
        raise FileExistsError(
            destination_path
        )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(
        source_path,
        "r",
    ) as source, h5py.File(
        destination_path,
        "w",
    ) as destination:
        copy_attributes(
            source,
            destination,
        )

        def copy_group(
            src_group,
            dst_group,
        ):
            for name, item in (
                src_group.items()
            ):
                if isinstance(
                    item,
                    h5py.Group,
                ):
                    new_group = (
                        dst_group.create_group(
                            name
                        )
                    )

                    copy_attributes(
                        item,
                        new_group,
                    )

                    copy_group(
                        item,
                        new_group,
                    )

                    continue

                if not isinstance(
                    item,
                    h5py.Dataset,
                ):
                    raise TypeError(
                        "unsupported HDF5 object: "
                        f"{item.name}"
                    )

                if (
                    item.ndim >= 1
                    and item.shape[0]
                    == source_transition_count
                ):
                    # h5py fancy indexing requires monotonically
                    # increasing indices. The source-defined
                    # random.sample trajectory order is intentionally
                    # non-monotonic, so first materialize the source
                    # dataset and then apply NumPy indexing. NumPy
                    # preserves the exact sampled trajectory order.
                    full_data = item[()]
                    data = full_data[
                        row_indices
                    ]
                else:
                    data = item[()]

                kwargs = {}

                if item.compression is not None:
                    kwargs[
                        "compression"
                    ] = item.compression

                if (
                    item.compression_opts
                    is not None
                ):
                    kwargs[
                        "compression_opts"
                    ] = (
                        item.compression_opts
                    )

                if (
                    item.shuffle
                    is not None
                ):
                    kwargs[
                        "shuffle"
                    ] = item.shuffle

                if (
                    item.fletcher32
                    is not None
                ):
                    kwargs[
                        "fletcher32"
                    ] = item.fletcher32

                # Source chunk shapes can be invalid after
                # downsampling, so let h5py choose chunks.
                if (
                    item.chunks is not None
                    and np.asarray(
                        data
                    ).ndim > 0
                ):
                    kwargs[
                        "chunks"
                    ] = True

                new_dataset = (
                    dst_group.create_dataset(
                        name,
                        data=data,
                        dtype=item.dtype,
                        **kwargs,
                    )
                )

                copy_attributes(
                    item,
                    new_dataset,
                )

        copy_group(
            source,
            destination,
        )


def generate_random_state_corruption(
    *,
    clean_observations: np.ndarray,
    corruption_seed: int,
    corruption_rate: float,
    corruption_scale: float,
):
    observations = np.asarray(
        clean_observations
    )

    rng = np.random.RandomState(
        int(
            corruption_seed
        )
    )

    random_num = rng.random(
        len(
            observations
        )
    )

    attacked_indices = np.where(
        random_num
        < float(
            corruption_rate
        )
    )[0].astype(
        np.int64
    )

    original = observations[
        attacked_indices
    ].copy()

    observation_std = np.std(
        observations,
        axis=0,
        keepdims=True,
    )

    # Important source-semantic detail:
    # use the SAME RandomState after index selection.
    random_noise = rng.uniform(
        -float(
            corruption_scale
        ),
        float(
            corruption_scale
        ),
        size=original.shape,
    )

    attacked = (
        original
        + random_noise
        * observation_std
    )

    poisoned = observations.copy()

    # Assignment into the original dtype reproduces the cast
    # that occurs when the public attack writes back into the
    # dataset array.
    poisoned[
        attacked_indices
    ] = attacked

    return (
        poisoned,
        attacked_indices,
        observation_std.reshape(
            -1
        ),
    )


def collect_hdf5_paths(
    handle,
):
    objects = {
        "/": "group"
    }

    def visitor(
        name,
        obj,
    ):
        objects[
            "/" + name
        ] = (
            "dataset"
            if isinstance(
                obj,
                h5py.Dataset,
            )
            else "group"
        )

    handle.visititems(
        visitor
    )

    return objects


def attribute_dict(obj):
    return {
        str(key): np.asarray(
            value
        )
        for key, value
        in obj.attrs.items()
    }


def assert_attributes_equal(
    left,
    right,
    path,
):
    a = attribute_dict(
        left
    )

    b = attribute_dict(
        right
    )

    if set(
        a
    ) != set(
        b
    ):
        raise RuntimeError(
            f"attribute keys changed: {path}"
        )

    for key in a:
        if (
            a[key].shape
            != b[key].shape
            or not np.array_equal(
                a[key],
                b[key],
            )
        ):
            raise RuntimeError(
                "attribute changed: "
                f"{path}:{key}"
            )


def verify_observation_only_difference(
    clean_path: Path,
    corrupted_path: Path,
    attacked_indices: np.ndarray,
):
    with h5py.File(
        clean_path,
        "r",
    ) as clean, h5py.File(
        corrupted_path,
        "r",
    ) as corrupted:
        clean_paths = (
            collect_hdf5_paths(
                clean
            )
        )

        corrupted_paths = (
            collect_hdf5_paths(
                corrupted
            )
        )

        if (
            clean_paths
            != corrupted_paths
        ):
            raise RuntimeError(
                "corrupted HDF5 hierarchy changed"
            )

        assert_attributes_equal(
            clean,
            corrupted,
            "/",
        )

        changed_rows = None

        for path, kind in (
            clean_paths.items()
        ):
            if path == "/":
                continue

            name = path.lstrip(
                "/"
            )

            left = clean[
                name
            ]

            right = corrupted[
                name
            ]

            assert_attributes_equal(
                left,
                right,
                path,
            )

            if kind == "group":
                continue

            if (
                left.shape
                != right.shape
                or left.dtype
                != right.dtype
            ):
                raise RuntimeError(
                    "dataset schema changed: "
                    f"{path}"
                )

            left_data = left[()]
            right_data = right[()]

            if path == "/observations":
                row_change = np.any(
                    left_data
                    != right_data,
                    axis=1,
                )

                changed_rows = np.where(
                    row_change
                )[0].astype(
                    np.int64
                )

                continue

            if not np.array_equal(
                left_data,
                right_data,
            ):
                raise RuntimeError(
                    "non-observation dataset changed: "
                    f"{path}"
                )

        if changed_rows is None:
            raise RuntimeError(
                "observations dataset not found"
            )

        if not np.array_equal(
            changed_rows,
            attacked_indices,
        ):
            raise RuntimeError(
                "changed observation rows do not "
                "match attacked indices"
            )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    cfg = load_config(
        args.config
    )

    source_path = (
        ROOT
        / cfg[
            "dataset"
        ][
            "path"
        ]
    ).resolve()

    if (
        sha256_file(
            source_path
        )
        != cfg[
            "dataset"
        ][
            "sha256"
        ]
    ):
        raise RuntimeError(
            "source dataset SHA256 mismatch"
        )

    with h5py.File(
        source_path,
        "r",
    ) as source:
        terminals = np.asarray(
            source[
                "terminals"
            ],
            dtype=bool,
        )

        timeouts = np.asarray(
            source[
                "timeouts"
            ],
            dtype=bool,
        )

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    used_transitions = int(
        sum(
            trajectory.length
            for trajectory
            in trajectories
        )
    )

    dataset_cfg = cfg[
        "dataset"
    ]

    if (
        len(
            trajectories
        )
        != int(
            dataset_cfg[
                "completed_trajectories"
            ]
        )
        or used_transitions
        != int(
            dataset_cfg[
                "used_transitions"
            ]
        )
        or trailing
        != int(
            dataset_cfg[
                "trailing_transitions"
            ]
        )
    ):
        raise RuntimeError(
            "source trajectory contract changed"
        )

    down = cfg[
        "downsampling"
    ]

    selected_trajectories = (
        select_trajectory_indices(
            len(
                trajectories
            ),
            ratio=float(
                down[
                    "ratio"
                ]
            ),
            seed=int(
                down[
                    "seed"
                ]
            ),
        )
    )

    if (
        len(
            selected_trajectories
        )
        != int(
            down[
                "expected_selected_trajectory_count"
            ]
        )
    ):
        raise RuntimeError(
            "selected trajectory count mismatch"
        )

    row_indices = selected_global_indices(
        trajectories,
        selected_trajectories,
    )

    clean_path = (
        args.data_root
        / "clean_ratio_0p02.hdf5"
    )

    create_downsampled_hdf5(
        source_path=source_path,
        destination_path=clean_path,
        row_indices=row_indices,
        source_transition_count=len(
            terminals
        ),
    )

    with h5py.File(
        clean_path,
        "r",
    ) as clean:
        clean_observations = np.asarray(
            clean[
                "observations"
            ]
        )

        clean_actions = np.asarray(
            clean[
                "actions"
            ]
        )

        clean_rewards = np.asarray(
            clean[
                "rewards"
            ]
        )

        clean_terminals = np.asarray(
            clean[
                "terminals"
            ],
            dtype=bool,
        )

        clean_timeouts = np.asarray(
            clean[
                "timeouts"
            ],
            dtype=bool,
        )

    clean_trajectories, clean_trailing = (
        find_completed_trajectories(
            clean_terminals,
            clean_timeouts,
        )
    )

    clean_transition_count = int(
        len(
            clean_observations
        )
    )

    if (
        len(
            clean_trajectories
        )
        != len(
            selected_trajectories
        )
        or clean_trailing
        != 0
    ):
        raise RuntimeError(
            "downsampled trajectory contract failed"
        )

    if (
        len(
            clean_actions
        )
        != clean_transition_count
        or len(
            clean_rewards
        )
        != clean_transition_count
    ):
        raise RuntimeError(
            "downsampled arrays have unequal length"
        )

    state_mean, state_std = (
        compute_state_statistics(
            clean_observations,
            clean_trajectories,
        )
    )

    args.metadata_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    normalization_path = (
        args.metadata_root
        / "clean_ratio_0p02_normalization.npz"
    )

    np.savez(
        normalization_path,
        state_mean=state_mean,
        state_std=state_std,
        epsilon=np.array(
            1e-6
        ),
        num_training_transitions=np.array(
            clean_transition_count
        ),
        num_trajectories=np.array(
            len(
                clean_trajectories
            )
        ),
        trailing_transitions=np.array(
            clean_trailing
        ),
    )

    selection_path = (
        args.metadata_root
        / "clean_ratio_0p02_selection.npz"
    )

    np.savez_compressed(
        selection_path,
        selected_trajectory_indices=np.asarray(
            selected_trajectories,
            dtype=np.int64,
        ),
        source_global_transition_indices=(
            row_indices
        ),
    )

    clean_meta = {
        "source_dataset_sha256": (
            cfg[
                "dataset"
            ][
                "sha256"
            ]
        ),
        "downsample_seed": int(
            down[
                "seed"
            ]
        ),
        "downsample_ratio": float(
            down[
                "ratio"
            ]
        ),
        "selected_trajectory_count": int(
            len(
                selected_trajectories
            )
        ),
        "selected_trajectory_indices": [
            int(x)
            for x
            in selected_trajectories
        ],
        "transition_count": (
            clean_transition_count
        ),
        "trailing_transitions": (
            clean_trailing
        ),
        "clean_dataset": str(
            clean_path.resolve()
        ),
        "clean_dataset_sha256": (
            sha256_file(
                clean_path
            )
        ),
        "normalization_file": str(
            normalization_path.resolve()
        ),
        "normalization_sha256": (
            sha256_file(
                normalization_path
            )
        ),
        "selection_file": str(
            selection_path.resolve()
        ),
        "selection_sha256": (
            sha256_file(
                selection_path
            )
        ),
    }

    (
        args.metadata_root
        / "clean_ratio_0p02.json"
    ).write_text(
        json.dumps(
            clean_meta,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=" * 104
    )

    print(
        "RDT-SOURCE DATA-SCARCE RANDOM STATE CORRUPTION"
    )

    print(
        "=" * 104
    )

    print(
        f"source trajectories: {len(trajectories)}"
    )

    print(
        "selected trajectories: "
        f"{len(selected_trajectories)}"
    )

    print(
        "selected trajectory indices:"
    )

    print(
        selected_trajectories
    )

    print(
        "clean downsampled transitions: "
        f"{clean_transition_count}"
    )

    print(
        "clean downsampled trailing: "
        f"{clean_trailing}"
    )

    print(
        "clean SHA256: "
        f"{clean_meta['clean_dataset_sha256']}"
    )

    print(
        "normalization SHA256: "
        f"{clean_meta['normalization_sha256']}"
    )

    print()

    corrupted_root = (
        args.data_root
        / "corrupted"
    )

    corrupted_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    corruption = cfg[
        "corruption"
    ]

    for corruption_seed in (
        corruption[
            "corruption_seeds"
        ]
    ):
        seed = int(
            corruption_seed
        )

        (
            corrupted_observations,
            attacked_indices,
            observation_std,
        ) = (
            generate_random_state_corruption(
                clean_observations=(
                    clean_observations
                ),
                corruption_seed=seed,
                corruption_rate=float(
                    corruption[
                        "corruption_rate"
                    ]
                ),
                corruption_scale=float(
                    corruption[
                        "corruption_scale"
                    ]
                ),
            )
        )

        corrupted_path = (
            corrupted_root
            / (
                f"corruption_seed_"
                f"{seed}.hdf5"
            )
        )

        if corrupted_path.exists():
            raise FileExistsError(
                corrupted_path
            )

        shutil.copy2(
            clean_path,
            corrupted_path,
        )

        try:
            with h5py.File(
                corrupted_path,
                "r+",
            ) as handle:
                observations_dataset = (
                    handle[
                        "observations"
                    ]
                )

                if (
                    observations_dataset.shape
                    != corrupted_observations.shape
                    or observations_dataset.dtype
                    != corrupted_observations.dtype
                ):
                    raise RuntimeError(
                        "observation schema changed"
                    )

                observations_dataset[
                    ...
                ] = corrupted_observations

                handle.flush()

        except Exception:
            corrupted_path.unlink(
                missing_ok=True
            )
            raise

        verify_observation_only_difference(
            clean_path,
            corrupted_path,
            attacked_indices,
        )

        record_path = (
            args.metadata_root
            / (
                f"corruption_seed_"
                f"{seed}_records.npz"
            )
        )

        np.savez_compressed(
            record_path,
            attacked_transition_indices=(
                attacked_indices
            ),
            clean_observation_std=(
                observation_std
            ),
        )

        changed_fraction = float(
            len(
                attacked_indices
            )
            / clean_transition_count
        )

        delta = (
            corrupted_observations[
                attacked_indices
            ].astype(
                np.float64
            )
            - clean_observations[
                attacked_indices
            ].astype(
                np.float64
            )
        )

        scaled = (
            delta
            / observation_std[
                None,
                :
            ]
        )

        row = {
            "corruption_seed": seed,
            "corruption_rate": float(
                corruption[
                    "corruption_rate"
                ]
            ),
            "corruption_scale": float(
                corruption[
                    "corruption_scale"
                ]
            ),
            "transition_count": (
                clean_transition_count
            ),
            "attacked_transition_count": int(
                len(
                    attacked_indices
                )
            ),
            "attacked_fraction": (
                changed_fraction
            ),
            "corrupted_dataset": str(
                corrupted_path.resolve()
            ),
            "corrupted_dataset_sha256": (
                sha256_file(
                    corrupted_path
                )
            ),
            "records_file": str(
                record_path.resolve()
            ),
            "records_file_sha256": (
                sha256_file(
                    record_path
                )
            ),
            "scaled_noise_abs": {
                "mean": float(
                    np.mean(
                        np.abs(
                            scaled
                        )
                    )
                ),
                "max": float(
                    np.max(
                        np.abs(
                            scaled
                        )
                    )
                ),
            },
            "integrity": {
                "only_observations_changed": True,
                "changed_rows_equal_attacked_indices": True,
                "actions_unchanged": True,
                "rewards_unchanged": True,
                "terminals_unchanged": True,
                "timeouts_unchanged": True,
                "same_clean_normalization_required": True,
            },
        }

        (
            args.metadata_root
            / (
                f"corruption_seed_"
                f"{seed}.json"
            )
        ).write_text(
            json.dumps(
                row,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        rows.append(
            row
        )

        print(
            f"seed={seed} "
            f"attacked="
            f"{len(attacked_indices)}/"
            f"{clean_transition_count} "
            f"fraction="
            f"{changed_fraction:.6f} "
            f"scaled|noise| mean="
            f"{row['scaled_noise_abs']['mean']:.6f} "
            f"max="
            f"{row['scaled_noise_abs']['max']:.6f}"
        )

    summary = {
        "schema_version": (
            "rdt-source-random-state-corruption-artifacts-v1"
        ),
        "status": (
            "ARTIFACT_GENERATION_PASS"
        ),
        "experiment": cfg[
            "name"
        ],
        "source_repository_commit": (
            cfg[
                "source"
            ][
                "repository_commit"
            ]
        ),
        "clean": clean_meta,
        "corrupted": rows,
        "claim_boundary": cfg[
            "claim_boundary"
        ],
    }

    (
        args.metadata_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()

    print(
        "RDT-SOURCE RANDOM STATE CORRUPTION "
        "ARTIFACT GENERATION: PASS"
    )

    print(
        "metadata ->",
        args.metadata_root,
    )


if __name__ == "__main__":
    main()
