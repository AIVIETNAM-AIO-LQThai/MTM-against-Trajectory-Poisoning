from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

from src.data.trajectories import find_completed_trajectories


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs"
    / "attack_qualification"
    / "a5_state_matched_low_return_action_substitution.json"
)

META_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "state_matched_low_return_action_substitution"
    / "walker2d-medium-v2"
)

POISON_ROOT = (
    ROOT
    / "data"
    / "poisoned"
    / "state_matched_low_return_action_substitution"
    / "walker2d-medium-v2"
)

FROZEN_CLEAN_ROOT = (
    ROOT
    / "results"
    / "attack_qualification"
    / "a2_rtg_inflation"
    / "clean"
)

OUTPUT = (
    ROOT
    / "experiments"
    / "attack_qualification"
    / "state_matched_action_substitution"
    / "preflight.json"
)

DT_REUSE_PATHS = [
    "src/methods/dt",
    "src/data/batching.py",
    "src/evaluation/walker2d.py",
    "scripts/train_dt_stress.py",
    "scripts/evaluate_dt_stress.py",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def trajectory_returns(rewards, trajectories):
    rewards = np.asarray(rewards)
    return np.asarray(
        [
            float(
                np.sum(
                    rewards[t.start:t.end],
                    dtype=np.float64,
                )
            )
            for t in trajectories
        ],
        dtype=np.float64,
    )


def transition_to_trajectory(
    trajectories,
    used_transitions,
):
    mapping = np.full(
        used_transitions,
        -1,
        dtype=np.int32,
    )
    for i, t in enumerate(trajectories):
        mapping[t.start:t.end] = i
    if np.any(mapping < 0):
        raise RuntimeError(
            "trajectory mapping incomplete"
        )
    return mapping


def collect_paths(handle):
    objects = {"/": "group"}

    def visitor(name, obj):
        objects["/" + name] = (
            "dataset"
            if isinstance(obj, h5py.Dataset)
            else "group"
        )

    handle.visititems(visitor)
    return objects


def attribute_dict(obj):
    return {
        str(key): np.asarray(value)
        for key, value
        in obj.attrs.items()
    }


def assert_attributes_equal(
    clean_obj,
    poison_obj,
    *,
    object_path,
):
    clean = attribute_dict(clean_obj)
    poison = attribute_dict(poison_obj)

    if set(clean) != set(poison):
        raise RuntimeError(
            "HDF5 attribute keys changed "
            f"at {object_path}"
        )

    for key in clean:
        if (
            clean[key].shape
            != poison[key].shape
        ):
            raise RuntimeError(
                "HDF5 attribute shape changed "
                f"at {object_path}:{key}"
            )

        if not np.array_equal(
            clean[key],
            poison[key],
        ):
            raise RuntimeError(
                "HDF5 attribute value changed "
                f"at {object_path}:{key}"
            )


def verify_action_only_hdf5(
    clean_path: Path,
    poison_path: Path,
):
    with h5py.File(
        clean_path,
        "r",
    ) as clean, h5py.File(
        poison_path,
        "r",
    ) as poison:
        clean_objects = collect_paths(clean)
        poison_objects = collect_paths(poison)

        if clean_objects != poison_objects:
            raise RuntimeError(
                "HDF5 hierarchy changed"
            )

        assert_attributes_equal(
            clean,
            poison,
            object_path="/",
        )

        action_changed = False

        for path, kind in clean_objects.items():
            if path == "/":
                continue

            name = path.lstrip("/")
            clean_obj = clean[name]
            poison_obj = poison[name]

            assert_attributes_equal(
                clean_obj,
                poison_obj,
                object_path=path,
            )

            if kind == "group":
                continue

            if (
                clean_obj.shape
                != poison_obj.shape
                or clean_obj.dtype
                != poison_obj.dtype
            ):
                raise RuntimeError(
                    "dataset schema changed: "
                    f"{path}"
                )

            clean_data = clean_obj[()]
            poison_data = poison_obj[()]

            if path == "/actions":
                action_changed = (
                    not np.array_equal(
                        clean_data,
                        poison_data,
                    )
                )
                continue

            if not np.array_equal(
                clean_data,
                poison_data,
            ):
                raise RuntimeError(
                    "non-action dataset changed: "
                    f"{path}"
                )

        if not action_changed:
            raise RuntimeError(
                "actions dataset did not change"
            )


def verify_dt_reuse() -> dict:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "a2-rtg-inflation-fail-v1",
            "--",
            *DT_REUSE_PATHS,
        ],
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "DT implementation differs from "
            "frozen clean-control runtime"
        )

    return {
        "unchanged": True,
        "baseline_tag": (
            "a2-rtg-inflation-fail-v1"
        ),
        "paths": DT_REUSE_PATHS,
    }


def verify_clean_controls(
    config: dict,
) -> dict:
    expected = {
        0: 70.97870489341976,
        1: 60.07549767262876,
        2: 68.59095661941191,
    }

    observed = {}

    for model_seed in [0, 1, 2]:
        path = (
            FROZEN_CLEAN_ROOT
            / f"model_seed_{model_seed}"
            / "eval_summary.json"
        )

        record = read_json(path)

        if (
            int(record["training_seed"])
            != model_seed
        ):
            raise RuntimeError(
                "clean training seed mismatch"
            )

        if (
            int(record["training_update"])
            != 100000
        ):
            raise RuntimeError(
                "clean training update mismatch"
            )

        if (
            int(record["num_episodes"])
            != 100
            or int(
                record["eval_seed_base"]
            )
            != 30000
        ):
            raise RuntimeError(
                "clean evaluation protocol mismatch"
            )

        value = float(
            record[
                "normalized_return_mean"
            ]
        )

        if not np.isclose(
            value,
            expected[model_seed],
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                "clean return mismatch"
            )

        observed[
            model_seed
        ] = value

    mean = float(
        np.mean(
            list(
                observed.values()
            )
        )
    )

    qualification = config[
        "qualification"
    ]

    if not np.isclose(
        mean,
        float(
            qualification[
                "clean_control_mean"
            ]
        ),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "clean-control mean mismatch"
        )

    floor = float(
        qualification[
            "gate"
        ][
            "required_mean_degradation"
        ]
    )

    if not np.isclose(
        floor,
        0.05 * mean,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "degradation floor mismatch"
        )

    return {
        "returns": {
            str(key): value
            for key, value
            in observed.items()
        },
        "mean": mean,
        "required_mean_degradation": (
            floor
        ),
    }


def verify_one_artifact(
    *,
    attack_seed: int,
    config: dict,
    clean_path: Path,
    clean_actions: np.ndarray,
    returns: np.ndarray,
    trajectories,
    transition_map: np.ndarray,
    used_transitions: int,
):
    meta_path = (
        META_ROOT
        / f"attack_seed_{attack_seed}.json"
    )

    record_path = (
        META_ROOT
        / (
            f"attack_seed_{attack_seed}"
            "_records.npz"
        )
    )

    poison_path = (
        POISON_ROOT
        / f"attack_seed_{attack_seed}.hdf5"
    )

    meta = read_json(meta_path)

    if (
        int(meta["attack_seed"])
        != attack_seed
    ):
        raise RuntimeError(
            "artifact attack seed mismatch"
        )

    if not record_path.exists():
        raise FileNotFoundError(
            record_path
        )

    if not poison_path.exists():
        raise FileNotFoundError(
            poison_path
        )

    if (
        sha256_file(
            poison_path
        )
        != meta[
            "poisoned_dataset_sha256"
        ]
    ):
        raise RuntimeError(
            "poison SHA mismatch"
        )

    if (
        sha256_file(
            record_path
        )
        != meta[
            "records_file_sha256"
        ]
    ):
        raise RuntimeError(
            "record NPZ SHA mismatch"
        )

    verify_action_only_hdf5(
        clean_path,
        poison_path,
    )

    with np.load(
        record_path
    ) as records:
        target = np.asarray(
            records[
                "target_global_index"
            ],
            dtype=np.int64,
        )

        donor = np.asarray(
            records[
                "donor_global_index"
            ],
            dtype=np.int64,
        )

        target_traj = np.asarray(
            records[
                "target_trajectory_index"
            ],
            dtype=np.int64,
        )

        donor_traj = np.asarray(
            records[
                "donor_trajectory_index"
            ],
            dtype=np.int64,
        )

        state_distance = np.asarray(
            records[
                "normalized_state_distance"
            ],
            dtype=np.float64,
        )

        action_displacement = np.asarray(
            records[
                "action_l2_displacement"
            ],
            dtype=np.float64,
        )

        donor_rank = np.asarray(
            records[
                "donor_neighbor_rank"
            ],
            dtype=np.int64,
        )

    requested = int(
        np.floor(
            float(
                config[
                    "attack"
                ][
                    "transition_budget_fraction"
                ]
            )
            * used_transitions
        )
    )

    n = len(target)

    if n != requested:
        raise RuntimeError(
            "exact target budget mismatch"
        )

    if len(
        np.unique(target)
    ) != requested:
        raise RuntimeError(
            "targets are not unique"
        )

    for array in (
        donor,
        target_traj,
        donor_traj,
        state_distance,
        action_displacement,
        donor_rank,
    ):
        if len(array) != n:
            raise RuntimeError(
                "record array length mismatch"
            )

    if not np.array_equal(
        target_traj,
        transition_map[
            target
        ],
    ):
        raise RuntimeError(
            "target trajectory mapping mismatch"
        )

    if not np.array_equal(
        donor_traj,
        transition_map[
            donor
        ],
    ):
        raise RuntimeError(
            "donor trajectory mapping mismatch"
        )

    q70 = float(
        meta[
            "q70_target_return_threshold"
        ]
    )

    q30 = float(
        meta[
            "q30_donor_return_threshold"
        ]
    )

    if not np.all(
        returns[
            target_traj
        ]
        >= q70 - 1e-10
    ):
        raise RuntimeError(
            "target outside Q70 pool"
        )

    if not np.all(
        returns[
            donor_traj
        ]
        <= q30 + 1e-10
    ):
        raise RuntimeError(
            "donor outside Q30 pool"
        )

    if not np.all(
        action_displacement > 0.0
    ):
        raise RuntimeError(
            "non-positive action displacement"
        )

    k = int(
        config[
            "attack"
        ][
            "state_matching"
        ][
            "k_nearest_donors"
        ]
    )

    if np.any(
        donor_rank < 1
    ) or np.any(
        donor_rank > k
    ):
        raise RuntimeError(
            "donor rank outside frozen K"
        )

    with h5py.File(
        poison_path,
        "r",
    ) as poison:
        poisoned_actions = np.asarray(
            poison["actions"]
        )

    if not np.array_equal(
        poisoned_actions[
            target
        ],
        clean_actions[
            donor
        ],
    ):
        raise RuntimeError(
            "poisoned actions do not equal "
            "recorded clean donor actions"
        )

    changed = np.any(
        poisoned_actions[
            target
        ]
        != clean_actions[
            target
        ],
        axis=1,
    )

    if not np.all(
        changed
    ):
        raise RuntimeError(
            "not every target changed"
        )

    selected_mask = np.zeros(
        len(
            clean_actions
        ),
        dtype=bool,
    )

    selected_mask[
        target
    ] = True

    if not np.array_equal(
        clean_actions[
            ~selected_mask
        ],
        poisoned_actions[
            ~selected_mask
        ],
    ):
        raise RuntimeError(
            "unselected action changed"
        )

    if not np.array_equal(
        clean_actions[
            used_transitions:
        ],
        poisoned_actions[
            used_transitions:
        ],
    ):
        raise RuntimeError(
            "trailing fragment changed"
        )

    unique_donors, counts = (
        np.unique(
            donor,
            return_counts=True,
        )
    )

    if int(
        len(
            unique_donors
        )
    ) != int(
        meta[
            "unique_donor_count"
        ]
    ):
        raise RuntimeError(
            "unique donor count mismatch"
        )

    if int(
        np.max(
            counts
        )
    ) != int(
        meta[
            "max_donor_reuse_count"
        ]
    ):
        raise RuntimeError(
            "max donor reuse mismatch"
        )

    return {
        "attack_seed": attack_seed,
        "target_count": n,
        "unique_donor_count": int(
            len(
                unique_donors
            )
        ),
        "max_donor_reuse_count": int(
            np.max(
                counts
            )
        ),
        "state_distance_mean": float(
            np.mean(
                state_distance
            )
        ),
        "action_displacement_mean": float(
            np.mean(
                action_displacement
            )
        ),
        "only_actions_hdf5_dataset_differs": True,
        "exact_mapping_verified": True,
    }


def main():
    config = read_json(
        CONFIG
    )

    dataset_path = (
        ROOT
        / config[
            "dataset"
        ][
            "path"
        ]
    ).resolve()

    if (
        sha256_file(
            dataset_path
        )
        != config[
            "dataset"
        ][
            "sha256"
        ]
    ):
        raise RuntimeError(
            "clean dataset SHA mismatch"
        )

    with h5py.File(
        dataset_path,
        "r",
    ) as clean:
        clean_actions = np.asarray(
            clean["actions"]
        )

        rewards = np.asarray(
            clean["rewards"]
        )

        terminals = np.asarray(
            clean["terminals"],
            dtype=bool,
        )

        timeouts = np.asarray(
            clean["timeouts"],
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
            t.length
            for t in trajectories
        )
    )

    if (
        len(
            trajectories
        )
        != 1190
        or used_transitions
        != 999995
        or trailing
        != 5
    ):
        raise RuntimeError(
            "trajectory contract changed"
        )

    returns = trajectory_returns(
        rewards,
        trajectories,
    )

    transition_map = (
        transition_to_trajectory(
            trajectories,
            used_transitions,
        )
    )

    dt_reuse = verify_dt_reuse()

    clean_controls = (
        verify_clean_controls(
            config
        )
    )

    rows = []

    for attack_seed in (
        config[
            "qualification"
        ][
            "attack_seeds"
        ]
    ):
        rows.append(
            verify_one_artifact(
                attack_seed=int(
                    attack_seed
                ),
                config=config,
                clean_path=(
                    dataset_path
                ),
                clean_actions=(
                    clean_actions
                ),
                returns=(
                    returns
                ),
                trajectories=(
                    trajectories
                ),
                transition_map=(
                    transition_map
                ),
                used_transitions=(
                    used_transitions
                ),
            )
        )

    result = {
        "status": (
            "PREFLIGHT_PASS"
        ),
        "attack_name": (
            config[
                "name"
            ]
        ),
        "clean_dataset_sha256": (
            config[
                "dataset"
            ][
                "sha256"
            ]
        ),
        "dt_clean_control_reuse": (
            dt_reuse
        ),
        "clean_controls": (
            clean_controls
        ),
        "attack_rows": rows,
        "matrix": {
            "clean_runs_reused": 3,
            "new_poisoned_runs": 9,
            "model_seeds": [
                0,
                1,
                2,
            ],
            "attack_seeds": [
                30,
                31,
                32,
            ],
            "crossed_cells": 9,
        },
        "qualification_gate": (
            config[
                "qualification"
            ][
                "gate"
            ]
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=" * 96
    )

    print(
        "STATE-MATCHED ACTION SUBSTITUTION "
        "QUALIFICATION PREFLIGHT"
    )

    print(
        "=" * 96
    )

    print(
        "clean dataset SHA256: PASS"
    )

    print(
        "DT implementation unchanged: PASS"
    )

    print(
        "frozen clean controls reusable: PASS"
    )

    print(
        "clean returns:",
        clean_controls[
            "returns"
        ],
    )

    print(
        f"clean mean: "
        f"{clean_controls['mean']:.6f}"
    )

    print(
        f"required mean degradation: "
        f"{clean_controls['required_mean_degradation']:.6f}"
    )

    print()

    print(
        "seed targets unique_donors max_reuse "
        "state_dist_mean action_L2_mean"
    )

    for row in rows:
        print(
            f"{row['attack_seed']:4d} "
            f"{row['target_count']:7d} "
            f"{row['unique_donor_count']:13d} "
            f"{row['max_donor_reuse_count']:9d} "
            f"{row['state_distance_mean']:.6f} "
            f"{row['action_displacement_mean']:.6f}"
        )

    print()

    print(
        "matrix: 3 reused clean + "
        "9 new poisoned DT runs"
    )

    print(
        "QUALIFICATION PREFLIGHT: PASS"
    )

    print(
        "output ->",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
