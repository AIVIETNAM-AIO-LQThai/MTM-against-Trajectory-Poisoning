from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from src.data.trajectories import (
    find_completed_trajectories,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = (
    ROOT
    / "data"
    / "raw"
    / "walker2d-medium-v2"
    / "walker2d_medium-v2.hdf5"
)

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "attack_qualification"
    / "a1_rtg_inflation.json"
)

DEFAULT_POISON_ROOT = (
    ROOT
    / "data"
    / "poisoned"
    / "rtg_inflation"
    / "walker2d-medium-v2"
)

DEFAULT_METADATA_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "rtg_inflation"
    / "walker2d-medium-v2"
)


def sha256_file(path):
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_config(path):
    config = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    if (
        config.get("experiment")
        != "A1_RTG_INFLATION_ATTACK_QUALIFICATION"
    ):
        raise ValueError(
            "unexpected A1 experiment identifier"
        )

    if (
        config.get("status")
        != "PREDECLARED_ATTACK_GENERATION"
    ):
        raise ValueError(
            "unexpected A1 status"
        )

    expected = {
        "attack_seeds": [10, 11, 12],
        "transition_budget_fraction": 0.05,
        "low_return_candidate_quantile": 0.30,
        "target_return_quantile": 0.90,
        "minimum_budget_utilization": 0.98,
    }

    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(
                f"A1 frozen field changed: {key}"
            )

    return config


def load_hdf5(path):
    with h5py.File(
        path,
        "r",
    ) as handle:
        return {
            key: np.asarray(
                handle[key]
            )
            for key in handle.keys()
        }


def write_hdf5(path, dataset):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if path.exists():
        raise FileExistsError(
            path
        )

    with h5py.File(
        path,
        "w",
    ) as handle:
        for key, value in dataset.items():
            handle.create_dataset(
                key,
                data=np.asarray(
                    value
                ),
            )


def trajectory_returns(
    rewards,
    trajectories,
):
    rewards = np.asarray(
        rewards
    )

    return np.asarray(
        [
            float(
                np.sum(
                    rewards[
                        trajectory.start:
                        trajectory.end
                    ],
                    dtype=np.float64,
                )
            )
            for trajectory
            in trajectories
        ],
        dtype=np.float64,
    )


def select_trajectories(
    trajectories,
    returns,
    *,
    candidate_threshold,
    requested_budget,
    attack_seed,
):
    eligible = np.asarray(
        [
            index
            for index, value
            in enumerate(returns)
            if value
            <= candidate_threshold
        ],
        dtype=np.int64,
    )

    if len(eligible) == 0:
        raise RuntimeError(
            "empty low-return candidate pool"
        )

    rng = np.random.default_rng(
        attack_seed
    )

    order = eligible.copy()
    rng.shuffle(order)

    selected = []
    used = 0

    for trajectory_index in order:
        trajectory = trajectories[
            int(trajectory_index)
        ]

        length = int(
            trajectory.length
        )

        if (
            used + length
            > requested_budget
        ):
            continue

        selected.append(
            int(trajectory_index)
        )

        used += length

        if used == requested_budget:
            break

    return (
        selected,
        int(used),
        [
            int(x)
            for x in order
        ],
    )


def apply_reward_inflation(
    clean,
    trajectories,
    selected_indices,
    *,
    target_return,
):
    poisoned = {
        key: np.asarray(
            value
        ).copy()
        for key, value
        in clean.items()
    }

    clean_rewards = np.asarray(
        clean["rewards"]
    )

    poisoned_rewards = (
        poisoned["rewards"]
    )

    selected_records = []

    for trajectory_index in (
        selected_indices
    ):
        trajectory = trajectories[
            trajectory_index
        ]

        start = int(
            trajectory.start
        )

        end = int(
            trajectory.end
        )

        length = int(
            trajectory.length
        )

        source_return = float(
            np.sum(
                clean_rewards[
                    start:end
                ],
                dtype=np.float64,
            )
        )

        total_shift = float(
            target_return
            - source_return
        )

        if total_shift < 0.0:
            raise RuntimeError(
                "selected trajectory exceeds "
                "target return"
            )

        per_transition_shift = (
            total_shift
            / length
        )

        poisoned_rewards[
            start:end
        ] = (
            poisoned_rewards[
                start:end
            ]
            + np.asarray(
                per_transition_shift,
                dtype=poisoned_rewards.dtype,
            )
        )

        realized_return = float(
            np.sum(
                poisoned_rewards[
                    start:end
                ],
                dtype=np.float64,
            )
        )

        selected_records.append(
            {
                "trajectory_index": int(
                    trajectory_index
                ),
                "start": start,
                "end": end,
                "length": length,
                "clean_return": (
                    source_return
                ),
                "target_return": float(
                    target_return
                ),
                "per_transition_reward_shift": float(
                    per_transition_shift
                ),
                "poisoned_return": (
                    realized_return
                ),
                "absolute_target_error": float(
                    abs(
                        realized_return
                        - target_return
                    )
                ),
            }
        )

    return (
        poisoned,
        selected_records,
    )


def validate_artifact(
    clean,
    poisoned,
    trajectories,
    selected_indices,
    *,
    used_transitions,
    requested_budget,
    actual_budget,
    minimum_budget_utilization,
    candidate_threshold,
    target_return,
):
    selected_set = set(
        int(x)
        for x in selected_indices
    )

    if len(selected_set) != len(
        selected_indices
    ):
        raise RuntimeError(
            "duplicate selected trajectory"
        )

    if actual_budget > requested_budget:
        raise RuntimeError(
            "transition budget exceeded"
        )

    utilization = (
        actual_budget
        / requested_budget
    )

    if (
        utilization
        < minimum_budget_utilization
    ):
        raise RuntimeError(
            "budget utilization below frozen minimum: "
            f"{utilization:.6f}"
        )

    for key in clean:
        if (
            np.asarray(
                clean[key]
            ).shape
            != np.asarray(
                poisoned[key]
            ).shape
        ):
            raise RuntimeError(
                f"shape changed for {key}"
            )

        if (
            np.asarray(
                clean[key]
            ).dtype
            != np.asarray(
                poisoned[key]
            ).dtype
        ):
            raise RuntimeError(
                f"dtype changed for {key}"
            )

    for key in clean:
        if key == "rewards":
            continue

        if not np.array_equal(
            clean[key],
            poisoned[key],
        ):
            raise RuntimeError(
                f"non-reward array changed: {key}"
            )

    clean_rewards = np.asarray(
        clean["rewards"]
    )

    poisoned_rewards = np.asarray(
        poisoned["rewards"]
    )

    if not np.array_equal(
        clean_rewards[
            used_transitions:
        ],
        poisoned_rewards[
            used_transitions:
        ],
    ):
        raise RuntimeError(
            "trailing fragment reward changed"
        )

    clean_returns = trajectory_returns(
        clean_rewards,
        trajectories,
    )

    poisoned_returns = trajectory_returns(
        poisoned_rewards,
        trajectories,
    )

    selected_mask = np.zeros(
        len(trajectories),
        dtype=bool,
    )

    selected_mask[
        list(selected_set)
    ] = True

    if np.any(
        clean_returns[
            selected_mask
        ]
        > candidate_threshold
        + 1e-10
    ):
        raise RuntimeError(
            "selected trajectory outside "
            "low-return candidate pool"
        )

    if not np.array_equal(
        clean_rewards[
            np.concatenate(
                [
                    np.arange(
                        trajectory.start,
                        trajectory.end,
                        dtype=np.int64,
                    )
                    for index, trajectory
                    in enumerate(
                        trajectories
                    )
                    if index
                    not in selected_set
                ]
            )
        ],
        poisoned_rewards[
            np.concatenate(
                [
                    np.arange(
                        trajectory.start,
                        trajectory.end,
                        dtype=np.int64,
                    )
                    for index, trajectory
                    in enumerate(
                        trajectories
                    )
                    if index
                    not in selected_set
                ]
            )
        ],
    ):
        raise RuntimeError(
            "unselected completed reward changed"
        )

    max_target_error = float(
        np.max(
            np.abs(
                poisoned_returns[
                    selected_mask
                ]
                - target_return
            )
        )
    )

    # Float32 reward storage can accumulate a few 1e-3
    # over long trajectories.
    if max_target_error > 1e-2:
        raise RuntimeError(
            "selected poisoned trajectory return "
            "misses Q90 target: "
            f"{max_target_error}"
        )

    return {
        "requested_transition_budget": int(
            requested_budget
        ),
        "actual_transition_budget": int(
            actual_budget
        ),
        "budget_utilization": float(
            utilization
        ),
        "selected_trajectory_count": int(
            len(selected_indices)
        ),
        "max_selected_target_return_error": (
            max_target_error
        ),
        "non_reward_arrays_identical": True,
        "trailing_fragment_unchanged": True,
        "selected_within_candidate_pool": True,
        "unselected_rewards_identical": True,
    }


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--poison-root",
        type=Path,
        default=DEFAULT_POISON_ROOT,
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = load_config(
        args.config
    )

    dataset_path = (
        args.dataset.resolve()
    )

    actual_sha = sha256_file(
        dataset_path
    )

    if (
        actual_sha
        != config[
            "dataset_sha256"
        ]
    ):
        raise RuntimeError(
            "frozen clean dataset SHA256 mismatch"
        )

    print(
        "Frozen clean dataset SHA256: PASS"
    )

    clean = load_hdf5(
        dataset_path
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
            trajectory.length
            for trajectory
            in trajectories
        )
    )

    if (
        len(trajectories) != 1190
        or used_transitions != 999_995
        or trailing != 5
    ):
        raise RuntimeError(
            "frozen trajectory contract changed"
        )

    clean_returns = trajectory_returns(
        clean["rewards"],
        trajectories,
    )

    candidate_threshold = float(
        np.quantile(
            clean_returns,
            config[
                "low_return_candidate_quantile"
            ],
        )
    )

    target_return = float(
        np.quantile(
            clean_returns,
            config[
                "target_return_quantile"
            ],
        )
    )

    if not (
        target_return
        > candidate_threshold
    ):
        raise RuntimeError(
            "Q90 must exceed Q30"
        )

    requested_budget = int(
        config[
            "transition_budget_fraction"
        ]
        * used_transitions
    )

    args.poison_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.metadata_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    print()
    print("=" * 112)
    print(
        "A1 - LOW-RETURN TRAJECTORY RTG INFLATION"
    )
    print("=" * 112)

    print(
        "seed candidate_q30 target_q90 selected "
        "budget/request utilization max_target_error"
    )

    for attack_seed in (
        config["attack_seeds"]
    ):
        selected, actual_budget, order = (
            select_trajectories(
                trajectories,
                clean_returns,
                candidate_threshold=(
                    candidate_threshold
                ),
                requested_budget=(
                    requested_budget
                ),
                attack_seed=int(
                    attack_seed
                ),
            )
        )

        poisoned, records = (
            apply_reward_inflation(
                clean,
                trajectories,
                selected,
                target_return=target_return,
            )
        )

        integrity = validate_artifact(
            clean,
            poisoned,
            trajectories,
            selected,
            used_transitions=(
                used_transitions
            ),
            requested_budget=(
                requested_budget
            ),
            actual_budget=(
                actual_budget
            ),
            minimum_budget_utilization=float(
                config[
                    "minimum_budget_utilization"
                ]
            ),
            candidate_threshold=(
                candidate_threshold
            ),
            target_return=target_return,
        )

        poison_path = (
            args.poison_root
            / (
                f"attack_seed_"
                f"{int(attack_seed)}.hdf5"
            )
        )

        metadata_path = (
            args.metadata_root
            / (
                f"attack_seed_"
                f"{int(attack_seed)}.json"
            )
        )

        write_hdf5(
            poison_path,
            poisoned,
        )

        poison_sha = sha256_file(
            poison_path
        )

        metadata = {
            "schema_version": (
                "a1-rtg-inflation-artifact-v1"
            ),
            "experiment": (
                config["experiment"]
            ),
            "attack_name": (
                config["attack_name"]
            ),
            "attack_seed": int(
                attack_seed
            ),
            "clean_dataset": str(
                dataset_path
            ),
            "clean_dataset_sha256": (
                actual_sha
            ),
            "poisoned_dataset": str(
                poison_path.resolve()
            ),
            "poisoned_dataset_sha256": (
                poison_sha
            ),
            "used_transitions": (
                used_transitions
            ),
            "completed_trajectories": int(
                len(trajectories)
            ),
            "trailing_transitions": int(
                trailing
            ),
            "clean_return_quantiles": {
                "q30": (
                    candidate_threshold
                ),
                "q90": (
                    target_return
                ),
            },
            "requested_budget_fraction": float(
                config[
                    "transition_budget_fraction"
                ]
            ),
            "integrity": integrity,
            "selected_trajectory_indices": [
                int(x)
                for x in selected
            ],
            "candidate_shuffle_order": (
                order
            ),
            "selected_trajectories": (
                records
            ),
            "claim_boundary": [
                "Reward-only artifact generation.",
                "No victim performance used.",
                "Attack effectiveness not established by A1."
            ],
        }

        metadata_path.write_text(
            json.dumps(
                metadata,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        row = {
            "attack_seed": int(
                attack_seed
            ),
            "poisoned_dataset": str(
                poison_path
            ),
            "poisoned_dataset_sha256": (
                poison_sha
            ),
            **integrity,
        }

        summary_rows.append(
            row
        )

        print(
            f"{int(attack_seed):4d} "
            f"{candidate_threshold:13.3f} "
            f"{target_return:10.3f} "
            f"{integrity['selected_trajectory_count']:8d} "
            f"{actual_budget:6d}/{requested_budget:<6d} "
            f"{integrity['budget_utilization']:11.6f} "
            f"{integrity['max_selected_target_return_error']:.3e}"
        )

    summary = {
        "schema_version": (
            "a1-rtg-inflation-summary-v1"
        ),
        "experiment": (
            config["experiment"]
        ),
        "attack_name": (
            config["attack_name"]
        ),
        "clean_dataset_sha256": (
            actual_sha
        ),
        "candidate_q30_return": (
            candidate_threshold
        ),
        "target_q90_return": (
            target_return
        ),
        "requested_transition_budget": (
            requested_budget
        ),
        "rows": summary_rows,
        "planned_a2_qualification_gate": (
            config[
                "planned_a2_qualification_gate"
            ]
        ),
    }

    summary_path = (
        args.metadata_root
        / "summary.json"
    )

    summary_path.write_text(
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
        "summary ->",
        summary_path,
    )


if __name__ == "__main__":
    main()
