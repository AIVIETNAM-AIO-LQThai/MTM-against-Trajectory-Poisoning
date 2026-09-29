from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

from src.data.trajectories import find_completed_trajectories


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "attack_qualification"
    / "a5_state_matched_low_return_action_substitution.json"
)

DEFAULT_POISON_ROOT = (
    ROOT
    / "data"
    / "poisoned"
    / "state_matched_low_return_action_substitution"
    / "walker2d-medium-v2"
)

DEFAULT_METADATA_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "state_matched_low_return_action_substitution"
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
    config = json.loads(
        Path(path).read_text(
            encoding="utf-8"
        )
    )

    expected = {
        "schema_version": (
            "a5-state-matched-low-return-action-substitution-v1"
        ),
        "status": "predeclared",
        "name": (
            "state_matched_low_return_action_substitution"
        ),
    }

    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"frozen config changed: {key}"
            )

    attack = config["attack"]

    checks = [
        (
            attack["attack_seeds"],
            [30, 31, 32],
            "attack_seeds",
        ),
        (
            attack[
                "transition_budget_fraction"
            ],
            0.05,
            "transition_budget_fraction",
        ),
        (
            attack["target_pool"][
                "trajectory_return_quantile_min"
            ],
            0.70,
            "target_q70",
        ),
        (
            attack["donor_pool"][
                "trajectory_return_quantile_max"
            ],
            0.30,
            "donor_q30",
        ),
        (
            attack["state_matching"][
                "k_nearest_donors"
            ],
            64,
            "k_nearest_donors",
        ),
        (
            attack["state_matching"]["p"],
            2,
            "knn_p",
        ),
        (
            attack["state_matching"]["eps"],
            0.0,
            "knn_eps",
        ),
        (
            attack["state_matching"][
                "workers"
            ],
            1,
            "knn_workers",
        ),
        (
            attack["state_matching"][
                "query_batch_size"
            ],
            4096,
            "query_batch_size",
        ),
        (
            attack["target_pool"][
                "selection"
            ],
            (
                "seeded_permutation_without_"
                "replacement_then_take_exact_budget"
            ),
            "target_selection",
        ),
        (
            attack["donor_selection"][
                "primary"
            ],
            (
                "maximize_l2_action_distance_from_"
                "clean_target_action_among_k_"
                "nearest_state_donors"
            ),
            "donor_selection",
        ),
    ]

    for actual, frozen, name in checks:
        if actual != frozen:
            raise ValueError(
                f"frozen config changed: {name}: "
                f"{actual!r} != {frozen!r}"
            )

    return config


def trajectory_returns(
    rewards: np.ndarray,
    trajectories,
) -> np.ndarray:
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
            for trajectory in trajectories
        ],
        dtype=np.float64,
    )


def build_transition_to_trajectory(
    trajectories,
    used_transitions: int,
) -> np.ndarray:
    mapping = np.full(
        used_transitions,
        -1,
        dtype=np.int32,
    )

    for trajectory_index, trajectory in enumerate(
        trajectories
    ):
        mapping[
            trajectory.start:
            trajectory.end
        ] = int(
            trajectory_index
        )

    if np.any(
        mapping < 0
    ):
        raise RuntimeError(
            "completed transition lacks trajectory index"
        )

    return mapping


def pool_transition_indices(
    trajectories,
    returns: np.ndarray,
    *,
    threshold: float,
    mode: str,
) -> np.ndarray:
    pieces = []

    for index, trajectory in enumerate(
        trajectories
    ):
        value = float(
            returns[
                index
            ]
        )

        include = (
            value >= threshold
            if mode == "ge"
            else value <= threshold
        )

        if include:
            pieces.append(
                np.arange(
                    trajectory.start,
                    trajectory.end,
                    dtype=np.int64,
                )
            )

    if not pieces:
        raise RuntimeError(
            f"empty transition pool: mode={mode}"
        )

    return np.concatenate(
        pieces
    )


def select_targets(
    target_pool: np.ndarray,
    *,
    requested_budget: int,
    attack_seed: int,
) -> np.ndarray:
    target_pool = np.asarray(
        target_pool,
        dtype=np.int64,
    )

    if requested_budget > len(
        target_pool
    ):
        raise RuntimeError(
            "requested budget exceeds target pool"
        )

    rng = np.random.default_rng(
        int(
            attack_seed
        )
    )

    permutation = rng.permutation(
        target_pool
    )

    return np.asarray(
        permutation[
            :requested_budget
        ],
        dtype=np.int64,
    )


def choose_donors_for_batch(
    *,
    target_indices: np.ndarray,
    normalized_states: np.ndarray,
    clean_actions: np.ndarray,
    donor_indices: np.ndarray,
    donor_tree: cKDTree,
    k: int,
    p: int,
    eps: float,
    workers: int,
):
    target_indices = np.asarray(
        target_indices,
        dtype=np.int64,
    )

    query_states = normalized_states[
        target_indices
    ]

    distances, local_neighbors = (
        donor_tree.query(
            query_states,
            k=k,
            p=p,
            eps=eps,
            workers=workers,
        )
    )

    if k == 1:
        distances = distances[
            :, None
        ]
        local_neighbors = local_neighbors[
            :, None
        ]

    selected_donors = np.empty(
        len(
            target_indices
        ),
        dtype=np.int64,
    )

    selected_state_distances = np.empty(
        len(
            target_indices
        ),
        dtype=np.float64,
    )

    selected_action_displacements = np.empty(
        len(
            target_indices
        ),
        dtype=np.float64,
    )

    selected_neighbor_ranks = np.empty(
        len(
            target_indices
        ),
        dtype=np.int16,
    )

    for row, target_index in enumerate(
        target_indices
    ):
        donor_global = donor_indices[
            np.asarray(
                local_neighbors[
                    row
                ],
                dtype=np.int64,
            )
        ]

        donor_dist = np.asarray(
            distances[
                row
            ],
            dtype=np.float64,
        )

        # Deterministic ordering of the returned K-neighbor set:
        # state distance first, then global transition index.
        order = np.lexsort(
            (
                donor_global,
                donor_dist,
            )
        )

        donor_global = donor_global[
            order
        ]

        donor_dist = donor_dist[
            order
        ]

        target_action = clean_actions[
            target_index
        ].astype(
            np.float64,
            copy=False,
        )

        candidate_actions = clean_actions[
            donor_global
        ].astype(
            np.float64,
            copy=False,
        )

        displacement = np.linalg.norm(
            candidate_actions
            - target_action[
                None,
                :
            ],
            axis=1,
        )

        # np.argmax returns the first maximum. Because candidates
        # are already ordered by (state distance, global index),
        # this implements the frozen tie-breaking rule.
        selected_rank = int(
            np.argmax(
                displacement
            )
        )

        selected_donor = int(
            donor_global[
                selected_rank
            ]
        )

        selected_displacement = float(
            displacement[
                selected_rank
            ]
        )

        if not (
            selected_displacement
            > 0.0
        ):
            raise RuntimeError(
                "selected target has no differing "
                "action among frozen K nearest donors: "
                f"target={int(target_index)}"
            )

        selected_donors[
            row
        ] = selected_donor

        selected_state_distances[
            row
        ] = float(
            donor_dist[
                selected_rank
            ]
        )

        selected_action_displacements[
            row
        ] = (
            selected_displacement
        )

        # 1-based rank is easier to interpret in metadata.
        selected_neighbor_ranks[
            row
        ] = (
            selected_rank + 1
        )

    return (
        selected_donors,
        selected_state_distances,
        selected_action_displacements,
        selected_neighbor_ranks,
    )


def build_poisoned_actions(
    *,
    clean_actions: np.ndarray,
    normalized_states: np.ndarray,
    target_indices: np.ndarray,
    donor_indices: np.ndarray,
    k: int,
    p: int,
    eps: float,
    workers: int,
    query_batch_size: int,
):
    clean_actions = np.asarray(
        clean_actions
    )

    poisoned_actions = (
        clean_actions.copy()
    )

    donor_tree = cKDTree(
        normalized_states[
            donor_indices
        ],
        compact_nodes=True,
        balanced_tree=True,
        copy_data=False,
    )

    all_selected_donors = np.empty(
        len(
            target_indices
        ),
        dtype=np.int64,
    )

    all_state_distances = np.empty(
        len(
            target_indices
        ),
        dtype=np.float64,
    )

    all_action_displacements = np.empty(
        len(
            target_indices
        ),
        dtype=np.float64,
    )

    all_neighbor_ranks = np.empty(
        len(
            target_indices
        ),
        dtype=np.int16,
    )

    for start in range(
        0,
        len(
            target_indices
        ),
        query_batch_size,
    ):
        end = min(
            start
            + query_batch_size,
            len(
                target_indices
            ),
        )

        (
            selected_donors,
            state_distances,
            action_displacements,
            neighbor_ranks,
        ) = choose_donors_for_batch(
            target_indices=target_indices[
                start:end
            ],
            normalized_states=(
                normalized_states
            ),
            clean_actions=(
                clean_actions
            ),
            donor_indices=(
                donor_indices
            ),
            donor_tree=(
                donor_tree
            ),
            k=k,
            p=p,
            eps=eps,
            workers=workers,
        )

        all_selected_donors[
            start:end
        ] = selected_donors

        all_state_distances[
            start:end
        ] = state_distances

        all_action_displacements[
            start:end
        ] = action_displacements

        all_neighbor_ranks[
            start:end
        ] = neighbor_ranks

        poisoned_actions[
            target_indices[
                start:end
            ]
        ] = clean_actions[
            selected_donors
        ]

        print(
            "matched targets "
            f"{end}/{len(target_indices)}"
        )

    return (
        poisoned_actions,
        all_selected_donors,
        all_state_distances,
        all_action_displacements,
        all_neighbor_ranks,
    )


def write_action_only_hdf5(
    clean_path: Path,
    poison_path: Path,
    poisoned_actions: np.ndarray,
):
    poison_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if poison_path.exists():
        raise FileExistsError(
            poison_path
        )

    shutil.copy2(
        clean_path,
        poison_path,
    )

    try:
        with h5py.File(
            poison_path,
            "r+",
        ) as handle:
            dataset = handle[
                "actions"
            ]

            poisoned_actions = np.asarray(
                poisoned_actions
            )

            if (
                dataset.shape
                != poisoned_actions.shape
            ):
                raise RuntimeError(
                    "action shape changed"
                )

            if (
                dataset.dtype
                != poisoned_actions.dtype
            ):
                raise RuntimeError(
                    "action dtype changed"
                )

            dataset[...] = (
                poisoned_actions
            )

            handle.flush()

    except Exception:
        poison_path.unlink(
            missing_ok=True
        )
        raise


def _attribute_dict(obj):
    return {
        str(key): np.asarray(
            value
        )
        for key, value in obj.attrs.items()
    }


def _assert_attributes_equal(
    clean_obj,
    poison_obj,
    *,
    object_path: str,
):
    clean_attrs = _attribute_dict(
        clean_obj
    )

    poison_attrs = _attribute_dict(
        poison_obj
    )

    if set(
        clean_attrs
    ) != set(
        poison_attrs
    ):
        raise RuntimeError(
            "HDF5 attribute keys changed "
            f"at {object_path}"
        )

    for key in clean_attrs:
        clean_value = clean_attrs[
            key
        ]

        poison_value = poison_attrs[
            key
        ]

        if (
            clean_value.shape
            != poison_value.shape
        ):
            raise RuntimeError(
                "HDF5 attribute shape changed "
                f"at {object_path}:{key}"
            )

        if not np.array_equal(
            clean_value,
            poison_value,
        ):
            raise RuntimeError(
                "HDF5 attribute value changed "
                f"at {object_path}:{key}"
            )


def _object_kind(obj):
    if isinstance(
        obj,
        h5py.Dataset,
    ):
        return "dataset"

    if isinstance(
        obj,
        h5py.Group,
    ):
        return "group"

    return type(
        obj
    ).__name__


def _collect_object_paths(handle):
    objects = {
        "/": "group"
    }

    def visitor(
        name,
        obj,
    ):
        objects[
            "/" + name
        ] = _object_kind(
            obj
        )

    handle.visititems(
        visitor
    )

    return objects


def assert_action_only_difference(
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
        clean_objects = (
            _collect_object_paths(
                clean
            )
        )

        poison_objects = (
            _collect_object_paths(
                poison
            )
        )

        if (
            clean_objects
            != poison_objects
        ):
            raise RuntimeError(
                "HDF5 hierarchy changed"
            )

        _assert_attributes_equal(
            clean,
            poison,
            object_path="/",
        )

        actions_changed = False

        for object_path, kind in (
            clean_objects.items()
        ):
            if object_path == "/":
                continue

            name = object_path.lstrip(
                "/"
            )

            clean_obj = clean[
                name
            ]

            poison_obj = poison[
                name
            ]

            _assert_attributes_equal(
                clean_obj,
                poison_obj,
                object_path=(
                    object_path
                ),
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
                    f"{object_path}"
                )

            clean_data = clean_obj[()]
            poison_data = (
                poison_obj[()]
            )

            if (
                object_path
                == "/actions"
            ):
                actions_changed = (
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
                    f"{object_path}"
                )

        if not actions_changed:
            raise RuntimeError(
                "actions dataset did not change"
            )


def parse_args():
    parser = argparse.ArgumentParser()

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
        ROOT
        / config[
            "dataset"
        ][
            "path"
        ]
    ).resolve()

    normalization_path = (
        ROOT
        / config[
            "dataset"
        ][
            "normalization_path"
        ]
    ).resolve()

    actual_sha = sha256_file(
        dataset_path
    )

    if actual_sha != config[
        "dataset"
    ][
        "sha256"
    ]:
        raise RuntimeError(
            "clean dataset SHA256 mismatch"
        )

    with h5py.File(
        dataset_path,
        "r",
    ) as handle:
        observations = np.asarray(
            handle[
                "observations"
            ]
        )

        clean_actions = np.asarray(
            handle[
                "actions"
            ]
        )

        rewards = np.asarray(
            handle[
                "rewards"
            ]
        )

        terminals = np.asarray(
            handle[
                "terminals"
            ],
            dtype=bool,
        )

        timeouts = np.asarray(
            handle[
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
            for trajectory in trajectories
        )
    )

    dataset_cfg = config[
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
            "frozen trajectory contract changed"
        )

    with np.load(
        normalization_path
    ) as handle:
        state_mean = np.asarray(
            handle[
                "state_mean"
            ],
            dtype=np.float64,
        )

        state_std = np.asarray(
            handle[
                "state_std"
            ],
            dtype=np.float64,
        )

        if (
            int(
                handle[
                    "num_training_transitions"
                ]
            )
            != used_transitions
            or int(
                handle[
                    "num_trajectories"
                ]
            )
            != len(
                trajectories
            )
        ):
            raise RuntimeError(
                "normalization metadata contract changed"
            )

    if np.any(
        state_std <= 0.0
    ):
        raise RuntimeError(
            "state normalization std must be positive"
        )

    normalized_states = (
        observations.astype(
            np.float64
        )
        - state_mean[
            None,
            :
        ]
    ) / state_std[
        None,
        :
    ]

    returns = trajectory_returns(
        rewards,
        trajectories,
    )

    attack = config[
        "attack"
    ]

    q70 = float(
        np.quantile(
            returns,
            attack[
                "target_pool"
            ][
                "trajectory_return_quantile_min"
            ],
        )
    )

    q30 = float(
        np.quantile(
            returns,
            attack[
                "donor_pool"
            ][
                "trajectory_return_quantile_max"
            ],
        )
    )

    target_pool = (
        pool_transition_indices(
            trajectories,
            returns,
            threshold=q70,
            mode="ge",
        )
    )

    donor_pool = (
        pool_transition_indices(
            trajectories,
            returns,
            threshold=q30,
            mode="le",
        )
    )

    requested_budget = int(
        np.floor(
            float(
                attack[
                    "transition_budget_fraction"
                ]
            )
            * used_transitions
        )
    )

    transition_to_trajectory = (
        build_transition_to_trajectory(
            trajectories,
            used_transitions,
        )
    )

    matching = attack[
        "state_matching"
    ]

    k = int(
        matching[
            "k_nearest_donors"
        ]
    )

    if k > len(
        donor_pool
    ):
        raise RuntimeError(
            "K exceeds donor transition pool"
        )

    args.poison_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.metadata_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 108
    )

    print(
        "STATE-MATCHED LOW-RETURN ACTION SUBSTITUTION"
    )

    print(
        "=" * 108
    )

    print(
        f"Q70 target trajectory threshold: {q70:.6f}"
    )

    print(
        f"Q30 donor trajectory threshold:  {q30:.6f}"
    )

    print(
        f"target transition pool: {len(target_pool)}"
    )

    print(
        f"donor transition pool:  {len(donor_pool)}"
    )

    print(
        f"exact target budget:     {requested_budget}"
    )

    print()

    summary_rows = []

    for attack_seed in attack[
        "attack_seeds"
    ]:
        print(
            "-" * 108
        )

        print(
            f"attack seed {attack_seed}"
        )

        target_indices = select_targets(
            target_pool,
            requested_budget=(
                requested_budget
            ),
            attack_seed=int(
                attack_seed
            ),
        )

        (
            poisoned_actions,
            selected_donors,
            state_distances,
            action_displacements,
            neighbor_ranks,
        ) = build_poisoned_actions(
            clean_actions=clean_actions,
            normalized_states=(
                normalized_states
            ),
            target_indices=(
                target_indices
            ),
            donor_indices=(
                donor_pool
            ),
            k=k,
            p=int(
                matching[
                    "p"
                ]
            ),
            eps=float(
                matching[
                    "eps"
                ]
            ),
            workers=int(
                matching[
                    "workers"
                ]
            ),
            query_batch_size=int(
                matching[
                    "query_batch_size"
                ]
            ),
        )

        if len(
            np.unique(
                target_indices
            )
        ) != requested_budget:
            raise RuntimeError(
                "target indices are not unique"
            )

        if not np.all(
            returns[
                transition_to_trajectory[
                    target_indices
                ]
            ]
            >= q70 - 1e-10
        ):
            raise RuntimeError(
                "target outside frozen Q70 pool"
            )

        if not np.all(
            returns[
                transition_to_trajectory[
                    selected_donors
                ]
            ]
            <= q30 + 1e-10
        ):
            raise RuntimeError(
                "donor outside frozen Q30 pool"
            )

        if not np.all(
            np.any(
                poisoned_actions[
                    target_indices
                ]
                != clean_actions[
                    target_indices
                ],
                axis=1,
            )
        ):
            raise RuntimeError(
                "not every selected target changed action"
            )

        selected_mask = np.zeros(
            len(
                clean_actions
            ),
            dtype=bool,
        )

        selected_mask[
            target_indices
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

        if not np.array_equal(
            poisoned_actions[
                target_indices
            ],
            clean_actions[
                selected_donors
            ],
        ):
            raise RuntimeError(
                "poisoned action does not equal donor action"
            )

        poison_path = (
            args.poison_root
            / (
                f"attack_seed_"
                f"{int(attack_seed)}.hdf5"
            )
        )

        write_action_only_hdf5(
            dataset_path,
            poison_path,
            poisoned_actions,
        )

        assert_action_only_difference(
            dataset_path,
            poison_path,
        )

        poison_sha = sha256_file(
            poison_path
        )

        target_trajectory_indices = (
            transition_to_trajectory[
                target_indices
            ]
        )

        donor_trajectory_indices = (
            transition_to_trajectory[
                selected_donors
            ]
        )

        unique_donors, donor_counts = (
            np.unique(
                selected_donors,
                return_counts=True,
            )
        )

        metadata_npz_path = (
            args.metadata_root
            / (
                f"attack_seed_"
                f"{int(attack_seed)}_records.npz"
            )
        )

        np.savez_compressed(
            metadata_npz_path,
            target_global_index=(
                target_indices
            ),
            donor_global_index=(
                selected_donors
            ),
            target_trajectory_index=(
                target_trajectory_indices
            ),
            donor_trajectory_index=(
                donor_trajectory_indices
            ),
            normalized_state_distance=(
                state_distances
            ),
            action_l2_displacement=(
                action_displacements
            ),
            donor_neighbor_rank=(
                neighbor_ranks
            ),
        )

        record_sha = sha256_file(
            metadata_npz_path
        )

        metadata_json_path = (
            args.metadata_root
            / (
                f"attack_seed_"
                f"{int(attack_seed)}.json"
            )
        )

        row = {
            "attack_seed": int(
                attack_seed
            ),
            "requested_transition_budget": (
                requested_budget
            ),
            "actual_transition_budget": int(
                len(
                    target_indices
                )
            ),
            "target_pool_transition_count": int(
                len(
                    target_pool
                )
            ),
            "donor_pool_transition_count": int(
                len(
                    donor_pool
                )
            ),
            "q70_target_return_threshold": (
                q70
            ),
            "q30_donor_return_threshold": (
                q30
            ),
            "k_nearest_donors": k,
            "changed_fraction": 1.0,
            "normalized_state_distance": {
                "mean": float(
                    np.mean(
                        state_distances
                    )
                ),
                "median": float(
                    np.median(
                        state_distances
                    )
                ),
                "p90": float(
                    np.quantile(
                        state_distances,
                        0.90,
                    )
                ),
                "p99": float(
                    np.quantile(
                        state_distances,
                        0.99,
                    )
                ),
                "max": float(
                    np.max(
                        state_distances
                    )
                ),
            },
            "action_l2_displacement": {
                "mean": float(
                    np.mean(
                        action_displacements
                    )
                ),
                "median": float(
                    np.median(
                        action_displacements
                    )
                ),
                "p10": float(
                    np.quantile(
                        action_displacements,
                        0.10,
                    )
                ),
                "p90": float(
                    np.quantile(
                        action_displacements,
                        0.90,
                    )
                ),
                "min": float(
                    np.min(
                        action_displacements
                    )
                ),
                "max": float(
                    np.max(
                        action_displacements
                    )
                ),
            },
            "donor_neighbor_rank": {
                "mean": float(
                    np.mean(
                        neighbor_ranks
                    )
                ),
                "median": float(
                    np.median(
                        neighbor_ranks
                    )
                ),
                "max": int(
                    np.max(
                        neighbor_ranks
                    )
                ),
            },
            "unique_donor_count": int(
                len(
                    unique_donors
                )
            ),
            "max_donor_reuse_count": int(
                np.max(
                    donor_counts
                )
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
            "records_file": str(
                metadata_npz_path.resolve()
            ),
            "records_file_sha256": (
                record_sha
            ),
            "integrity": {
                "exact_transition_budget": True,
                "all_selected_targets_changed": True,
                "all_targets_in_q70_or_higher_trajectories": True,
                "all_donors_in_q30_or_lower_trajectories": True,
                "every_poisoned_action_equals_recorded_clean_donor_action": True,
                "unselected_actions_identical": True,
                "non_action_hdf5_content_identical": True,
                "trailing_fragment_unchanged": True,
            },
            "claim_boundary": config[
                "claim_boundary"
            ],
        }

        metadata_json_path.write_text(
            json.dumps(
                row,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        summary_rows.append(
            row
        )

        print(
            f"budget={len(target_indices)}/"
            f"{requested_budget} "
            f"state_dist_mean="
            f"{np.mean(state_distances):.6f} "
            f"action_L2_mean="
            f"{np.mean(action_displacements):.6f} "
            f"unique_donors="
            f"{len(unique_donors)} "
            f"max_reuse="
            f"{np.max(donor_counts)}"
        )

    summary = {
        "schema_version": (
            "state-matched-low-return-action-"
            "substitution-summary-v1"
        ),
        "status": (
            "ARTIFACT_GENERATION_PASS"
        ),
        "attack_name": config[
            "name"
        ],
        "clean_dataset_sha256": (
            actual_sha
        ),
        "q70_target_return_threshold": (
            q70
        ),
        "q30_donor_return_threshold": (
            q30
        ),
        "requested_transition_budget": (
            requested_budget
        ),
        "rows": summary_rows,
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
        "STATE-MATCHED ACTION SUBSTITUTION "
        "ARTIFACT GENERATION: PASS"
    )

    print(
        "metadata ->",
        args.metadata_root,
    )


if __name__ == "__main__":
    main()
