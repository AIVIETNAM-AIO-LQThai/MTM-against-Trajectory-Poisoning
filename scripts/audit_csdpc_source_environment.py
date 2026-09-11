from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from scripts.audit_csdpc_dedup_metric_reconciliation import (
    _compute_metrics,
)
from src.attacks.csdpc.clustering import (
    build_raw_decision_units,
    fit_kmeans_decision_units,
)
from src.attacks.csdpc.metadata import (
    sha256_file,
    write_metadata_json,
)
from src.data.trajectories import (
    find_completed_trajectories,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "gates"
    / "csdpc_source_environment_reconstruction.json"
)


def _read_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8")
    )


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)

    digest = hashlib.sha256()
    digest.update(
        contiguous.tobytes(order="C")
    )

    return digest.hexdigest()


def _infer_legacy_timeouts(
    terminals: np.ndarray,
    max_episode_steps: int,
) -> np.ndarray:
    """
    Mirror the legacy D4RL sequence_dataset fallback:

        final_timestep =
            episode_step == env._max_episode_steps - 1

    when an explicit `timeouts` field is absent.
    """
    terminals = np.asarray(
        terminals
    ).reshape(-1)

    inferred = np.zeros(
        terminals.shape[0],
        dtype=bool,
    )

    episode_step = 0

    for index in range(
        terminals.shape[0]
    ):
        final_timestep = (
            episode_step
            == max_episode_steps - 1
        )

        if final_timestep:
            inferred[index] = True

        if (
            bool(terminals[index])
            or final_timestep
        ):
            episode_step = 0

        episode_step += 1

    return inferred


def _load_core_dataset(
    path: Path,
    *,
    max_episode_steps: int,
):
    with h5py.File(
        path,
        "r",
    ) as handle:
        required = {
            "observations",
            "actions",
            "rewards",
            "terminals",
        }

        missing = sorted(
            required.difference(
                handle.keys()
            )
        )

        if missing:
            raise RuntimeError(
                f"{path} missing keys: "
                f"{missing}"
            )

        observations = np.asarray(
            handle["observations"]
        )
        actions = np.asarray(
            handle["actions"]
        )
        rewards = np.asarray(
            handle["rewards"]
        )
        terminals = np.asarray(
            handle["terminals"]
        )

        top_level_keys = sorted(
            handle.keys()
        )

        has_explicit_timeouts = (
            "timeouts" in handle
        )

        if has_explicit_timeouts:
            explicit_timeouts = (
                np.asarray(
                    handle["timeouts"]
                )
                .reshape(-1)
                .astype(bool)
            )

            resolved_timeouts = (
                explicit_timeouts
            )

            boundary_mode = (
                "explicit_timeouts"
            )

        else:
            explicit_timeouts = None

            resolved_timeouts = (
                _infer_legacy_timeouts(
                    terminals,
                    max_episode_steps,
                )
            )

            boundary_mode = (
                "d4rl_legacy_horizon_"
                "fallback"
            )

    n = observations.shape[0]

    for name, array in (
        ("actions", actions),
        ("rewards", rewards),
        ("terminals", terminals),
        (
            "resolved_timeouts",
            resolved_timeouts,
        ),
    ):
        if array.shape[0] != n:
            raise RuntimeError(
                f"{name} length mismatch"
            )

    return {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "resolved_timeouts": (
            resolved_timeouts
        ),
        "explicit_timeouts": (
            explicit_timeouts
        ),
        "has_explicit_timeouts": (
            has_explicit_timeouts
        ),
        "boundary_mode": boundary_mode,
        "top_level_keys": (
            top_level_keys
        ),
    }


def _analyze_dataset(
    dataset_spec: dict,
    *,
    config: dict,
    verify_sha: bool,
) -> dict:
    path = (
        ROOT
        / dataset_spec["path"]
    )

    if not path.exists():
        raise FileNotFoundError(path)

    file_sha = sha256_file(path)

    if verify_sha:
        expected = dataset_spec[
            "expected_sha256"
        ]

        if file_sha != expected:
            raise RuntimeError(
                "reference dataset SHA256 "
                "mismatch"
            )

    max_episode_steps = int(
        config[
            "legacy_timeout_fallback"
        ][
            "max_episode_steps"
        ]
    )

    data = _load_core_dataset(
        path,
        max_episode_steps=(
            max_episode_steps
        ),
    )

    observations = data[
        "observations"
    ]
    actions = data["actions"]
    rewards = data["rewards"]
    terminals = data["terminals"]
    timeouts = data[
        "resolved_timeouts"
    ]

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    raw_features = (
        build_raw_decision_units(
            observations,
            actions,
        )
    )

    (
        _,
        clustering,
    ) = fit_kmeans_decision_units(
        raw_features,
        num_clusters=int(
            config[
                "clustering"
            ][
                "num_clusters"
            ]
        ),
        seed=int(
            config[
                "clustering"
            ][
                "seed"
            ]
        ),
    )

    metrics = _compute_metrics(
        np.asarray(
            clustering.labels,
            dtype=np.int64,
        ),
        trajectories,
        sequence_length=int(
            config[
                "sequence"
            ][
                "length"
            ]
        ),
    )

    record = {
        "env_id": dataset_spec[
            "env_id"
        ],
        "path": str(path),
        "file_sha256": file_sha,

        "shape": {
            "transitions": int(
                observations.shape[0]
            ),
            "observation_dim": int(
                observations.shape[1]
            ),
            "action_dim": int(
                actions.shape[1]
            ),
        },

        "array_sha256": {
            "observations": (
                _array_sha256(
                    observations
                )
            ),
            "actions": (
                _array_sha256(
                    actions
                )
            ),
            "rewards": (
                _array_sha256(
                    rewards
                )
            ),
            "terminals": (
                _array_sha256(
                    terminals
                )
            ),
        },

        "top_level_keys": data[
            "top_level_keys"
        ],

        "boundary_semantics": {
            "mode": data[
                "boundary_mode"
            ],
            "has_explicit_timeouts": bool(
                data[
                    "has_explicit_timeouts"
                ]
            ),
            "terminal_count": int(
                np.count_nonzero(
                    terminals
                )
            ),
            "resolved_timeout_count": int(
                np.count_nonzero(
                    timeouts
                )
            ),
            "explicit_timeout_count": (
                int(
                    np.count_nonzero(
                        data[
                            "explicit_timeouts"
                        ]
                    )
                )
                if data[
                    "explicit_timeouts"
                ]
                is not None
                else None
            ),
            "completed_trajectory_count": int(
                len(trajectories)
            ),
            "trailing_transition_count": int(
                trailing
            ),
        },

        "clustering": {
            "num_clusters": int(
                config[
                    "clustering"
                ][
                    "num_clusters"
                ]
            ),
            "seed": int(
                config[
                    "clustering"
                ][
                    "seed"
                ]
            ),
            "inertia": float(
                clustering.inertia
            ),
            "n_iter": int(
                clustering.n_iter
            ),
        },

        "dedup_metrics": metrics,
    }

    del (
        observations,
        actions,
        rewards,
        terminals,
        timeouts,
        raw_features,
        clustering,
        data,
    )

    gc.collect()

    return record


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    config_path = (
        args.config.resolve()
    )

    config = _read_json(
        config_path
    )

    if (
        config.get("experiment")
        != "CSDPC_SOURCE_ENVIRONMENT_RECONSTRUCTION"
    ):
        raise RuntimeError(
            "unexpected experiment"
        )

    if (
        config.get("status")
        != "DIAGNOSTIC_ONLY"
    ):
        raise RuntimeError(
            "diagnostic status changed"
        )

    if not config.get(
        "canonical_attack_is_unchanged",
        False,
    ):
        raise RuntimeError(
            "canonical attack must "
            "remain unchanged"
        )

    print(
        "Analyzing frozen v2 reference..."
    )

    reference = _analyze_dataset(
        config[
            "reference_dataset"
        ],
        config=config,
        verify_sha=True,
    )

    print(
        "Analyzing v0 source candidate..."
    )

    candidate = _analyze_dataset(
        config[
            "candidate_dataset"
        ],
        config=config,
        verify_sha=False,
    )

    target = float(
        config[
            "source_reported_fingerprints"
        ][
            "deduplicated_distinct_"
            "pattern_reduction_approx"
        ]
    )

    reference_reduction = float(
        reference[
            "dedup_metrics"
        ][
            "canonical_distinct_type_"
            "reduction_fraction"
        ]
    )

    candidate_reduction = float(
        candidate[
            "dedup_metrics"
        ][
            "canonical_distinct_type_"
            "reduction_fraction"
        ]
    )

    comparison = {
        "observations_sha_equal": (
            reference[
                "array_sha256"
            ][
                "observations"
            ]
            == candidate[
                "array_sha256"
            ][
                "observations"
            ]
        ),
        "actions_sha_equal": (
            reference[
                "array_sha256"
            ][
                "actions"
            ]
            == candidate[
                "array_sha256"
            ][
                "actions"
            ]
        ),
        "rewards_sha_equal": (
            reference[
                "array_sha256"
            ][
                "rewards"
            ]
            == candidate[
                "array_sha256"
            ][
                "rewards"
            ]
        ),
        "terminals_sha_equal": (
            reference[
                "array_sha256"
            ][
                "terminals"
            ]
            == candidate[
                "array_sha256"
            ][
                "terminals"
            ]
        ),
        "source_target_reduction": (
            target
        ),
        "reference_reduction": (
            reference_reduction
        ),
        "candidate_reduction": (
            candidate_reduction
        ),
        "reference_distance_to_source": (
            abs(
                reference_reduction
                - target
            )
        ),
        "candidate_distance_to_source": (
            abs(
                candidate_reduction
                - target
            )
        ),
        "candidate_is_closer_to_source": (
            abs(
                candidate_reduction
                - target
            )
            < abs(
                reference_reduction
                - target
            )
        ),
    }

    summary = {
        "schema_version": (
            "csdpc-source-environment-"
            "reconstruction-v1"
        ),
        "experiment": config[
            "experiment"
        ],
        "status": config[
            "status"
        ],
        "canonical_attack_is_unchanged": (
            True
        ),
        "canonical_gate_b_is_unchanged": (
            True
        ),
        "reference": reference,
        "candidate": candidate,
        "comparison": comparison,
        "learner_training_performed": False,
        "poison_generation_performed": False,
    }

    output_path = (
        ROOT
        / config["output"]
    )

    write_metadata_json(
        output_path,
        summary,
    )

    print()
    print("=" * 72)
    print(
        "CSDPC SOURCE ENVIRONMENT "
        "RECONSTRUCTION"
    )
    print("=" * 72)

    for name, record in (
        ("reference_v2", reference),
        ("candidate_v0", candidate),
    ):
        metrics = record["dedup_metrics"]

        boundary = record["boundary_semantics"]

        distinct_reduction_pct = (
            100.0
            * float(
                metrics["canonical_distinct_type_reduction_fraction"]
            )
        )

        average_dedup_length = float(
            metrics["average_deduplicated_pattern_length"]
        )

        print()
        print(name)
        print(
            "  file sha256:",
            record["file_sha256"],
        )
        print(
            "  transitions:",
            record["shape"]["transitions"],
        )
        print(
            "  boundary mode:",
            boundary["mode"],
        )
        print(
            "  terminals:",
            boundary["terminal_count"],
        )
        print(
            "  resolved timeouts:",
            boundary["resolved_timeout_count"],
        )
        print(
            "  completed trajectories:",
            boundary["completed_trajectory_count"],
        )
        print(
            "  trailing transitions:",
            boundary["trailing_transition_count"],
        )
        print(
            "  windows:",
            metrics["window_count"],
        )
        print(
            "  raw distinct types:",
            metrics["raw_distinct_sequence_type_count"],
        )
        print(
            "  dedup distinct types:",
            metrics["deduplicated_distinct_pattern_type_count"],
        )
        print(
            "  distinct-type reduction:",
            f"{distinct_reduction_pct:.3f}%",
        )
        print(
            "  avg dedup length:",
            f"{average_dedup_length:.6f}",
        )

    print()
    print(
        "observations identical:",
        comparison["observations_sha_equal"],
    )
    print(
        "actions identical:",
        comparison["actions_sha_equal"],
    )
    reference_distance_pp = (
        100.0
        * float(
            comparison["reference_distance_to_source"]
        )
    )

    candidate_distance_pp = (
        100.0
        * float(
            comparison["candidate_distance_to_source"]
        )
    )
    print(
        "reference distance to ~80%:",
        f"{reference_distance_pp:.3f} pp",
    )

    print(
        "candidate distance to ~80%:",
        f"{candidate_distance_pp:.3f} pp",
    )
    print(
        "candidate closer to source:",
        comparison[
            "candidate_is_closer_to_source"
        ],
    )

    print()
    print(
        "SOURCE-ENVIRONMENT "
        "DIAGNOSTIC: RECORDED"
    )
    print(
        "LEARNER TRAINING: NOT RUN"
    )
    print(
        "POISON GENERATION: NOT RUN"
    )
    print(
        "summary:",
        output_path,
    )


if __name__ == "__main__":
    main()
