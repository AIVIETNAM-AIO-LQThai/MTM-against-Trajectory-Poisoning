from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.attacks.csdpc.metadata import (
    logical_dataset_sha256,
    write_metadata_json,
)
from src.baselines.cql.dataset_adapter import (
    audit_poison_exposure,
    build_cql_training_view,
)
from src.data.hdf5_io import (
    load_hdf5_dataset,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CLEAN = (
    ROOT
    / "data"
    / "raw"
    / "walker2d-medium-v2"
    / "walker2d_medium-v2.hdf5"
)

DEFAULT_POISON_ROOT = (
    ROOT
    / "data"
    / "poisoned"
    / "csdpc"
    / "walker2d-medium-v2"
)

DEFAULT_METADATA_ROOT = (
    ROOT
    / "data"
    / "metadata"
    / "csdpc"
    / "walker2d-medium-v2"
)

DEFAULT_OUTPUT = (
    ROOT
    / "data"
    / "metadata"
    / "cql_training_view"
    / "walker2d-medium-v2"
    / "training_view_audit.json"
)


def _rho_code(rho: float) -> str:
    return f"{int(round(rho * 100)):03d}"


def _load_metadata(
    root: Path,
    *,
    seed: int,
    rho: float,
):
    code = _rho_code(rho)

    path = (
        root
        / f"rho_{code}_seed_{seed}.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return json.load(handle)


def _index_sets(
    raw_indices: np.ndarray,
    modified_indices: np.ndarray,
):
    raw_indices = np.asarray(
        raw_indices,
        dtype=np.int64,
    )

    modified_indices = np.asarray(
        modified_indices,
        dtype=np.int64,
    )

    current_mask = np.isin(
        raw_indices,
        modified_indices,
    )

    next_mask = np.isin(
        raw_indices + 1,
        modified_indices,
    )

    current = set(
        int(value)
        for value
        in raw_indices[current_mask]
    )

    next_exposed = set(
        int(value)
        for value
        in raw_indices[next_mask]
    )

    any_exposed = set(
        int(value)
        for value
        in raw_indices[
            current_mask | next_mask
        ]
    )

    return (
        current,
        next_exposed,
        any_exposed,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clean",
        type=Path,
        default=DEFAULT_CLEAN,
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

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    print(
        "Loading frozen clean dataset..."
    )

    clean_dataset = load_hdf5_dataset(
        args.clean
    )

    clean_view = build_cql_training_view(
        clean_dataset
    )

    baseline_raw_indices = (
        clean_view["raw_indices"]
    )

    baseline_training_n = len(
        baseline_raw_indices
    )

    print(
        "Raw transitions:",
        len(
            clean_dataset[
                "observations"
            ]
        ),
    )

    print(
        "CQL training transitions:",
        baseline_training_n,
    )

    print(
        "Dropped from current-transition view:",
        (
            len(
                clean_dataset[
                    "observations"
                ]
            )
            - baseline_training_n
        ),
    )

    records = []

    all_integrity_pass = True

    for seed in (0, 1, 2):
        exposure_sets = {}

        for rho in (
            0.0,
            0.01,
            0.05,
        ):
            if rho == 0.0:
                dataset = clean_dataset

                modified = np.empty(
                    0,
                    dtype=np.int64,
                )

                requested_rho = 0.0
                actual_raw_rho = 0.0

            else:
                code = _rho_code(
                    rho
                )

                dataset_path = (
                    args.poison_root
                    / (
                        f"rho_{code}"
                        f"_seed_{seed}.hdf5"
                    )
                )

                if not dataset_path.exists():
                    raise FileNotFoundError(
                        dataset_path
                    )

                metadata = (
                    _load_metadata(
                        args.metadata_root,
                        seed=seed,
                        rho=rho,
                    )
                )

                dataset = (
                    load_hdf5_dataset(
                        dataset_path
                    )
                )

                modified = np.asarray(
                    metadata[
                        "modified_transition_indices"
                    ],
                    dtype=np.int64,
                )

                requested_rho = float(
                    metadata[
                        "attack"
                    ][
                        "requested_rho"
                    ]
                )

                actual_raw_rho = float(
                    metadata[
                        "attack"
                    ][
                        "actual_rho"
                    ]
                )

            view = build_cql_training_view(
                dataset
            )

            indices_identical = bool(
                np.array_equal(
                    view["raw_indices"],
                    baseline_raw_indices,
                )
            )

            rewards_identical = bool(
                np.array_equal(
                    view["rewards"],
                    clean_view["rewards"],
                )
            )

            terminals_identical = bool(
                np.array_equal(
                    view["terminals"],
                    clean_view["terminals"],
                )
            )

            training_n_identical = (
                len(
                    view["raw_indices"]
                )
                == baseline_training_n
            )

            exposure = (
                audit_poison_exposure(
                    view["raw_indices"],
                    modified,
                )
            )

            (
                current_set,
                next_set,
                any_set,
            ) = _index_sets(
                view["raw_indices"],
                modified,
            )

            exposure_sets[rho] = {
                "current": current_set,
                "next": next_set,
                "any": any_set,
            }

            modified_set = set(
                int(value)
                for value in modified
            )

            surviving_current = set(
                int(value)
                for value
                in baseline_raw_indices
            )

            dropped_modified = (
                modified_set
                - surviving_current
            )

            integrity = {
                "training_size_matches_clean": bool(
                    training_n_identical
                ),
                "raw_indices_match_clean": (
                    indices_identical
                ),
                "rewards_match_clean": (
                    rewards_identical
                ),
                "terminals_match_clean": (
                    terminals_identical
                ),
            }

            condition_pass = bool(
                all(
                    integrity.values()
                )
            )

            all_integrity_pass = (
                all_integrity_pass
                and condition_pass
            )

            record = {
                "attack_seed": seed,
                "requested_rho": (
                    requested_rho
                ),
                "actual_raw_rho": (
                    actual_raw_rho
                ),
                "raw_transition_count": int(
                    len(
                        dataset[
                            "observations"
                        ]
                    )
                ),
                "cql_training_transition_count": int(
                    len(
                        view[
                            "raw_indices"
                        ]
                    )
                ),
                "raw_modified_transition_count": int(
                    len(modified)
                ),
                "modified_dropped_as_current_transition": int(
                    len(
                        dropped_modified
                    )
                ),
                "exposure": exposure,
                "integrity": {
                    **integrity,
                    "all_checks_passed": (
                        condition_pass
                    ),
                },
                "training_view_logical_sha256": (
                    logical_dataset_sha256(
                        view
                    )
                ),
            }

            records.append(
                record
            )

            print()
            print(
                f"seed={seed} "
                f"rho={rho:.2%}"
            )

            print(
                "  integrity:",
                condition_pass,
            )

            print(
                "  raw modified:",
                len(modified),
            )

            print(
                "  dropped modified current:",
                len(
                    dropped_modified
                ),
            )

            print(
                "  action/current exposure:",
                f"{100.0 * exposure['action_exposure_rate']:.6f}%",
            )

            print(
                "  next-state exposure:",
                f"{100.0 * exposure['next_state_exposure_rate']:.6f}%",
            )

            print(
                "  any-tuple exposure:",
                f"{100.0 * exposure['any_tuple_exposure_rate']:.6f}%",
            )

        low = exposure_sets[
            0.01
        ]

        high = exposure_sets[
            0.05
        ]

        cross_rho_pass = bool(
            low["current"].issubset(
                high["current"]
            )
            and low["next"].issubset(
                high["next"]
            )
            and low["any"].issubset(
                high["any"]
            )
        )

        if not cross_rho_pass:
            raise RuntimeError(
                "CQL training-view cross-rho "
                f"nesting failed for seed {seed}"
            )

    result = {
        "schema_version": (
            "cql-training-view-audit-v1"
        ),
        "raw_clean_transition_count": int(
            len(
                clean_dataset[
                    "observations"
                ]
            )
        ),
        "clean_cql_training_transition_count": int(
            baseline_training_n
        ),
        "clean_training_view_logical_sha256": (
            logical_dataset_sha256(
                clean_view
            )
        ),
        "all_integrity_checks_passed": bool(
            all_integrity_pass
        ),
        "conditions": records,
    }

    if not all_integrity_pass:
        raise RuntimeError(
            "CQL training-view integrity "
            "audit failed"
        )

    write_metadata_json(
        args.output,
        result,
    )

    print()
    print(
        "=================================="
    )

    print(
        "CQL TRAINING-VIEW AUDIT: PASS"
    )

    print(
        "Saved:",
        args.output,
    )


if __name__ == "__main__":
    main()