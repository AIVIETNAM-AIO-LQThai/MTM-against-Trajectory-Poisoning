from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from scripts.audit_csdpc_overlap_resolution import (
    _build_merged_dataset,
    _build_requirements,
    _compute_rule_metrics,
    _generate_proposals,
    _is_prefix,
    _predict_selected_labels,
    _resolve_requirements,
    _to_selected_window,
)
from scripts.audit_csdpc_selection_semantics import (
    _build_pattern_index,
    _select_pattern_type_atomic_prefix,
)
from src.attacks.csdpc.attack import (
    prepare_csdpc_attack,
)
from src.attacks.csdpc.metadata import (
    logical_dataset_sha256,
    sha256_file,
    write_metadata_json,
)
from src.attacks.csdpc.selection import (
    compute_transition_budget,
)
from src.data.hdf5_io import (
    load_hdf5_dataset,
    write_hdf5_dataset,
)
from src.data.trajectories import (
    find_completed_trajectories,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "gates"
    / "csdpc_s2_overlap_r0_artifact_contract.json"
)

DEFAULT_DATASET = (
    ROOT
    / "data"
    / "raw"
    / "walker2d-medium-v2"
    / "walker2d_medium-v2.hdf5"
)

EXPECTED_CLEAN_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

R0 = "R0_FIRST_RARE_WINS"


def _rho_key(rho):
    return f"{float(rho):.2f}"


def _rho_code(rho):
    if np.isclose(rho, 0.01):
        return "001"

    if np.isclose(rho, 0.05):
        return "005"

    raise ValueError(
        f"unsupported frozen rho: {rho}"
    )


def _load_config(path):
    config = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        config["experiment"]
        == "CSDPC_S2_OVERLAP_R0_ARTIFACT_REALIZATION"
    )

    assert (
        config["variant_id"]
        == "csdpc_s2_overlap_r0_v1"
    )

    assert (
        config["selection"]
        == "S2_PATTERN_TYPE_ATOMIC_PREFIX"
    )

    assert (
        config["resolution"]
        == R0
    )

    assert (
        config["proposal_generation"]
        == "independent_canonical_C100"
    )

    assert int(
        config["num_clusters"]
    ) == 8

    assert int(
        config["sequence_length"]
    ) == 5

    assert np.isclose(
        float(
            config["eta"]
        ),
        0.05,
    )

    assert int(
        config["num_candidates"]
    ) == 100

    assert list(
        config["attack_seeds"]
    ) == [
        0,
        1,
        2,
    ]

    assert list(
        config["poison_rates"]
    ) == [
        0.01,
        0.05,
    ]

    return config


def _assert_dataset_schema_preserved(
    clean,
    poisoned,
):
    assert set(
        clean
    ) == set(
        poisoned
    )

    for key in clean:
        clean_array = np.asarray(
            clean[key]
        )

        poison_array = np.asarray(
            poisoned[key]
        )

        assert (
            clean_array.shape
            == poison_array.shape
        )

        assert (
            clean_array.dtype
            == poison_array.dtype
        )


def _assert_nonattack_arrays_identical(
    clean,
    poisoned,
):
    for key in clean:
        if key in {
            "observations",
            "actions",
        }:
            continue

        assert np.array_equal(
            np.asarray(
                clean[key]
            ),
            np.asarray(
                poisoned[key]
            ),
        )


def _assert_nonselected_rows_identical(
    clean,
    poisoned,
    selected_indices,
):
    n = len(
        clean[
            "observations"
        ]
    )

    mask = np.zeros(
        n,
        dtype=bool,
    )

    mask[
        np.asarray(
            selected_indices,
            dtype=np.int64,
        )
    ] = True

    assert np.array_equal(
        np.asarray(
            clean[
                "observations"
            ]
        )[
            ~mask
        ],
        np.asarray(
            poisoned[
                "observations"
            ]
        )[
            ~mask
        ],
    )

    assert np.array_equal(
        np.asarray(
            clean[
                "actions"
            ]
        )[
            ~mask
        ],
        np.asarray(
            poisoned[
                "actions"
            ]
        )[
            ~mask
        ],
    )


def _compare_r0_reference(
    *,
    actual,
    expected,
):
    keys = [
        "selected_window_count",
        "unique_transition_footprint",
        "modified_transition_cluster_label_change_fraction",
        "selected_window_pattern_change_fraction",
        "selected_window_frequency_improvement_fraction",
        "independent_target_pattern_preservation_fraction",
        "mean_source_pattern_frequency",
        "mean_independent_target_pattern_frequency",
        "mean_merged_target_pattern_frequency",
        "selected_source_pattern_eradication_fraction",
        "selected_source_occurrence_mass_reduction_fraction",
        "clean_to_poison_distinct_pattern_reduction_fraction",
        "removed_pattern_type_count",
        "new_pattern_type_count",
    ]

    for key in keys:
        a = actual[
            key
        ]

        e = expected[
            key
        ]

        if isinstance(
            e,
            int,
        ):
            if int(a) != int(e):
                raise RuntimeError(
                    "R0 reference mismatch: "
                    f"{key}: {a} != {e}"
                )
        else:
            if not np.isclose(
                float(a),
                float(e),
                rtol=0.0,
                atol=1.0e-12,
            ):
                raise RuntimeError(
                    "R0 reference mismatch: "
                    f"{key}: {a} != {e}"
                )


def _run_seed(
    *,
    seed,
    config,
    clean_dataset,
    clean_file_sha,
):
    prepared = prepare_csdpc_attack(
        clean_dataset,
        attack_seed=int(
            seed
        ),
        num_clusters=int(
            config[
                "num_clusters"
            ]
        ),
        sequence_length=int(
            config[
                "sequence_length"
            ]
        ),
        eta=float(
            config[
                "eta"
            ]
        ),
        num_candidates=int(
            config[
                "num_candidates"
            ]
        ),
    )

    (
        occurrences,
        ranked_patterns,
    ) = _build_pattern_index(
        prepared.windows,
        prepared.pattern_frequencies,
    )

    selections = {}

    for rho in config[
        "poison_rates"
    ]:
        rho = float(
            rho
        )

        budget = (
            compute_transition_budget(
                num_transitions=(
                    prepared.num_transitions
                ),
                rho=rho,
            )
        )

        selected = (
            _select_pattern_type_atomic_prefix(
                occurrences_by_pattern=(
                    occurrences
                ),
                ranked_patterns=(
                    ranked_patterns
                ),
                transition_budget=(
                    budget
                ),
            )
        )

        selections[
            rho
        ] = tuple(
            _to_selected_window(
                window
            )
            for window
            in selected.selected_windows
        )

    if not _is_prefix(
        selections[
            0.01
        ],
        selections[
            0.05
        ],
    ):
        raise RuntimeError(
            "rho=0.01 S2 selection is not "
            "an exact rho=0.05 prefix"
        )

    print(
        "S2 rho-prefix check: PASS"
    )

    max_windows = (
        selections[
            0.05
        ]
    )

    print(
        "Generating max-rho independent C100 proposals..."
    )

    max_proposals = (
        _generate_proposals(
            selected_windows=(
                max_windows
            ),
            clean_dataset=(
                clean_dataset
            ),
            prepared=prepared,
        )
    )

    terminals = np.asarray(
        clean_dataset[
            "terminals"
        ]
    )

    timeouts = np.asarray(
        clean_dataset[
            "timeouts"
        ]
    )

    trajectories, _ = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    clean_labels = np.asarray(
        prepared.clustering.labels,
        dtype=np.int64,
    )

    output_root = (
        ROOT
        / config[
            "output_root"
        ]
    )

    metadata_root = (
        ROOT
        / config[
            "metadata_root"
        ]
    )

    reference_path = (
        ROOT
        / config[
            "resolution_reference_root"
        ]
        / f"seed_{seed}.json"
    )

    reference = json.loads(
        reference_path.read_text(
            encoding="utf-8"
        )
    )

    low_indices = None
    low_observations = None
    low_actions = None

    records = []

    for rho in config[
        "poison_rates"
    ]:
        rho = float(
            rho
        )

        key = _rho_key(
            rho
        )

        code = _rho_code(
            rho
        )

        windows = selections[
            rho
        ]

        proposals = max_proposals[
            :len(
                windows
            )
        ]

        requirements = (
            _build_requirements(
                selected_windows=(
                    windows
                ),
                proposals=(
                    proposals
                ),
            )
        )

        resolved = (
            _resolve_requirements(
                requirements,
                rule_id=R0,
            )
        )

        poisoned = (
            _build_merged_dataset(
                clean_dataset=(
                    clean_dataset
                ),
                resolved=resolved,
            )
        )

        (
            poisoned_labels,
            selected_indices,
        ) = _predict_selected_labels(
            merged_dataset=(
                poisoned
            ),
            resolved=resolved,
            model=(
                prepared.clustering_model
            ),
            clean_labels=(
                clean_labels
            ),
        )

        metrics = (
            _compute_rule_metrics(
                rule_id=R0,
                clean_dataset=(
                    clean_dataset
                ),
                merged_dataset=(
                    poisoned
                ),
                clean_labels=(
                    clean_labels
                ),
                merged_labels=(
                    poisoned_labels
                ),
                selected_windows=(
                    windows
                ),
                proposals=(
                    proposals
                ),
                clean_counts=(
                    prepared.pattern_frequencies
                ),
                trajectories=(
                    trajectories
                ),
                selected_indices=(
                    selected_indices
                ),
                eta=float(
                    config[
                        "eta"
                    ]
                ),
            )
        )

        expected = (
            reference[
                "rho_results"
            ][key][
                "rules"
            ][R0]
        )

        _compare_r0_reference(
            actual=metrics,
            expected=expected,
        )

        print(
            f"rho={key} R0 reference identity: PASS"
        )

        _assert_dataset_schema_preserved(
            clean_dataset,
            poisoned,
        )

        _assert_nonattack_arrays_identical(
            clean_dataset,
            poisoned,
        )

        _assert_nonselected_rows_identical(
            clean_dataset,
            poisoned,
            selected_indices,
        )

        output_path = (
            output_root
            / (
                f"rho_{code}"
                f"_seed_{seed}.hdf5"
            )
        )

        metadata_path = (
            metadata_root
            / (
                f"rho_{code}"
                f"_seed_{seed}.json"
            )
        )

        write_hdf5_dataset(
            output_path,
            poisoned,
        )

        reloaded = (
            load_hdf5_dataset(
                output_path
            )
        )

        in_memory_logical_sha = (
            logical_dataset_sha256(
                poisoned
            )
        )

        on_disk_logical_sha = (
            logical_dataset_sha256(
                reloaded
            )
        )

        if (
            in_memory_logical_sha
            != on_disk_logical_sha
        ):
            raise RuntimeError(
                "written HDF5 logical hash "
                "does not match in-memory artifact"
            )

        poisoned_file_sha = (
            sha256_file(
                output_path
            )
        )

        metadata = {
            "schema_version": (
                "csdpc-s2-overlap-r0-artifact-v1"
            ),

            "variant_id": (
                config[
                    "variant_id"
                ]
            ),

            "canonical_attack_is_unchanged": True,

            "attack_seed": int(
                seed
            ),

            "rho": float(
                rho
            ),

            "selection": (
                config[
                    "selection"
                ]
            ),

            "resolution": (
                config[
                    "resolution"
                ]
            ),

            "num_clusters": int(
                config[
                    "num_clusters"
                ]
            ),

            "sequence_length": int(
                config[
                    "sequence_length"
                ]
            ),

            "eta": float(
                config[
                    "eta"
                ]
            ),

            "num_candidates": int(
                config[
                    "num_candidates"
                ]
            ),

            "requested_transition_budget": int(
                compute_transition_budget(
                    num_transitions=(
                        prepared.num_transitions
                    ),
                    rho=rho,
                )
            ),

            "selected_window_count": int(
                len(
                    windows
                )
            ),

            "unique_transition_footprint": int(
                len(
                    selected_indices
                )
            ),

            "clean_file_sha256": (
                clean_file_sha
            ),

            "clean_logical_sha256": (
                logical_dataset_sha256(
                    clean_dataset
                )
            ),

            "poisoned_file_sha256": (
                poisoned_file_sha
            ),

            "poisoned_logical_sha256": (
                on_disk_logical_sha
            ),

            "r0_reference_identity": True,

            "mechanism_metrics": metrics,

            "integrity": {
                "dataset_schema_preserved": True,
                "nonattack_arrays_identical": True,
                "nonselected_attack_rows_identical": True,
                "resolved_labels_match_kmeans": True,
                "written_hdf5_matches_in_memory": True,
                "r0_reference_identity": True
            }
        }

        write_metadata_json(
            metadata_path,
            metadata,
        )

        if np.isclose(
            rho,
            0.01,
        ):
            low_indices = np.asarray(
                selected_indices,
                dtype=np.int64,
            )

            low_observations = np.asarray(
                poisoned[
                    "observations"
                ]
            )[
                low_indices
            ].copy()

            low_actions = np.asarray(
                poisoned[
                    "actions"
                ]
            )[
                low_indices
            ].copy()

        if np.isclose(
            rho,
            0.05,
        ):
            high_indices = set(
                int(index)
                for index
                in selected_indices
            )

            subset = all(
                int(index)
                in high_indices
                for index
                in low_indices
            )

            if not subset:
                raise RuntimeError(
                    "rho=0.01 footprint is not "
                    "a subset of rho=0.05"
                )

            if not np.array_equal(
                low_observations,
                np.asarray(
                    poisoned[
                        "observations"
                    ]
                )[
                    low_indices
                ],
            ):
                raise RuntimeError(
                    "shared cross-rho observation "
                    "rows differ"
                )

            if not np.array_equal(
                low_actions,
                np.asarray(
                    poisoned[
                        "actions"
                    ]
                )[
                    low_indices
                ],
            ):
                raise RuntimeError(
                    "shared cross-rho action "
                    "rows differ"
                )

            print(
                "Cross-rho shared-row identity: PASS"
            )

        records.append(
            {
                "rho": float(
                    rho
                ),
                "artifact": str(
                    output_path.relative_to(
                        ROOT
                    )
                ),
                "metadata": str(
                    metadata_path.relative_to(
                        ROOT
                    )
                ),
                "logical_sha256": (
                    on_disk_logical_sha
                ),
                "file_sha256": (
                    poisoned_file_sha
                ),
            }
        )

        print("Wrote:", output_path)

        del poisoned
        del reloaded
        del poisoned_labels
        gc.collect()

    manifest = {
        "schema_version": (
            "csdpc-s2-overlap-r0-seed-manifest-v1"
        ),
        "variant_id": (config["variant_id"]),
        "attack_seed": int(seed),
        "cross_rho_consistency": True,
        "artifacts": records,
    }

    manifest_path = (
        metadata_root
        / f"seed_{seed}_manifest.json"
    )

    write_metadata_json(
        manifest_path,
        manifest,
    )

    print(
        "Wrote:",
        manifest_path
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    config = _load_config(args.config)

    if (args.seed not in config["attack_seeds"]):
        raise ValueError(
            "seed is not in frozen contract"
        )

    clean_file_sha = (sha256_file(args.dataset))

    if (clean_file_sha!= EXPECTED_CLEAN_SHA256):
        raise RuntimeError("clean dataset SHA256 mismatch")

    print("Frozen clean dataset SHA256: PASS")

    clean_dataset = (
        load_hdf5_dataset(args.dataset)
    )

    _run_seed(
        seed=args.seed,
        config=config,
        clean_dataset=(clean_dataset),
        clean_file_sha=(clean_file_sha),
    )


if __name__ == "__main__":
    main()