from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .types import (
    CSDPCAttackResult,
    PreparedCSDPC,
)


def sha256_file(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    path = Path(path)

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _update_length_prefixed(
    digest,
    payload: bytes,
) -> None:
    digest.update(
        len(payload).to_bytes(
            8,
            byteorder="little",
            signed=False,
        )
    )

    digest.update(payload)


def logical_dataset_sha256(
    dataset: Mapping[str, np.ndarray],
) -> str:
    """
    Stable logical hash over dataset arrays.

    Unlike the HDF5 file SHA256, this does not depend on
    HDF5 container layout or file-level metadata.

    The hash includes:
      - sorted dataset keys
      - dtype
      - shape
      - C-order array bytes
    """

    digest = hashlib.sha256()

    digest.update(
        b"CSDPC_LOGICAL_DATASET_V1"
    )

    for key in sorted(dataset):
        array = np.asarray(
            dataset[key]
        )

        if array.dtype.hasobject:
            raise TypeError(
                f"object dtype is not supported: {key}"
            )

        contiguous = np.ascontiguousarray(
            array
        )

        _update_length_prefixed(
            digest,
            key.encode("utf-8"),
        )

        _update_length_prefixed(
            digest,
            contiguous.dtype.str.encode(
                "ascii"
            ),
        )

        shape_bytes = json.dumps(
            list(contiguous.shape),
            separators=(",", ":"),
        ).encode("ascii")

        _update_length_prefixed(
            digest,
            shape_bytes,
        )

        _update_length_prefixed(
            digest,
            contiguous.tobytes(
                order="C"
            ),
        )

    return digest.hexdigest()


def _dataset_schema(
    dataset: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    schema = {}

    for key in sorted(dataset):
        array = np.asarray(
            dataset[key]
        )

        schema[key] = {
            "shape": [
                int(value)
                for value
                in array.shape
            ],
            "dtype": str(
                array.dtype
            ),
        }

    return schema


def _selected_windows_nonoverlapping(
    result: CSDPCAttackResult,
) -> bool:
    seen = set()

    for window in result.selected_windows:
        for index in (
            window.transition_indices
        ):
            if index in seen:
                return False

            seen.add(index)

    return True


def _perturbation_bounds_valid(
    clean_dataset: Mapping[str, np.ndarray],
    result: CSDPCAttackResult,
    *,
    eta: float,
    tolerance: float = 1.0e-10,
) -> bool:
    observations = np.asarray(
        clean_dataset["observations"]
    )

    actions = np.asarray(
        clean_dataset["actions"]
    )

    for window in (
        result.perturbed_windows
    ):
        start = window.global_start
        end = window.global_end

        source_observations = (
            observations[
                start:end
            ]
        )

        source_actions = (
            actions[
                start:end
            ]
        )

        allowed_state = (
            eta
            * np.max(
                np.abs(
                    source_observations
                ),
                axis=1,
            )
        )

        actual_state = np.max(
            np.abs(
                window.state_deltas
            ),
            axis=1,
        )

        if np.any(
            actual_state
            > allowed_state
            + tolerance
        ):
            return False

        allowed_action = (
            eta
            * np.max(
                np.abs(
                    source_actions
                ),
                axis=1,
            )
        )

        actual_action = np.max(
            np.abs(
                window.action_deltas
            ),
            axis=1,
        )

        if np.any(
            actual_action
            > allowed_action
            + tolerance
        ):
            return False

    return True


def build_csdpc_metadata(
    clean_dataset: Mapping[str, np.ndarray],
    prepared: PreparedCSDPC,
    result: CSDPCAttackResult,
    *,
    clean_file_sha256: str | None = None,
    poisoned_file_sha256: str | None = None,
    action_low: float = -1.0,
    action_high: float = 1.0,
) -> dict[str, Any]:
    poisoned = result.poisoned_dataset

    clean_keys = set(
        clean_dataset.keys()
    )

    poisoned_keys = set(
        poisoned.keys()
    )

    keys_preserved = (
        clean_keys
        == poisoned_keys
    )

    shapes_and_dtypes_preserved = (
        keys_preserved
    )

    if shapes_and_dtypes_preserved:
        for key in clean_keys:
            clean_array = np.asarray(
                clean_dataset[key]
            )

            poisoned_array = np.asarray(
                poisoned[key]
            )

            if (
                clean_array.shape
                != poisoned_array.shape
                or clean_array.dtype
                != poisoned_array.dtype
            ):
                shapes_and_dtypes_preserved = (
                    False
                )
                break

    def preserved_field(
        key: str,
    ) -> bool:
        if (
            key not in clean_dataset
            or key not in poisoned
        ):
            return False

        return bool(
            np.array_equal(
                np.asarray(
                    clean_dataset[key]
                ),
                np.asarray(
                    poisoned[key]
                ),
            )
        )

    modified_indices = tuple(
        int(index)
        for index
        in result.modified_transition_indices
    )

    indices_unique = (
        len(modified_indices)
        == len(set(modified_indices))
    )

    indices_in_range = all(
        0 <= index
        < prepared.num_transitions
        for index in modified_indices
    )

    action_array = np.asarray(
        poisoned["actions"]
    )

    action_bounds_valid = bool(
        np.all(
            action_array
            >= action_low - 1.0e-10
        )
        and np.all(
            action_array
            <= action_high + 1.0e-10
        )
    )

    actual_rho_expected = (
        result.actual_transition_budget
        / prepared.num_transitions
        if prepared.num_transitions
        else 0.0
    )

    actual_rho_consistent = bool(
        np.isclose(
            result.actual_rho,
            actual_rho_expected,
            rtol=0.0,
            atol=1.0e-15,
        )
    )

    integrity = {
        "keys_preserved": bool(
            keys_preserved
        ),
        "shapes_and_dtypes_preserved": bool(
            shapes_and_dtypes_preserved
        ),
        "rewards_identical": (
            preserved_field(
                "rewards"
            )
        ),
        "terminals_identical": (
            preserved_field(
                "terminals"
            )
        ),
        "timeouts_identical": (
            preserved_field(
                "timeouts"
            )
        ),
        "modified_indices_unique": bool(
            indices_unique
        ),
        "modified_indices_in_range": bool(
            indices_in_range
        ),
        "selected_windows_nonoverlapping": (
            _selected_windows_nonoverlapping(
                result
            )
        ),
        "budget_not_exceeded": bool(
            result.actual_transition_budget
            <= result.requested_transition_budget
        ),
        "budget_matches_index_count": bool(
            result.actual_transition_budget
            == len(modified_indices)
        ),
        "actual_rho_consistent": (
            actual_rho_consistent
        ),
        "action_bounds_valid": (
            action_bounds_valid
        ),
        "perturbation_bounds_valid": (
            _perturbation_bounds_valid(
                clean_dataset,
                result,
                eta=prepared.eta,
            )
        ),
    }

    source_frequencies = np.asarray(
        [
            window.source_frequency
            for window
            in result.perturbed_windows
        ],
        dtype=np.float64,
    )

    target_frequencies = np.asarray(
        [
            window.target_frequency
            for window
            in result.perturbed_windows
        ],
        dtype=np.float64,
    )

    if len(source_frequencies):
        improved = (
            target_frequencies
            > source_frequencies
        )

        ratios = (
            target_frequencies
            / np.maximum(
                source_frequencies,
                1.0,
            )
        )

        unchanged_patterns = np.asarray(
            [
                (
                    window.target_pattern
                    == window.source_pattern
                )
                for window
                in result.perturbed_windows
            ],
            dtype=bool,
        )

        effect = {
            "fraction_target_frequency_improved": float(
                np.mean(improved)
            ),
            "mean_source_frequency": float(
                np.mean(
                    source_frequencies
                )
            ),
            "mean_target_frequency": float(
                np.mean(
                    target_frequencies
                )
            ),
            "median_target_source_frequency_ratio": float(
                np.median(ratios)
            ),
            "fraction_target_pattern_unchanged": float(
                np.mean(
                    unchanged_patterns
                )
            ),
            "distinct_source_patterns": int(
                len(
                    {
                        window.source_pattern
                        for window
                        in result.perturbed_windows
                    }
                )
            ),
            "distinct_target_patterns": int(
                len(
                    {
                        window.target_pattern
                        for window
                        in result.perturbed_windows
                    }
                )
            ),
        }
    else:
        effect = {
            "fraction_target_frequency_improved": 0.0,
            "mean_source_frequency": 0.0,
            "mean_target_frequency": 0.0,
            "median_target_source_frequency_ratio": 0.0,
            "fraction_target_pattern_unchanged": 0.0,
            "distinct_source_patterns": 0,
            "distinct_target_patterns": 0,
        }

    selected_windows = [
        {
            "trajectory_id": int(
                window.trajectory_id
            ),
            "global_start": int(
                window.global_start
            ),
            "global_end": int(
                window.global_end
            ),
            "source_pattern": [
                int(value)
                for value
                in window.source_pattern
            ],
        }
        for window
        in result.selected_windows
    ]

    perturbed_windows = [
        {
            "trajectory_id": int(
                window.trajectory_id
            ),
            "global_start": int(
                window.global_start
            ),
            "global_end": int(
                window.global_end
            ),
            "source_pattern": [
                int(value)
                for value
                in window.source_pattern
            ],
            "target_pattern": [
                int(value)
                for value
                in window.target_pattern
            ],
            "source_frequency": int(
                window.source_frequency
            ),
            "target_frequency": int(
                window.target_frequency
            ),
            "candidate_index": int(
                window.candidate_index
            ),
            "total_linf_perturbation": float(
                window.total_linf_perturbation
            ),
        }
        for window
        in result.perturbed_windows
    ]

    return {
        "schema_version": (
            "csdpc-artifact-metadata-v1"
        ),
        "attack": {
            "name": "CSDPC",
            "attack_seed": int(
                prepared.attack_seed
            ),
            "requested_rho": float(
                result.requested_rho
            ),
            "actual_rho": float(
                result.actual_rho
            ),
            "requested_transition_budget": int(
                result.requested_transition_budget
            ),
            "actual_transition_budget": int(
                result.actual_transition_budget
            ),
            "sequence_length": int(
                prepared.sequence_length
            ),
            "eta": float(
                prepared.eta
            ),
            "num_candidates": int(
                prepared.num_candidates
            ),
        },
        "clustering": {
            "method": "kmeans",
            "num_clusters": int(
                prepared.clustering.centers.shape[
                    0
                ]
            ),
            "inertia": float(
                prepared.clustering.inertia
            ),
            "n_iter": int(
                prepared.clustering.n_iter
            ),
            "inertia_is_reproducibility_gate": False,
        },
        "dataset": {
            "num_transitions": int(
                prepared.num_transitions
            ),
            "clean_file_sha256": (
                clean_file_sha256
            ),
            "poisoned_file_sha256": (
                poisoned_file_sha256
            ),
            "clean_logical_sha256": (
                logical_dataset_sha256(
                    clean_dataset
                )
            ),
            "poisoned_logical_sha256": (
                logical_dataset_sha256(
                    poisoned
                )
            ),
            "schema": _dataset_schema(
                poisoned
            ),
        },
        "integrity": {
            **integrity,
            "all_checks_passed": bool(
                all(
                    integrity.values()
                )
            ),
        },
        "attack_effect": effect,
        "num_selected_windows": int(
            len(
                result.selected_windows
            )
        ),
        "modified_transition_indices": [
            int(index)
            for index
            in modified_indices
        ],
        "selected_windows": (
            selected_windows
        ),
        "perturbed_windows": (
            perturbed_windows
        ),
    }


def write_metadata_json(
    path: str | Path,
    metadata: Mapping[str, Any],
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

    with temporary_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            sort_keys=True,
        )

        handle.write("\n")

    os.replace(
        temporary_path,
        path,
    )