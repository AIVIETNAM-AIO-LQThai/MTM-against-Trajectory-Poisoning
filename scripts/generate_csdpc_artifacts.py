from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

from src.attacks.csdpc.attack import (
    apply_csdpc_attack,
    prepare_csdpc_attack,
)
from src.attacks.csdpc.metadata import (
    build_csdpc_metadata,
    logical_dataset_sha256,
    sha256_file,
    write_metadata_json,
)
from src.data.hdf5_io import (
    load_hdf5_dataset,
    write_hdf5_dataset,
)


ROOT = Path(
    __file__
).resolve().parents[1]

DEFAULT_DATASET = (
    ROOT
    / "data"
    / "raw"
    / "walker2d-medium-v2"
    / "walker2d_medium-v2.hdf5"
)

DEFAULT_OUTPUT_ROOT = (
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

EXPECTED_CLEAN_SHA256 = (
    "cf00f43add04c17fdfc2958dd581dea"
    "0851b2e5bedbe6fda073758a8f841aeda"
)

def _portable_path(
    path: Path,
) -> str:
    path = path.resolve()
    try:
        return str(
            path.relative_to(
                ROOT.resolve()
            )
        )
    except ValueError:
        return str(path)

def _rho_code(
    rho: float,
) -> str:
    percent = int(round(rho * 100))

    reconstructed = (percent / 100.0)

    if not np.isclose(
        rho,
        reconstructed,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "canonical rho values must "
            "be whole percentages"
        )

    return f"{percent:03d}"


def _parse_args():
    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--rhos",
        type=float,
        nargs="+",
        default=[
            0.0,
            0.01,
            0.05,
        ],
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DEFAULT_METADATA_ROOT,
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    clean_file_sha = sha256_file(
        args.dataset
    )

    if (
        clean_file_sha.lower()
        != EXPECTED_CLEAN_SHA256
    ):
        raise RuntimeError(
            "clean dataset SHA256 mismatch\n"
            f"expected: {EXPECTED_CLEAN_SHA256}\n"
            f"actual:   {clean_file_sha}"
        )

    print(
        "Loading clean dataset..."
    )

    clean_dataset = (
        load_hdf5_dataset(
            args.dataset
        )
    )

    clean_logical_sha = (
        logical_dataset_sha256(
            clean_dataset
        )
    )

    print(
        "Preparing CSDPC once for "
        f"attack seed {args.seed}..."
    )

    prepared = (
        prepare_csdpc_attack(
            clean_dataset,
            attack_seed=args.seed,
            num_clusters=8,
            sequence_length=5,
            eta=0.05,
            num_candidates=100,
        )
    )

    rhos = sorted(
        set(
            float(rho)
            for rho
            in args.rhos
        )
    )

    low_indices = None
    low_observations = None
    low_actions = None

    artifact_records = []

    cross_rho = {
        "checked": False,
        "low_indices_subset_of_high": None,
        "shared_observations_identical": None,
        "shared_actions_identical": None,
    }

    for rho in rhos:
        code = _rho_code(
            rho
        )

        print()
        print(
            "=" * 60
        )
        print(
            f"Applying rho={rho:.2%}, "
            f"seed={args.seed}"
        )
        print(
            "=" * 60
        )

        result = apply_csdpc_attack(
            clean_dataset,
            prepared,
            rho=rho,
        )

        preliminary_metadata = (
            build_csdpc_metadata(
                clean_dataset,
                prepared,
                result,
                clean_file_sha256=(
                    clean_file_sha
                ),
            )
        )

        if not preliminary_metadata[
            "integrity"
        ]["all_checks_passed"]:
            raise RuntimeError(
                "CSDPC integrity gate failed "
                f"for rho={rho}"
            )

        if rho == 0.0:
            if (
                preliminary_metadata[
                    "dataset"
                ][
                    "clean_logical_sha256"
                ]
                != preliminary_metadata[
                    "dataset"
                ][
                    "poisoned_logical_sha256"
                ]
            ):
                raise RuntimeError(
                    "rho=0 logical dataset "
                    "does not equal clean dataset"
                )

        output_path = (
            args.output_root
            / (
                f"rho_{code}"
                f"_seed_{args.seed}.hdf5"
            )
        )

        metadata_path = (
            args.metadata_root
            / (
                f"rho_{code}"
                f"_seed_{args.seed}.json"
            )
        )

        write_hdf5_dataset(
            output_path,
            result.poisoned_dataset,
        )

        poisoned_file_sha = (
            sha256_file(
                output_path
            )
        )

        metadata = (
            build_csdpc_metadata(
                clean_dataset,
                prepared,
                result,
                clean_file_sha256=(
                    clean_file_sha
                ),
                poisoned_file_sha256=(
                    poisoned_file_sha
                ),
            )
        )

        write_metadata_json(
            metadata_path,
            metadata,
        )

        artifact_records.append(
            {
                "rho": rho,
                "hdf5": _portable_path(
                    output_path
                ),
                "metadata": _portable_path(
                    metadata_path
                ),
                "logical_sha256": (
                    metadata[
                        "dataset"
                    ][
                        "poisoned_logical_sha256"
                    ]
                ),
                "file_sha256": (
                    poisoned_file_sha
                ),
                "actual_transition_budget": (
                    result.actual_transition_budget
                ),
            }
        )

        print(
            "Requested budget:",
            result.requested_transition_budget,
        )
        print(
            "Actual budget:",
            result.actual_transition_budget,
        )
        print(
            "Actual rho:",
            f"{result.actual_rho:.6%}",
        )

        effect = metadata["attack_effect"]

        print(
            "Target-frequency improved:",
            (f"{100.0 * effect['fraction_target_frequency_improved']:.2f}%"),
        )
        print(
            "Target pattern unchanged:",
            (f"{100.0 * effect['fraction_target_pattern_unchanged']:.2f}%"),
        )
        print(
            "Target pattern changed:",
            (f"{100.0 * effect['fraction_target_pattern_changed']:.2f}%"),
        )
        print(
            "Improved given pattern changed:",
            (f"{100.0 * effect['fraction_frequency_improved_given_pattern_changed']:.2f}%"),
        )
        print(
            "Mean source frequency:",
            effect["mean_source_frequency"],
        )
        print(
            "Mean target frequency:",
            effect["mean_target_frequency"],
        )
        print(
            "Median target/source ratio:",
            effect["median_target_source_frequency_ratio"],
        )
        print(
            "Logical SHA256:",
            metadata["dataset"]["poisoned_logical_sha256"],
        )

        if np.isclose(
            rho,
            0.01,
        ):
            low_indices = np.asarray(
                result.modified_transition_indices,
                dtype=np.int64,
            )

            low_observations = (
                result.poisoned_dataset[
                    "observations"
                ][
                    low_indices
                ].copy()
            )

            low_actions = (
                result.poisoned_dataset[
                    "actions"
                ][
                    low_indices
                ].copy()
            )

        if (
            np.isclose(
                rho,
                0.05,
            )
            and low_indices
            is not None
        ):
            high_indices = set(
                result.modified_transition_indices
            )

            subset = all(
                int(index)
                in high_indices
                for index
                in low_indices
            )

            if subset:
                shared_obs_equal = bool(
                    np.array_equal(
                        low_observations,
                        result.poisoned_dataset[
                            "observations"
                        ][
                            low_indices
                        ],
                    )
                )

                shared_actions_equal = bool(
                    np.array_equal(
                        low_actions,
                        result.poisoned_dataset[
                            "actions"
                        ][
                            low_indices
                        ],
                    )
                )
            else:
                shared_obs_equal = False
                shared_actions_equal = False

            cross_rho = {
                "checked": True,
                "low_indices_subset_of_high": (
                    subset
                ),
                "shared_observations_identical": (
                    shared_obs_equal
                ),
                "shared_actions_identical": (
                    shared_actions_equal
                ),
            }

            if not all(
                [
                    subset,
                    shared_obs_equal,
                    shared_actions_equal,
                ]
            ):
                raise RuntimeError(
                    "cross-rho consistency "
                    "gate failed"
                )

        del result
        gc.collect()

    manifest = {
        "schema_version": (
            "csdpc-seed-manifest-v1"
        ),
        "attack_seed": int(
            args.seed
        ),
        "clean_file_sha256": (
            clean_file_sha
        ),
        "clean_logical_sha256": (
            clean_logical_sha
        ),
        "kmeans_inertia": float(
            prepared.clustering.inertia
        ),
        "kmeans_inertia_is_reproducibility_gate": (
            False
        ),
        "kmeans_iterations": int(
            prepared.clustering.n_iter
        ),
        "cross_rho_consistency": (
            cross_rho
        ),
        "artifacts": (
            artifact_records
        ),
    }

    manifest_path = (
        args.metadata_root
        / (
            f"seed_{args.seed}"
            "_manifest.json"
        )
    )

    write_metadata_json(
        manifest_path,
        manifest,
    )

    print()
    print("========== SEED MANIFEST ==========")

    print(
        json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()