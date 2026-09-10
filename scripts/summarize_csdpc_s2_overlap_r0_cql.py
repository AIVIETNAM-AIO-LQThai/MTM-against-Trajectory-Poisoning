from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.attacks.csdpc.metadata import (
    logical_dataset_sha256,
    sha256_file,
    write_metadata_json,
)
from src.data.hdf5_io import (
    load_hdf5_dataset,
)


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "gates"
    / "csdpc_s2_overlap_r0_cql_sensitivity.json"
)


def _read_json(path: Path):
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def _rho_code(rho: float) -> str:
    return f"{int(round(float(rho) * 100)):03d}"


def _evaluation_return_column(frame):
    """
    Return the frozen Gate-B raw evaluation-return column.

    The canonical CQL runs used:
        evaluation/Returns Mean

    Do not substitute D4RL-normalized scores or another
    evaluation statistic when reproducing the frozen
    Gate-B late-window raw-return metric.
    """
    frozen_column = (
        "evaluation/Returns Mean"
    )

    if (
        frozen_column
        not in frame.columns
    ):
        evaluation_columns = [
            column
            for column
            in frame.columns
            if (
                "evaluation"
                in column.lower()
            )
        ]

        raise RuntimeError(
            "Frozen Gate-B raw-return "
            "column is missing: "
            f"{frozen_column}. "
            "Available evaluation columns: "
            f"{evaluation_columns}"
        )

    return frozen_column


def _late_stats(
    run_dir: Path,
    start_epoch: int,
    end_epoch: int,
):
    progress_path = (
        run_dir
        / "progress.csv"
    )

    if not progress_path.exists():
        raise FileNotFoundError(
            progress_path
        )

    frame = pd.read_csv(
        progress_path
    )

    column = (
        _evaluation_return_column(
            frame
        )
    )

    values = (
        pd.to_numeric(
            frame[column],
            errors="coerce",
        )
        .dropna()
        .to_numpy(
            dtype=np.float64
        )
    )

    if (
        len(values)
        <= end_epoch
    ):
        raise RuntimeError(
            f"{run_dir}: only "
            f"{len(values)} evaluation rows; "
            f"need epoch {end_epoch}"
        )

    late = values[
        start_epoch:
        end_epoch + 1
    ]

    expected_count = (
        end_epoch
        - start_epoch
        + 1
    )

    if (
        len(late)
        != expected_count
    ):
        raise RuntimeError(
            "late-window length mismatch"
        )

    return {
        "late_mean": float(
            np.mean(late)
        ),
        "late_std": float(
            np.std(
                late,
                ddof=0,
            )
        ),
        "final": float(
            values[end_epoch]
        ),
        "best": float(
            np.max(values)
        ),
        "num_evaluation_rows": int(
            len(values)
        ),
        "return_column": column,
    }


def _artifact_record(
    config,
    seed: int,
    rho: float,
):
    manifest_path = (
        ROOT
        / config[
            "artifact_metadata_root"
        ]
        / f"seed_{seed}_manifest.json"
    )

    manifest = _read_json(
        manifest_path
    )

    matches = [
        record
        for record
        in manifest["artifacts"]
        if np.isclose(
            float(record["rho"]),
            float(rho),
            rtol=0.0,
            atol=1.0e-12,
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            "Artifact manifest lookup "
            f"failed for seed={seed}, "
            f"rho={rho}"
        )

    return matches[0]


def _verify_clean_reference(
    config,
    gate_reference,
):
    """
    Verify the frozen clean CQL reference using committed
    Gate-B metadata, not mutable local training outputs.

    The canonical clean CQL runs have already completed and
    passed B2. They must not be rerun or reconstructed here.
    """
    clean_reference_path = (
        ROOT
        / config[
            "canonical_clean_reference"
        ]
    )

    if not clean_reference_path.exists():
        raise FileNotFoundError(
            clean_reference_path
        )

    clean_reference = _read_json(
        clean_reference_path
    )

    if (
        clean_reference.get("gate")
        != "B2_CLEAN_CQL"
    ):
        raise RuntimeError(
            "Unexpected canonical clean "
            "reference gate"
        )

    if (
        clean_reference.get("status")
        != "PASS"
    ):
        raise RuntimeError(
            "Frozen clean CQL reference "
            "did not pass B2"
        )

    if int(
        clean_reference[
            "training_epochs"
        ]
    ) != 500:
        raise RuntimeError(
            "Frozen clean CQL training "
            "horizon changed"
        )

    if list(
        clean_reference[
            "late_window"
        ]
    ) != [400, 499]:
        raise RuntimeError(
            "Frozen clean CQL late "
            "window changed"
        )

    clean_by_seed = {
        str(
            int(record["seed"])
        ): record
        for record
        in clean_reference[
            "seeds"
        ]
    }

    expected_seeds = [
        str(int(seed))
        for seed
        in config[
            "model_seeds"
        ]
    ]

    if (
        sorted(clean_by_seed)
        != sorted(expected_seeds)
    ):
        raise RuntimeError(
            "Frozen clean CQL seed "
            "set changed"
        )

    verified = {}

    for seed in (
        config["model_seeds"]
    ):
        seed = int(seed)
        key = str(seed)

        b2_record = (
            clean_by_seed[key]
        )

        b2_late_mean = float(
            b2_record[
                "late_mean"
            ]
        )

        gate_b_late_mean = float(
            gate_reference[
                "runs"
            ][
                "clean"
            ][key][
                "late_mean"
            ]
        )

        if not np.isclose(
            b2_late_mean,
            gate_b_late_mean,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise RuntimeError(
                "Frozen clean references "
                "disagree: "
                f"seed={seed}, "
                f"B2={b2_late_mean}, "
                f"GateB={gate_b_late_mean}"
            )

        if not np.isfinite(
            b2_late_mean
        ):
            raise RuntimeError(
                "Non-finite frozen clean "
                f"return for seed={seed}"
            )

        verified[key] = {
            "late_mean": (
                b2_late_mean
            ),
            "late_std": float(
                b2_record[
                    "late_std"
                ]
            ),
            "final": float(
                b2_record[
                    "final"
                ]
            ),
            "best": float(
                b2_record[
                    "best"
                ]
            ),
            "source": (
                "frozen_committed_"
                "b2_clean_summary"
            ),
        }

    return verified


def _preflight(config):
    gate_reference = _read_json(
        ROOT
        / config[
            "canonical_gate_b_reference"
        ]
    )

    print(
        "Checking frozen clean "
        "CQL reference..."
    )

    _verify_clean_reference(
        config,
        gate_reference,
    )

    print(
        "Frozen clean CQL reference: PASS"
    )

    for seed in (
        config["attack_seeds"]
    ):
        seed = int(seed)

        for rho in (
            config["poison_rates"]
        ):
            rho = float(rho)

            record = _artifact_record(
                config,
                seed,
                rho,
            )

            artifact = (
                ROOT
                / record[
                    "artifact"
                ]
            )

            if not artifact.exists():
                raise FileNotFoundError(
                    artifact
                )

            observed_file_sha = (
                sha256_file(
                    artifact
                )
            )

            if (
                observed_file_sha
                != record[
                    "file_sha256"
                ]
            ):
                raise RuntimeError(
                    "Artifact file SHA256 "
                    "mismatch: "
                    f"{artifact}"
                )

            dataset = (
                load_hdf5_dataset(
                    artifact
                )
            )

            observed_logical_sha = (
                logical_dataset_sha256(
                    dataset
                )
            )

            del dataset

            if (
                observed_logical_sha
                != record[
                    "logical_sha256"
                ]
            ):
                raise RuntimeError(
                    "Artifact logical SHA256 "
                    "mismatch: "
                    f"{artifact}"
                )

            print(
                "artifact PASS:",
                f"seed={seed}",
                f"rho={rho:.2f}",
                observed_logical_sha,
            )

    output_root = (
        ROOT
        / config[
            "run_output_root"
        ]
    )

    if (
        output_root.exists()
        and any(
            output_root.iterdir()
        )
    ):
        raise RuntimeError(
            "Sensitivity output directory "
            "already contains results: "
            f"{output_root}"
        )

    print()
    print(
        "S2/R0 CQL PREFLIGHT: PASS"
    )


def _interpret(
    mean_degradation: float,
    all_seed_pairs_degrade: bool,
):
    if (
        mean_degradation
        >= 0.50
        and all_seed_pairs_degrade
    ):
        return (
            "STRONG_EFFECT_UNDER_"
            "REUSED_GATE_B_RULE"
        )

    if (
        (
            0.25
            <= mean_degradation
            < 0.50
        )
        or not (
            all_seed_pairs_degrade
        )
    ):
        return (
            "INCONCLUSIVE_EFFECT_UNDER_"
            "REUSED_GATE_B_RULE"
        )

    return (
        "WEAK_EFFECT_UNDER_"
        "REUSED_GATE_B_RULE"
    )


def _summarize(config):
    gate_reference = _read_json(
        ROOT
        / config[
            "canonical_gate_b_reference"
        ]
    )

    clean_stats = (
        _verify_clean_reference(
            config,
            gate_reference,
        )
    )

    metric = config[
        "primary_metric"
    ]

    start = int(
        metric["start_epoch"]
    )

    end = int(
        metric["end_epoch"]
    )

    rho_results = {}

    for rho in (
        config["poison_rates"]
    ):
        rho = float(rho)
        code = _rho_code(rho)

        per_seed = {}
        degradations = []

        for seed in (
            config["attack_seeds"]
        ):
            seed = int(seed)

            run_name = (
                f"csdpc_rho_{code}"
                f"_attack_seed_{seed}"
                f"_model_seed_{seed}"
            )

            run_dir = (
                ROOT
                / config[
                    "run_output_root"
                ]
                / run_name
            )

            poison_stats = (
                _late_stats(
                    run_dir,
                    start,
                    end,
                )
            )

            result_path = (
                run_dir
                / "result.json"
            )

            manifest_path = (
                run_dir
                / "manifest.json"
            )

            if (
                not result_path.exists()
                or not manifest_path.exists()
            ):
                raise FileNotFoundError(
                    f"Incomplete CQL run: "
                    f"{run_dir}"
                )

            result = _read_json(
                result_path
            )

            run_manifest = _read_json(
                manifest_path
            )

            artifact_record = (
                _artifact_record(
                    config,
                    seed,
                    rho,
                )
            )

            expected_dataset_sha = (
                artifact_record[
                    "logical_sha256"
                ]
            )

            observed_dataset_sha = (
                run_manifest[
                    "dataset_logical_sha256"
                ]
            )

            if (
                observed_dataset_sha
                != expected_dataset_sha
            ):
                raise RuntimeError(
                    "CQL run used wrong "
                    "dataset artifact: "
                    f"seed={seed}, rho={rho}"
                )

            clean_mean = float(
                clean_stats[
                    str(seed)
                ][
                    "late_mean"
                ]
            )

            poison_mean = float(
                poison_stats[
                    "late_mean"
                ]
            )

            degradation = (
                (
                    clean_mean
                    - poison_mean
                )
                / abs(
                    clean_mean
                )
            )

            degradations.append(
                degradation
            )

            evaluation = result[
                "evaluation"
            ]

            per_seed[
                str(seed)
            ] = {
                "clean_late_mean": (
                    clean_mean
                ),
                "poison_late_mean": (
                    poison_mean
                ),
                "poison_late_std": (
                    poison_stats[
                        "late_std"
                    ]
                ),
                "poison_final_progress_return": (
                    poison_stats[
                        "final"
                    ]
                ),
                "poison_best_progress_return": (
                    poison_stats[
                        "best"
                    ]
                ),
                "final_evaluation_raw_return": float(
                    evaluation[
                        "mean_raw_return"
                    ]
                ),
                "final_evaluation_normalized_return": float(
                    evaluation[
                        "d4rl_normalized_return"
                    ]
                ),
                "paired_degradation": float(
                    degradation
                ),
                "dataset_logical_sha256": (
                    observed_dataset_sha
                ),
            }

        degradation_array = np.asarray(
            degradations,
            dtype=np.float64,
        )

        all_degrade = bool(
            np.all(
                degradation_array
                > 0.0
            )
        )

        mean_degradation = float(
            np.mean(
                degradation_array
            )
        )

        canonical_key = (
            "rho_"
            + code
        )

        canonical_mean = float(
            gate_reference[
                canonical_key
            ][
                "mean_paired_degradation"
            ]
        )

        rho_results[
            canonical_key
        ] = {
            "per_seed": per_seed,
            "mean_paired_degradation": (
                mean_degradation
            ),
            "std_paired_degradation": float(
                np.std(
                    degradation_array,
                    ddof=0,
                )
            ),
            "all_seed_pairs_degrade": (
                all_degrade
            ),
            "canonical_mean_paired_degradation": (
                canonical_mean
            ),
            "mean_degradation_change_vs_canonical": float(
                mean_degradation
                - canonical_mean
            ),
        }

    primary = (
        rho_results[
            "rho_"
            + _rho_code(
                config[
                    "primary_poison_rate"
                ]
            )
        ]
    )

    verdict = _interpret(
        primary[
            "mean_paired_degradation"
        ],
        primary[
            "all_seed_pairs_degrade"
        ],
    )

    summary = {
        "schema_version": (
            "csdpc-s2-overlap-r0-"
            "cql-sensitivity-v1"
        ),
        "experiment": (
            config["experiment"]
        ),
        "status": (
            config["status"]
        ),
        "variant_id": (
            config["variant_id"]
        ),
        "canonical_gate_b_verdict": (
            gate_reference[
                "verdict"
            ]
        ),
        "canonical_gate_b_unchanged": True,
        "primary_metric": metric,
        "rho_results": rho_results,
        "sensitivity_verdict": verdict,
    }

    output_path = (
        ROOT
        / config[
            "summary_output"
        ]
    )

    write_metadata_json(
        output_path,
        summary,
    )

    print()
    print(
        "=" * 72
    )
    print(
        "CSDPC S2/OVERLAP/R0 "
        "CQL SENSITIVITY"
    )
    print(
        "=" * 72
    )

    for key, record in (
        rho_results.items()
    ):
        print()
        print(key)

        for seed, seed_record in (
            record[
                "per_seed"
            ].items()
        ):
            print(
                f"  seed {seed}: "
                f"clean="
                f"{seed_record['clean_late_mean']:.3f}, "
                f"poison="
                f"{seed_record['poison_late_mean']:.3f}, "
                f"degradation="
                f"{100.0 * seed_record['paired_degradation']:.3f}%"
            )

        print(
            "  mean degradation:",
            f"{100.0 * record['mean_paired_degradation']:.3f}%",
        )

        print(
            "  canonical mean:",
            f"{100.0 * record['canonical_mean_paired_degradation']:.3f}%",
        )

        print(
            "  change vs canonical:",
            f"{100.0 * record['mean_degradation_change_vs_canonical']:.3f}",
            "percentage points",
        )

        print(
            "  all pairs degrade:",
            record[
                "all_seed_pairs_degrade"
            ],
        )

    print()
    print(
        "SENSITIVITY VERDICT:",
        verdict,
    )

    print(
        "CANONICAL GATE B:",
        gate_reference[
            "verdict"
        ],
        "(UNCHANGED)",
    )

    print(
        "summary:",
        output_path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--preflight",
        action="store_true",
    )

    args = parser.parse_args()

    config = _read_json(
        args.config
    )

    if (
        config.get(
            "experiment"
        )
        != "CSDPC_S2_OVERLAP_R0_CQL_SENSITIVITY"
    ):
        raise ValueError(
            "unexpected experiment"
        )

    if (
        config.get(
            "status"
        )
        != "SENSITIVITY_ONLY"
    ):
        raise ValueError(
            "status must remain "
            "SENSITIVITY_ONLY"
        )

    if not bool(
        config.get(
            "canonical_gate_b_unchanged",
            False,
        )
    ):
        raise ValueError(
            "canonical Gate B must "
            "remain unchanged"
        )

    if args.preflight:
        _preflight(
            config
        )
    else:
        _summarize(
            config
        )


if __name__ == "__main__":
    main()