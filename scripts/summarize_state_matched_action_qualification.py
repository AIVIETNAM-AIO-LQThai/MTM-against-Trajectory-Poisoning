from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CONFIG = (
    ROOT
    / "configs"
    / "attack_qualification"
    / "a5_state_matched_low_return_action_substitution.json"
)

OUTROOT = (
    ROOT
    / "experiments"
    / "attack_qualification"
    / "state_matched_action_substitution"
)

FROZEN_CLEAN_ROOT = (
    ROOT
    / "results"
    / "attack_qualification"
    / "a2_rtg_inflation"
    / "clean"
)

PREFLIGHT = (
    OUTROOT
    / "preflight.json"
)

OUTPUT = (
    OUTROOT
    / "qualification_summary.json"
)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main():
    config = read_json(
        CONFIG
    )

    preflight = read_json(
        PREFLIGHT
    )

    if (
        preflight.get("status")
        != "PREFLIGHT_PASS"
    ):
        raise RuntimeError(
            "qualification preflight is not PASS"
        )

    qualification = config[
        "qualification"
    ]

    model_seeds = [
        int(x)
        for x
        in qualification[
            "model_seeds"
        ]
    ]

    attack_seeds = [
        int(x)
        for x
        in qualification[
            "attack_seeds"
        ]
    ]

    if model_seeds != [0, 1, 2]:
        raise RuntimeError(
            "model seeds changed"
        )

    if attack_seeds != [30, 31, 32]:
        raise RuntimeError(
            "attack seeds changed"
        )

    clean = {}

    for model_seed in model_seeds:
        path = (
            FROZEN_CLEAN_ROOT
            / f"model_seed_{model_seed}"
            / "eval_summary.json"
        )

        record = read_json(
            path
        )

        if (
            int(
                record[
                    "training_seed"
                ]
            )
            != model_seed
        ):
            raise RuntimeError(
                "clean seed mismatch"
            )

        if (
            int(
                record[
                    "training_update"
                ]
            )
            != 100000
        ):
            raise RuntimeError(
                "clean checkpoint mismatch"
            )

        if (
            int(
                record[
                    "num_episodes"
                ]
            )
            != 100
            or int(
                record[
                    "eval_seed_base"
                ]
            )
            != 30000
        ):
            raise RuntimeError(
                "clean evaluation protocol mismatch"
            )

        clean[
            model_seed
        ] = float(
            record[
                "normalized_return_mean"
            ]
        )

    clean_mean = float(
        np.mean(
            [
                clean[s]
                for s in model_seeds
            ]
        )
    )

    expected_clean_mean = float(
        qualification[
            "clean_control_mean"
        ]
    )

    if not np.isclose(
        clean_mean,
        expected_clean_mean,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "clean mean changed"
        )

    rows = []
    matrix = {}

    for attack_seed in attack_seeds:
        matrix[
            str(
                attack_seed
            )
        ] = {}

        for model_seed in model_seeds:
            path = (
                OUTROOT
                / "poison"
                / f"attack_seed_{attack_seed}"
                / f"model_seed_{model_seed}"
                / "eval"
                / "summary.json"
            )

            record = read_json(
                path
            )

            if (
                record.get(
                    "condition"
                )
                != (
                    "state_matched_low_return_"
                    "action_substitution"
                )
            ):
                raise RuntimeError(
                    "evaluation condition mismatch"
                )

            if (
                int(
                    record[
                        "attack_seed"
                    ]
                )
                != attack_seed
            ):
                raise RuntimeError(
                    "evaluation attack seed mismatch"
                )

            if (
                int(
                    record[
                        "training_seed"
                    ]
                )
                != model_seed
            ):
                raise RuntimeError(
                    "evaluation model seed mismatch"
                )

            if (
                int(
                    record[
                        "training_update"
                    ]
                )
                != 100000
            ):
                raise RuntimeError(
                    "evaluation checkpoint is not 100k"
                )

            if (
                int(
                    record[
                        "num_episodes"
                    ]
                )
                != 100
                or int(
                    record[
                        "eval_seed_base"
                    ]
                )
                != 30000
            ):
                raise RuntimeError(
                    "evaluation protocol mismatch"
                )

            poison_return = float(
                record[
                    "normalized_return_mean"
                ]
            )

            degradation = float(
                clean[
                    model_seed
                ]
                - poison_return
            )

            row = {
                "attack_seed": (
                    attack_seed
                ),
                "model_seed": (
                    model_seed
                ),
                "clean_return": float(
                    clean[
                        model_seed
                    ]
                ),
                "poison_return": (
                    poison_return
                ),
                "degradation": (
                    degradation
                ),
                "positive_degradation": bool(
                    degradation > 0.0
                ),
            }

            rows.append(
                row
            )

            matrix[
                str(
                    attack_seed
                )
            ][
                str(
                    model_seed
                )
            ] = row

    deltas = np.asarray(
        [
            row[
                "degradation"
            ]
            for row in rows
        ],
        dtype=np.float64,
    )

    mean_delta = float(
        deltas.mean()
    )

    median_delta = float(
        np.median(
            deltas
        )
    )

    positive_cells = int(
        np.sum(
            deltas > 0.0
        )
    )

    model_seed_means = {}

    for model_seed in model_seeds:
        model_seed_means[
            str(
                model_seed
            )
        ] = float(
            np.mean(
                [
                    row[
                        "degradation"
                    ]
                    for row in rows
                    if (
                        row[
                            "model_seed"
                        ]
                        == model_seed
                    )
                ]
            )
        )

    attack_seed_means = {}

    for attack_seed in attack_seeds:
        attack_seed_means[
            str(
                attack_seed
            )
        ] = float(
            np.mean(
                [
                    row[
                        "degradation"
                    ]
                    for row in rows
                    if (
                        row[
                            "attack_seed"
                        ]
                        == attack_seed
                    )
                ]
            )
        )

    gate = qualification[
        "gate"
    ]

    floor = float(
        gate[
            "required_mean_degradation"
        ]
    )

    expected_floor = float(
        gate[
            "mean_degradation_fraction_of_clean_mean"
        ]
        * clean_mean
    )

    if not np.isclose(
        floor,
        expected_floor,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "qualification floor changed"
        )

    effect_pass = bool(
        mean_delta >= floor
    )

    positive_cells_pass = bool(
        positive_cells
        >= int(
            gate[
                "minimum_positive_cells"
            ]
        )
    )

    model_seed_pass = bool(
        all(
            value > 0.0
            for value
            in model_seed_means.values()
        )
    )

    attack_seed_pass = bool(
        all(
            value > 0.0
            for value
            in attack_seed_means.values()
        )
    )

    qualification_pass = bool(
        effect_pass
        and positive_cells_pass
        and model_seed_pass
        and attack_seed_pass
    )

    result = {
        "experiment": (
            "VANILLA_DT_STATE_MATCHED_"
            "ACTION_SUBSTITUTION_QUALIFICATION"
        ),
        "attack_name": (
            config[
                "name"
            ]
        ),
        "clean_controls_reused_from": (
            "frozen A2 clean controls"
        ),
        "clean_returns": {
            str(key): value
            for key, value
            in clean.items()
        },
        "clean_mean": (
            clean_mean
        ),
        "degradation_floor": (
            floor
        ),
        "mean_degradation": (
            mean_delta
        ),
        "median_degradation": (
            median_delta
        ),
        "positive_cells": (
            positive_cells
        ),
        "total_cells": len(
            rows
        ),
        "model_seed_mean_degradation": (
            model_seed_means
        ),
        "attack_seed_mean_degradation": (
            attack_seed_means
        ),
        "gate_components": {
            "effect_floor_pass": (
                effect_pass
            ),
            "positive_cells_pass": (
                positive_cells_pass
            ),
            "model_seed_consistency_pass": (
                model_seed_pass
            ),
            "attack_seed_consistency_pass": (
                attack_seed_pass
            ),
        },
        "qualification_pass": (
            qualification_pass
        ),
        "matrix": matrix,
    }

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
        "=" * 104
    )

    print(
        "VANILLA DT STATE-MATCHED ACTION "
        "SUBSTITUTION QUALIFICATION"
    )

    print(
        "=" * 104
    )

    print(
        "clean:",
        {
            key: round(
                value,
                6,
            )
            for key, value
            in clean.items()
        },
    )

    print(
        f"clean mean: "
        f"{clean_mean:.6f}"
    )

    print()

    print(
        "attack_seed model_seed clean "
        "poison degradation"
    )

    for row in rows:
        print(
            f"{row['attack_seed']:11d} "
            f"{row['model_seed']:10d} "
            f"{row['clean_return']:7.3f} "
            f"{row['poison_return']:7.3f} "
            f"{row['degradation']:+11.3f}"
        )

    print()

    print(
        f"mean degradation:   "
        f"{mean_delta:+.6f}"
    )

    print(
        f"median degradation: "
        f"{median_delta:+.6f}"
    )

    print(
        f"required floor:     "
        f"{floor:.6f}"
    )

    print(
        f"positive cells:     "
        f"{positive_cells}/9"
    )

    print(
        "model-seed means:",
        model_seed_means,
    )

    print(
        "attack-seed means:",
        attack_seed_means,
    )

    print()

    print(
        "ATTACK QUALIFICATION:",
        (
            "PASS"
            if qualification_pass
            else "FAIL"
        ),
    )

    print(
        "output ->",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
