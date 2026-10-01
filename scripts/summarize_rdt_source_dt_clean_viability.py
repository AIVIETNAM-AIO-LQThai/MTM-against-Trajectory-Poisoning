from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    __file__
).resolve().parents[
    1
]

EXPERIMENT_ROOT = (
    ROOT
    / "experiments"
    / "attack_qualification"
    / "rdt_source_dt_legacy_mha_compat"
)

OUTPUT = (
    EXPERIMENT_ROOT
    / "clean_viability_summary.json"
)


def read_json(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            path
        )

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def main() -> None:
    preflight = read_json(
        EXPERIMENT_ROOT
        / "clean_preflight.json"
    )

    if (
        preflight[
            "status"
        ]
        != "PREFLIGHT_PASS"
    ):
        raise RuntimeError(
            "clean preflight is not PASS"
        )

    rows = []

    for seed in (
        0,
        1,
        2,
    ):
        run_dir = (
            EXPERIMENT_ROOT
            / "clean"
            / f"model_seed_{seed}"
        )

        failure_path = (
            run_dir
            / "failure.json"
        )

        if failure_path.exists():
            failure = (
                read_json(
                    failure_path
                )
            )

            rows.append(
                {
                    "seed": seed,
                    "pass": False,
                    "failure": (
                        failure
                    ),
                }
            )

            continue

        summary = read_json(
            run_dir
            / "summary.json"
        )

        passed = bool(
            summary[
                "status"
            ]
            == "complete"
            and int(
                summary[
                    "final_update"
                ]
            )
            == 100_000
            and summary[
                "loss_finite"
            ]
            and summary[
                "gradient_elements_finite"
            ]
            and summary[
                "model_parameters_finite"
            ]
            and summary[
                "optimizer_state_finite"
            ]
        )

        rows.append(
            {
                "seed": (
                    seed
                ),
                "pass": (
                    passed
                ),
                "summary": (
                    summary
                ),
            }
        )

    viability_pass = all(
        row[
            "pass"
        ]
        for row in rows
    )

    result = {
        "experiment": (
            "RDT_SOURCE_DT_LEGACY_MHA_COMPAT_CLEAN_VIABILITY"
        ),
        "seeds": rows,
        "clean_viability_pass": (
            viability_pass
        ),
        "corrupted_training_allowed": (
            viability_pass
        ),
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
        "=" * 92
    )

    print(
        "SOURCE-COMPATIBLE RDT DT — CLEAN VIABILITY"
    )

    print(
        "=" * 92
    )

    for row in rows:
        if row[
            "pass"
        ]:
            print(
                f"seed {row['seed']}: PASS — 100000 updates"
            )
        else:
            print(
                f"seed {row['seed']}: FAIL"
            )

            if (
                "failure"
                in row
            ):
                print(
                    "  reason:",
                    row[
                        "failure"
                    ][
                        "reason"
                    ],
                )

                print(
                    "  update:",
                    row[
                        "failure"
                    ][
                        "update"
                    ],
                )

    print()

    print(
        "CLEAN VIABILITY:",
        (
            "PASS"
            if viability_pass
            else "FAIL"
        ),
    )

    print(
        "corrupted training:",
        (
            "ALLOWED"
            if viability_pass
            else "BLOCKED"
        ),
    )

    print(
        "output ->",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
