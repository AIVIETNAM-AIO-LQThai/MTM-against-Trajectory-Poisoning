from __future__ import annotations

import argparse
from pathlib import Path

from src.baselines.cql.runner import (
    DEFAULT_CONFIG,
    DEFAULT_CQL_ROOT,
    run_cql,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--condition",
        choices=[
            "clean",
            "csdpc",
        ],
        required=True,
    )

    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--attack-seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--cql-root",
        type=Path,
        default=DEFAULT_CQL_ROOT,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "outputs/g2/cql"
        ),
    )

    args = parser.parse_args()

    result = run_cql(
        dataset_path=args.dataset,
        seed=args.seed,
        condition=args.condition,
        rho=args.rho,
        attack_seed=args.attack_seed,
        output_root=args.output_root,
        config_path=args.config,
        cql_root=args.cql_root,
        gpu_id=args.gpu,
    )

    evaluation = result[
        "evaluation"
    ]

    print()
    print("CQL RUN COMPLETE")
    print(
        "raw return:",
        evaluation[
            "mean_raw_return"
        ],
    )
    print(
        "normalized return:",
        evaluation[
            "d4rl_normalized_return"
        ],
    )


if __name__ == "__main__":
    main()