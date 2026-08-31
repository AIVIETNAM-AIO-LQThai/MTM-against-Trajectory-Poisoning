import json
from pathlib import Path

import numpy as np


REFERENCE_MEAN = 74.0
REFERENCE_STD = 1.4


def load_summary(seed):
    path = Path(
        "experiments/dt/"
        "walker2d_medium_clean/"
        f"seed_{seed}/full/"
        "eval_5000/"
        "summary.json"
    )

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def main():
    seed_results = []

    for seed in (
        0,
        1,
        2,
    ):
        summary = load_summary(
            seed
        )

        score = float(
            summary[
                "normalized_return_mean"
            ]
        )

        seed_results.append(
            score
        )

        print(
            f"seed {seed}: "
            f"{score:.4f}"
        )

    scores = np.asarray(
        seed_results,
        dtype=np.float64,
    )

    our_mean = float(
        scores.mean()
    )

    our_std = float(
        scores.std()
    )

    absolute_difference = abs(
        our_mean
        - REFERENCE_MEAN
    )

    relative_error = (
        absolute_difference
        / REFERENCE_MEAN
    )

    print()
    print(
        "Our clean DT:"
    )

    print(
        f"{our_mean:.4f} "
        f"+/- {our_std:.4f}"
    )

    print()
    print(
        "Reference:"
    )

    print(
        f"{REFERENCE_MEAN:.4f} "
        f"+/- {REFERENCE_STD:.4f}"
    )

    print()
    print(
        "Absolute difference:",
        absolute_difference,
    )

    print(
        "Relative reproduction error:",
        relative_error,
    )

    if relative_error <= 0.10:
        verdict = "PASS"

    elif relative_error <= 0.15:
        verdict = "INCONCLUSIVE"

    else:
        verdict = "FAIL"

    print()
    print(
        "GATE A:",
        verdict,
    )


if __name__ == "__main__":
    main()