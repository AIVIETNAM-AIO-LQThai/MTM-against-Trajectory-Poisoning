from pathlib import Path

import numpy as np
import pandas as pd
import json

from scripts.summarize_csdpc_s2_overlap_r0_cql import (
    _evaluation_return_column,
    _interpret,
    _late_stats,
)


def test_evaluation_return_column():
    frame = pd.DataFrame(
        {
            "evaluation/Returns Mean": [
                1000.0,
                2000.0,
            ],
            "evaluation/Average Returns": [
                20.0,
                40.0,
            ],
        }
    )

    assert (
        _evaluation_return_column(
            frame
        )
        == "evaluation/Returns Mean"
    )

def test_evaluation_return_column_rejects_missing_frozen_metric():
    frame = pd.DataFrame(
        {
            "evaluation/Average Returns": [
                20.0,
                40.0,
            ]
        }
    )

    try:
        _evaluation_return_column(
            frame
        )
    except RuntimeError as exc:
        assert (
            "evaluation/Returns Mean"
            in str(exc)
        )
    else:
        raise AssertionError(
            "Missing frozen raw-return "
            "column should fail"
        )


def test_late_stats_uses_epochs_400_to_499(
    tmp_path: Path,
):
    run_dir = (
        tmp_path
        / "run"
    )

    run_dir.mkdir()

    values = np.arange(
        500,
        dtype=np.float64,
    )

    pd.DataFrame(
        {
            "evaluation/Returns Mean": (
                values
            )
        }
    ).to_csv(
        run_dir
        / "progress.csv",
        index=False,
    )

    result = _late_stats(
        run_dir,
        400,
        499,
    )

    assert np.isclose(
        result[
            "late_mean"
        ],
        449.5,
    )

    assert np.isclose(
        result[
            "final"
        ],
        499.0,
    )

    assert (
        result[
            "return_column"
        ]
        == "evaluation/Returns Mean"
    )

def test_reused_gate_b_interpretation():
    assert (
        _interpret(
            0.60,
            True,
        )
        == "STRONG_EFFECT_UNDER_REUSED_GATE_B_RULE"
    )

    assert (
        _interpret(
            0.30,
            True,
        )
        == "INCONCLUSIVE_EFFECT_UNDER_REUSED_GATE_B_RULE"
    )

    assert (
        _interpret(
            0.60,
            False,
        )
        == "INCONCLUSIVE_EFFECT_UNDER_REUSED_GATE_B_RULE"
    )

    assert (
        _interpret(
            0.10,
            True,
        )
        == "WEAK_EFFECT_UNDER_REUSED_GATE_B_RULE"
    )