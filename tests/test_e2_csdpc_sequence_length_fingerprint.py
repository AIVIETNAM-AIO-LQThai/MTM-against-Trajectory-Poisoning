import copy

import numpy as np
import pytest

from scripts.audit_e2_csdpc_sequence_length_fingerprint import (
    _check_regression_anchors,
    _reference_proximity,
)


def _config():
    return {
        "attack_seeds": [0, 1, 2],
        "actual_sequence_lengths": [
            2, 3, 4, 5, 6, 7, 8, 9, 10
        ],
        "source_reference": {
            "reference_fraction": 0.8,
            "approximate": True,
            "is_acceptance_gate": False,
        },
        "historical_regression_anchors": [
            {
                "attack_seed": 0,
                "actual_sequence_length": 5,
                "canonical_distinct_type_reduction_fraction": 0.42,
                "absolute_tolerance": 1e-6,
                "origin": "synthetic",
            },
            {
                "attack_seed": 0,
                "actual_sequence_length": 6,
                "canonical_distinct_type_reduction_fraction": 0.54,
                "absolute_tolerance": 1e-6,
                "origin": "synthetic",
            },
        ],
    }


def _seed_records():
    records = {}

    for seed in [0, 1, 2]:
        per_seed = {
            "seed": seed,
        }

        for length in range(2, 11):
            if length == 5:
                value = 0.42
            elif length == 6:
                value = 0.54
            else:
                value = min(
                    0.1 * length,
                    0.95,
                )

            per_seed[str(length)] = {
                "metrics": {
                    "canonical_distinct_type_reduction_fraction": value,
                }
            }

        records[str(seed)] = per_seed

    return records


def test_regression_anchors_pass():
    result = _check_regression_anchors(
        _config(),
        _seed_records(),
    )

    assert result["all_passed"]
    assert len(result["checks"]) == 2


def test_regression_anchor_failure_is_visible():
    records = _seed_records()

    records["0"]["5"]["metrics"][
        "canonical_distinct_type_reduction_fraction"
    ] = 0.43

    result = _check_regression_anchors(
        _config(),
        records,
    )

    assert not result["all_passed"]
    assert (
        result["checks"][0]["passed"]
        is False
    )


def test_reference_proximity_uses_all_lengths():
    config = _config()

    aggregate = {}

    for length in config["actual_sequence_lengths"]:
        aggregate[str(length)] = {
            "metrics": {
                "canonical_distinct_type_reduction_fraction": {
                    "mean": 0.1 * length,
                    "std": 0.0,
                }
            }
        }

    result = _reference_proximity(
        config, aggregate,
    )

    assert len(result["rows"]) == 9
    assert result["nearest_predeclared_length"]["actual_sequence_length"] == 8


def test_reference_is_not_gate():
    config = _config()

    aggregate = {}

    for length in config["actual_sequence_lengths"]:
        aggregate[str(length)] = {
            "metrics": {
                "canonical_distinct_type_reduction_fraction": {
                    "mean": 0.05 * length,
                    "std": 0.0,
                }
            }
        }

    result = _reference_proximity(
        config, aggregate,
    )

    assert result["reference_is_acceptance_gate"] is False


def test_monotonic_curve_detected():
    config = _config()

    aggregate = {}

    for length in config["actual_sequence_lengths"]:
        aggregate[str(length)] = {
            "metrics": {
                "canonical_distinct_type_reduction_fraction": {
                    "mean": 0.05 * length,
                    "std": 0.0,
                }
            }
        }

    result = _reference_proximity(
        config, aggregate,
    )

    assert result["mean_curve_monotonic_nondecreasing"] is True


def test_nonmonotonic_curve_detected():
    config = _config()

    values = {
        2: 0.1,
        3: 0.2,
        4: 0.3,
        5: 0.4,
        6: 0.5,
        7: 0.45,
        8: 0.6,
        9: 0.7,
        10: 0.8,
    }

    aggregate = {}

    for length, value in values.items():
        aggregate[str(length)] = {
            "metrics": {
                "canonical_distinct_type_reduction_fraction": {
                    "mean": value,
                    "std": 0.0,
                }
            }
        }

    result = _reference_proximity(
        config, aggregate
    )

    assert result["mean_curve_monotonic_nondecreasing"] is False
