from __future__ import annotations

from scripts.preflight_dt_stress_matrix import expected_artifacts
from scripts.summarize_dt_stress_transfer import classify


def test_expected_poison_matrix_has_twelve_artifacts():
    rows = expected_artifacts()
    assert len(rows) == 12
    assert len({str(row[3]) for row in rows}) == 12


def test_classification_rule_is_frozen_and_monotone():
    floor = 3.5
    assert classify(4.0, 3, floor) == "consistent degradation"
    assert classify(4.0, 2, floor) == "weak/inconsistent degradation"
    assert classify(3.0, 3, floor) == "weak/inconsistent degradation"
    assert classify(0.1, 1, floor) == "weak/inconsistent degradation"
    assert classify(0.0, 3, floor) == "no detectable degradation"
    assert classify(-1.0, 3, floor) == "no detectable degradation"
