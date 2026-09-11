import numpy as np

from scripts.audit_csdpc_source_environment import (
    _infer_legacy_timeouts,
)


def test_legacy_timeout_at_1000th_transition():
    terminals = np.zeros(
        1000,
        dtype=np.float32,
    )

    timeouts = _infer_legacy_timeouts(
        terminals,
        1000,
    )

    assert np.flatnonzero(
        timeouts
    ).tolist() == [999]


def test_no_timeout_before_horizon():
    terminals = np.zeros(
        999,
        dtype=np.float32,
    )

    timeouts = _infer_legacy_timeouts(
        terminals,
        1000,
    )

    assert not np.any(timeouts)


def test_terminal_resets_legacy_counter():
    terminals = np.zeros(
        1004,
        dtype=np.float32,
    )

    terminals[4] = 1.0

    timeouts = _infer_legacy_timeouts(
        terminals,
        1000,
    )

    assert np.flatnonzero(
        timeouts
    ).tolist() == [1003]
