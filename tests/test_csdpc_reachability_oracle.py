from __future__ import annotations

import numpy as np

from scripts.audit_csdpc_reachability_oracle import (
    _cluster_reachable,
    _reachable_patterns,
    _transition_box,
)


def test_transition_box_matches_relative_linf_rule():
    observation = np.asarray(
        [2.0, -4.0],
        dtype=np.float32,
    )

    action = np.asarray(
        [0.5, -1.0],
        dtype=np.float32,
    )

    lower, upper = _transition_box(
        observation,
        action,
        eta=0.05,
        action_low=-1.0,
        action_high=1.0,
    )

    # State scale:
    # 0.05 * max(2,4) = 0.2
    np.testing.assert_allclose(
        lower[:2],
        [1.8, -4.2],
    )

    np.testing.assert_allclose(
        upper[:2],
        [2.2, -3.8],
    )

    # Action scale:
    # 0.05 * max(.5,1) = .05
    # second lower clips naturally at -1.
    np.testing.assert_allclose(
        lower[2:],
        [0.45, -1.0],
    )

    np.testing.assert_allclose(
        upper[2:],
        [0.55, -0.95],
    )


def test_distant_cluster_is_not_reachable():
    centers = np.asarray(
        [
            [0.0],
            [10.0],
        ],
        dtype=np.float64,
    )

    assert not _cluster_reachable(
        target_cluster=1,
        source_cluster=0,
        centers=centers,
        lower=np.asarray(
            [-1.0]
        ),
        upper=np.asarray(
            [1.0]
        ),
    )


def test_target_cluster_is_reachable_when_box_crosses_boundary():
    centers = np.asarray(
        [
            [0.0],
            [10.0],
        ],
        dtype=np.float64,
    )

    assert _cluster_reachable(
        target_cluster=1,
        source_cluster=0,
        centers=centers,
        lower=np.asarray(
            [4.9]
        ),
        upper=np.asarray(
            [6.0]
        ),
    )


def test_reachable_patterns_deduplicate_dynamically():
    reachable = [
        (0, 1),
        (1,),
        (1, 2),
    ]

    patterns = _reachable_patterns(
        reachable
    )

    assert (0, 1) in patterns
    assert (0, 1, 2) in patterns
    assert (1,) in patterns
    assert (1, 2) in patterns

    # Consecutive duplicate labels must
    # never appear in the final pattern.
    for pattern in patterns:
        assert all(
            left != right
            for left, right
            in zip(
                pattern,
                pattern[1:],
            )
        )