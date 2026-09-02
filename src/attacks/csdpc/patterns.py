from __future__ import annotations

from collections import Counter
from typing import Counter as CounterType
from typing import Iterable, Iterator

import numpy as np

from src.data.trajectories import TrajectorySlice

from .types import Pattern, SequenceWindow


def deduplicate_consecutive(
    labels: Iterable[int],
) -> Pattern:
    """
    Remove consecutive duplicate cluster labels.

    Examples
    --------
    [1, 1, 1, 4, 4] -> (1, 4)
    [1, 1, 4, 1, 1] -> (1, 4, 1)

    Non-consecutive repetitions are preserved.
    """

    values = tuple(int(label) for label in labels)

    if not values:
        return tuple()

    deduplicated = [values[0]]

    for label in values[1:]:
        if label != deduplicated[-1]:
            deduplicated.append(label)

    return tuple(deduplicated)


def iter_sequence_windows(
    cluster_labels: np.ndarray,
    trajectories: Iterable[TrajectorySlice],
    *,
    sequence_length: int,
) -> Iterator[SequenceWindow]:
    """
    Yield fixed-length CSDPC sequence windows trajectory by trajectory.

    The operation order is deliberately:

        original trajectory
            -> fixed-length original-position window
            -> consecutive-label deduplication

    A window is never allowed to cross a trajectory boundary.
    """

    cluster_labels = np.asarray(cluster_labels)

    if cluster_labels.ndim != 1:
        raise ValueError(
            "cluster_labels must be a 1D array"
        )

    if sequence_length <= 0:
        raise ValueError(
            "sequence_length must be positive"
        )

    num_transitions = len(cluster_labels)

    for trajectory_id, trajectory in enumerate(trajectories):
        start = int(trajectory.start)
        end = int(trajectory.end)

        if start < 0:
            raise ValueError(
                "trajectory start cannot be negative"
            )

        if end < start:
            raise ValueError(
                "trajectory end cannot precede start"
            )

        if end > num_transitions:
            raise ValueError(
                "trajectory extends beyond cluster_labels"
            )

        if trajectory.length < sequence_length:
            continue

        last_start = end - sequence_length

        for global_start in range(start, last_start + 1):
            global_end = global_start + sequence_length

            transition_indices = tuple(
                range(global_start, global_end)
            )

            raw_cluster_labels = tuple(
                int(label)
                for label in cluster_labels[
                    global_start:global_end
                ]
            )

            pattern = deduplicate_consecutive(
                raw_cluster_labels
            )

            yield SequenceWindow(
                trajectory_id=trajectory_id,
                global_start=global_start,
                global_end=global_end,
                transition_indices=transition_indices,
                raw_cluster_labels=raw_cluster_labels,
                pattern=pattern,
            )


def count_pattern_frequencies(
    windows: Iterable[SequenceWindow],
) -> CounterType[Pattern]:
    """
    Count CSDPC pattern occurrences.

    Frequency O(p) is the number of sequence windows whose
    deduplicated cluster-label pattern equals p.
    """

    return Counter(
        window.pattern
        for window in windows
    )