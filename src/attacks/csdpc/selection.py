from __future__ import annotations

from typing import Iterable, Mapping

from .types import (
    Pattern,
    SelectedWindow,
    SelectionResult,
    SequenceWindow,
)


def compute_transition_budget(
    *,
    num_transitions: int,
    rho: float,
) -> int:
    """
    Compute the requested transition-level poison budget.

    Frozen reproduction convention:

        budget = floor(rho * N)

    where N is the number of transitions in the original dataset.
    """

    if num_transitions < 0:
        raise ValueError(
            "num_transitions cannot be negative"
        )

    if not 0.0 <= rho <= 1.0:
        raise ValueError(
            "rho must lie in [0, 1]"
        )

    return int(rho * num_transitions)


def select_rare_nonoverlapping_windows(
    windows: Iterable[SequenceWindow],
    pattern_frequencies: Mapping[Pattern, int],
    *,
    transition_budget: int,
) -> SelectionResult:
    """
    Select complete, non-overlapping windows in deterministic
    rare-pattern-first order.

    Ranking:
        1. lower pattern occurrence count
        2. lexicographic pattern tuple
        3. trajectory_id
        4. global_start

    Complete windows only are selected, and the transition budget
    is never exceeded.
    """

    if transition_budget < 0:
        raise ValueError(
            "transition_budget cannot be negative"
        )

    if transition_budget == 0:
        return SelectionResult(
            selected_windows=tuple(),
            requested_transition_budget=0,
            actual_transition_budget=0,
            skipped_overlap_windows=0,
        )

    ranked_patterns = sorted(
        pattern_frequencies,
        key=lambda pattern: (
            int(pattern_frequencies[pattern]),
            pattern,
        ),
    )

    pattern_rank = {
        pattern: rank
        for rank, pattern
        in enumerate(ranked_patterns)
    }

    # Store only the minimal information needed for selection.
    # Each inner list is already encountered in trajectory/start order
    # because iter_sequence_windows yields in that order.
    occurrences_by_rank = [
        []
        for _ in ranked_patterns
    ]

    for window in windows:
        if window.pattern not in pattern_rank:
            raise ValueError(
                "window pattern missing from pattern_frequencies"
            )

        rank = pattern_rank[window.pattern]

        occurrences_by_rank[rank].append(
            (
                int(window.trajectory_id),
                int(window.global_start),
                int(window.global_end),
            )
        )

    selected = []
    used_transitions = set()

    actual_budget = 0
    skipped_overlap_windows = 0

    for rank, pattern in enumerate(
        ranked_patterns
    ):
        occurrences = occurrences_by_rank[
            rank
        ]

        for (
            trajectory_id,
            global_start,
            global_end,
        ) in occurrences:
            window_length = (
                global_end
                - global_start
            )

            if window_length <= 0:
                raise ValueError(
                    "window length must be positive"
                )

            remaining_budget = (
                transition_budget
                - actual_budget
            )

            if window_length > remaining_budget:
                continue

            indices = tuple(
                range(
                    global_start,
                    global_end,
                )
            )

            if any(
                index in used_transitions
                for index in indices
            ):
                skipped_overlap_windows += 1
                continue

            selected_window = SelectedWindow(
                trajectory_id=trajectory_id,
                global_start=global_start,
                global_end=global_end,
                source_pattern=pattern,
            )

            selected.append(
                selected_window
            )

            used_transitions.update(
                indices
            )

            actual_budget += (
                window_length
            )

            if actual_budget == transition_budget:
                return SelectionResult(
                    selected_windows=tuple(
                        selected
                    ),
                    requested_transition_budget=(
                        transition_budget
                    ),
                    actual_transition_budget=(
                        actual_budget
                    ),
                    skipped_overlap_windows=(
                        skipped_overlap_windows
                    ),
                )

    return SelectionResult(
        selected_windows=tuple(selected),
        requested_transition_budget=(
            transition_budget
        ),
        actual_transition_budget=(
            actual_budget
        ),
        skipped_overlap_windows=(
            skipped_overlap_windows
        ),
    )