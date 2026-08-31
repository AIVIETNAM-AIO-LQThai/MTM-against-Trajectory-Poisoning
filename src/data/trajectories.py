from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TrajectorySlice:
    start: int
    end: int # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start

def find_completed_trajectories(
    terminals: np.ndarray,
    timeouts: np.ndarray,
) -> tuple[list[TrajectorySlice], int]:
    """
    Find completed trajectories using DT/D4RL-compatible boundaries.

    A trajectory ends whenever either:
        terminal[t] == True
    or:
        timeout[t] == True

    Any final fragment without an explicit terminal/timeout boundary
    is excluded from training.

    Returns
    -------
    trajectories:
        Completed trajectory slices [start, end).

    trailing_transitions:
        Number of transitions after the final completed trajectory.
    """
    terminals = np.asarray(terminals, dtype=bool)
    timeouts = np.asarray(timeouts, dtype=bool)

    if terminals.ndim != 1 or timeouts.ndim != 1:
        raise ValueError(
            "terminals and timeouts must both be 1D arrays"
        )
    if len(terminals) != len(timeouts):
        raise ValueError(
            "termminals and timeouts must have equal length"
        )

    boundaries = terminals | timeouts
    trajectories: list[TrajectorySlice] = []
    start = 0

    for end_index in np.flatnonzero(boundaries):
        trajectories.append(
            TrajectorySlice(
                start=start,
                end=int(end_index) + 1,
            )
        )
        start = int(end_index) + 1

    trailing_transitions = len(terminals) - start
    return trajectories, trailing_transitions