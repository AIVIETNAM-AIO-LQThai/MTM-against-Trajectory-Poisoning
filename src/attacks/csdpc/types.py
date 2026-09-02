from dataclasses import dataclass
from typing import Tuple

import numpy as np

Pattern = Tuple[int, ...]

@dataclass(frozen=True)
class SequenceWindow:
    trajectory_id: int
    start: int
    end: int
    transition_indices: np.ndarray
    cluster_labels: Tuple[int, ...]
    pattern: Pattern