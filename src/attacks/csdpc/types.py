from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


Pattern = Tuple[int, ...]


@dataclass(frozen=True)
class ClusteringResult:
    """
    Immutable summary of the fitted CSDPC clustering stage.

    The sklearn KMeans object itself is returned separately by the
    clustering function. This dataclass stores only the information
    required by the downstream attack/audit pipeline.
    """

    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    n_iter: int


@dataclass(frozen=True)
class SequenceWindow:
    """
    One original fixed-length trajectory window and its CSDPC pattern.

    global_start/global_end refer to indices in the original flat
    offline dataset. global_end is exclusive.

    raw_cluster_labels always contains exactly sequence_length labels.

    pattern is obtained by removing consecutive duplicate labels from
    raw_cluster_labels *after* the original window has been extracted.
    """

    trajectory_id: int
    global_start: int
    global_end: int
    transition_indices: Tuple[int, ...]
    raw_cluster_labels: Pattern
    pattern: Pattern