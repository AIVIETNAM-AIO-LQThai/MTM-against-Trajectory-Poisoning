from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Any, Mapping

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

@dataclass(frozen=True)
class SelectedWindow:
    trajectory_id: int
    global_start: int
    global_end: int
    source_pattern: Pattern

    @property
    def transition_indices(self) -> Tuple[int, ...]:
        return tuple(
            range(
                self.global_start,
                self.global_end,
            )
        )

    @property
    def length(self) -> int:
        return (
            self.global_end
            - self.global_start
        )


@dataclass(frozen=True)
class SelectionResult:
    selected_windows: Tuple[SelectedWindow, ...]
    requested_transition_budget: int
    actual_transition_budget: int
    skipped_overlap_windows: int

    @property
    def num_selected_windows(self) -> int:
        return len(self.selected_windows)

@dataclass(frozen=True)
class PerturbedWindow:
    trajectory_id: int
    global_start: int
    global_end: int

    source_pattern: Pattern
    target_pattern: Pattern

    source_frequency: int
    target_frequency: int

    candidate_index: int
    total_linf_perturbation: float

    observations: np.ndarray
    actions: np.ndarray

    state_deltas: np.ndarray
    action_deltas: np.ndarray

    @property
    def transition_indices(self) -> Tuple[int, ...]:
        return tuple(
            range(
                self.global_start,
                self.global_end,
            )
        )

@dataclass(frozen=True)
class PreparedCSDPC:
    attack_seed: int

    num_transitions: int
    sequence_length: int
    eta: float
    num_candidates: int

    clustering_model: Any
    clustering: ClusteringResult

    windows: Tuple[SequenceWindow, ...]
    pattern_frequencies: Mapping[Pattern, int]

@dataclass(frozen=True)
class CSDPCAttackResult:
    poisoned_dataset: Mapping[str, np.ndarray]

    requested_rho: float
    actual_rho: float

    requested_transition_budget: int
    actual_transition_budget: int

    selected_windows: Tuple[SelectedWindow, ...]
    perturbed_windows: Tuple[PerturbedWindow, ...]

    modified_transition_indices: Tuple[int, ...]