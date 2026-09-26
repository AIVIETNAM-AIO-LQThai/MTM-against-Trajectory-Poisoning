from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.mtm_dataset import (
    ReferenceMTMDataset,
)

from src.data.mtm_split import (
    MTMTrajectorySplit,
)


@dataclass(frozen=True)
class MTMWindowRange:
    """
    Half-open range of ReferenceMTMDataset window indices.

        [start, end)

    Since the reference trajectory split preserves trajectory
    order, train windows form one contiguous prefix and
    validation windows form one contiguous suffix.
    """

    start: int
    end: int

    @property
    def count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class MTMWindowRanges:
    train: MTMWindowRange
    validation: MTMWindowRange


@dataclass(frozen=True)
class MTMBatch:
    states: np.ndarray
    actions: np.ndarray
    returns: np.ndarray

    dataset_indices: np.ndarray
    trajectory_ids: np.ndarray
    global_starts: np.ndarray

class MTMShuffledWindowSampler:
    """
    Source-aligned shuffled-epoch sampler for MTM training.

    Each training window is visited once per shuffled epoch
    before a new permutation is generated.

    This mirrors the sampling semantics of the official
    single-process MTM training pipeline:

        RandomSampler(dataset)
        -> DataLoader(...)
        -> new iterator after StopIteration

    The sampler operates only inside one MTMWindowRange.
    """

    def __init__(
        self,
        window_range: MTMWindowRange,
        *,
        seed: int,
    ) -> None:
        if window_range.count <= 0:
            raise ValueError(
                "window range must be non-empty"
            )

        self.window_range = (
            window_range
        )

        self.rng = (
            np.random.RandomState(
                seed
            )
        )

        self.epoch = 0
        self.position = 0

        self._permutation = (
            self._new_permutation()
        )

    def _new_permutation(
        self,
    ) -> np.ndarray:
        permutation = (
            self.rng.permutation(
                self.window_range.count
            )
            .astype(
                np.int64,
                copy=False,
            )
        )

        permutation += (
            self.window_range.start
        )

        return permutation

    def next_indices(
        self,
        batch_size: int,
    ) -> np.ndarray:
        """
        Return the next batch from the current shuffled epoch.

        Like a PyTorch DataLoader with drop_last=False, the
        final batch of an epoch may contain fewer than
        batch_size samples.
        """

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        end = min(
            self.position
            + batch_size,
            self.window_range.count,
        )

        indices = (
            self._permutation[
                self.position:end
            ]
            .copy()
        )

        self.position = end

        if (
            self.position
            == self.window_range.count
        ):
            self.epoch += 1
            self.position = 0
            self._permutation = (
                self._new_permutation()
            )

        return indices

    def state_dict(
        self,
    ) -> dict:
        return {
            "epoch": self.epoch,
            "position": (
                self.position
            ),
            "permutation": (
                self._permutation.copy()
            ),
            "rng_state": (
                self.rng.get_state()
            ),
        }

    def load_state_dict(
        self,
        state: dict,
    ) -> None:
        self.epoch = int(
            state[
                "epoch"
            ]
        )

        self.position = int(
            state[
                "position"
            ]
        )

        permutation = np.asarray(
            state[
                "permutation"
            ],
            dtype=np.int64,
        )

        if permutation.shape != (
            self.window_range.count,
        ):
            raise ValueError(
                "sampler permutation size "
                "does not match window range"
            )

        self._permutation = (
            permutation.copy()
        )

        self.rng.set_state(
            state[
                "rng_state"
            ]
        )

def get_mtm_batch(
    dataset: ReferenceMTMDataset,
    indices: np.ndarray
) -> MTMBatch:
    """
    Materialize one MTM batch from explicit dataset indices.
    """
    indices = np.asarray(indices, dtype=np.int64)
    if (indices.ndim != 1 or len(indices) == 0):
        raise ValueError(
            "indices must be a non-empty 1D array"
        )

    if (np.any(indices < 0) or np.any(indices >= len(dataset))):
        raise ValueError(
            "dataset index out of range"
        )

    samples = [dataset[int(index)] for index in indices]
    states = np.stack(
        [sample.states for sample in samples], axis=0
    ).astype(np.float32, copy=False)
    actions = np.stack(
        [sample.actions for sample in samples], axis=0
    ).astype(np.float32, copy=False)
    returns = np.stack(
        [sample.returns for sample in samples], axis=0
    ).astype(np.float32, copy=False)

    trajectory_ids = np.asarray(
        [sample.trajectory_id for sample in samples], dtype=np.int64
    )
    global_starts = np.asarray(
        [sample.global_start for sample in samples], dtype=np.int64
    )

    return MTMBatch(
        states=states, actions=actions, returns=returns,
        dataset_indices=indices.copy(),
        trajectory_ids=trajectory_ids,
        global_starts=global_starts
    )

def _trajectory_window_count(
    trajectory_length: int,
    *,
    window_length: int,
) -> int:
    return max(
        0,
        trajectory_length
        - window_length
        + 1,
    )


def build_split_window_ranges(
    dataset: ReferenceMTMDataset,
    split: MTMTrajectorySplit,
) -> MTMWindowRanges:
    """
    Map the trajectory-level reference split to the lazy
    ReferenceMTMDataset window-index space.

    This relies on an important frozen property of the reference
    split:

        training trajectories come first,
        validation trajectories come second,
        no shuffling occurs.
    """

    num_trajectories = (
        dataset.num_completed_trajectories
    )

    all_ids = (
        split.train_ids
        + split.validation_ids
    )

    if all_ids != tuple(
        range(
            num_trajectories
        )
    ):
        raise ValueError(
            "split trajectory IDs must preserve "
            "the complete original trajectory order"
        )

    if len(
        split.train_ids
    ) == 0:
        raise ValueError(
            "training split is empty"
        )

    if len(
        split.validation_ids
    ) == 0:
        raise ValueError(
            "validation split is empty"
        )

    train_window_count = sum(
        _trajectory_window_count(
            dataset.trajectories[
                trajectory_id
            ].length,
            window_length=(
                dataset.trajectory_length
            ),
        )
        for trajectory_id
        in split.train_ids
    )

    validation_window_count = sum(
        _trajectory_window_count(
            dataset.trajectories[
                trajectory_id
            ].length,
            window_length=(
                dataset.trajectory_length
            ),
        )
        for trajectory_id
        in split.validation_ids
    )

    if (
        train_window_count
        + validation_window_count
        != len(dataset)
    ):
        raise RuntimeError(
            "split window counts do not "
            "cover the full MTM dataset"
        )

    train_range = MTMWindowRange(
        start=0,
        end=train_window_count,
    )

    validation_range = MTMWindowRange(
        start=train_window_count,
        end=len(dataset),
    )

    return MTMWindowRanges(
        train=train_range,
        validation=validation_range,
    )


def sample_mtm_batch(
    dataset: ReferenceMTMDataset,
    window_range: MTMWindowRange,
    *,
    batch_size: int,
    rng: np.random.RandomState,
) -> MTMBatch:
    """
    Uniformly sample fixed-length windows from one split.

    Sampling is over windows, not trajectories.
    """

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive"
        )

    if window_range.start < 0:
        raise ValueError(
            "window range start "
            "must be non-negative"
        )

    if (
        window_range.end
        > len(dataset)
    ):
        raise ValueError(
            "window range exceeds dataset"
        )

    if window_range.count <= 0:
        raise ValueError(
            "window range is empty"
        )

    indices = rng.randint(
        window_range.start,
        window_range.end,
        size=batch_size,
    )

    samples = [
        dataset[
            int(index)
        ]
        for index in indices
    ]

    states = np.stack(
        [
            sample.states
            for sample in samples
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    actions = np.stack(
        [
            sample.actions
            for sample in samples
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    returns = np.stack(
        [
            sample.returns
            for sample in samples
        ],
        axis=0,
    ).astype(
        np.float32,
        copy=False,
    )

    trajectory_ids = np.asarray(
        [
            sample.trajectory_id
            for sample in samples
        ],
        dtype=np.int64,
    )

    global_starts = np.asarray(
        [
            sample.global_start
            for sample in samples
        ],
        dtype=np.int64,
    )

    return MTMBatch(
        states=states,
        actions=actions,
        returns=returns,
        dataset_indices=(
            indices.astype(
                np.int64,
                copy=False,
            )
        ),
        trajectory_ids=(
            trajectory_ids
        ),
        global_starts=(
            global_starts
        ),
    )