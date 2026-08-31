from pathlib import Path
import h5py
import numpy as np

from src.data.trajectories import find_completed_trajectories

DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)

def test_walker2d_trajectory_boundaries():
    with h5py.File(DATASET_PATH, "r") as f:
        terminals = f["terminals"][:].astype(bool)
        timeouts = f["timeouts"][:].astype(bool)

    trajectories, trailing = find_completed_trajectories(
        terminals, timeouts
    )

    assert len(trajectories) == 1190
    assert trailing == 5

    used_transitions = sum(
        trajectory.length
        for trajectory in trajectories
    )

    assert used_transitions == 999_995

    # Every recorded trajectory must end at an explicit boundary.
    for trajectory in trajectories:
        last_index = trajectory.end - 1
        assert (
            terminals[last_index]
            or timeouts[last_index]
        )

    # No trajectory may overlap the next one
    for current, next_trajectory in zip(
        trajectories[:-1],
        trajectories[1:],
    ):
        assert current.end == next_trajectory.start

    # Known observed range from our dataset audit.
    lengths = np.array(
        [t.length for t in trajectories]
    )

    assert lengths.min() == 60
    assert lengths.max() == 1000