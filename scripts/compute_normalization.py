from pathlib import Path

import h5py
import numpy as np

from src.data.normalization import (
    compute_state_statistics,
)
from src.data.trajectories import (
    find_completed_trajectories,
)


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)

OUTPUT_PATH = Path(
    "data/metadata/"
    "walker2d_medium_normalization.npz"
)


def main() -> None:
    with h5py.File(DATASET_PATH, "r") as f:
        observations = f["observations"][:]

        terminals = (
            f["terminals"][:]
            .astype(bool)
        )

        timeouts = (
            f["timeouts"][:]
            .astype(bool)
        )

    trajectories, trailing = (
        find_completed_trajectories(
            terminals,
            timeouts,
        )
    )

    state_mean, state_std = (
        compute_state_statistics(
            observations,
            trajectories,
        )
    )

    used_transitions = sum(
        trajectory.length
        for trajectory in trajectories
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        OUTPUT_PATH,
        state_mean=state_mean,
        state_std=state_std,
        epsilon=np.array(1e-6),
        num_training_transitions=np.array(
            used_transitions
        ),
        num_trajectories=np.array(
            len(trajectories)
        ),
        trailing_transitions=np.array(
            trailing
        ),
    )

    print("Normalization statistics saved:")
    print(f"  output: {OUTPUT_PATH}")
    print(
        f"  state_mean shape: "
        f"{state_mean.shape}"
    )
    print(
        f"  state_std shape: "
        f"{state_std.shape}"
    )
    print(
        f"  training transitions: "
        f"{used_transitions}"
    )
    print(
        f"  trajectories: "
        f"{len(trajectories)}"
    )
    print(
        f"  excluded trailing transitions: "
        f"{trailing}"
    )

    print("\nstate_mean:")
    print(state_mean)

    print("\nstate_std:")
    print(state_std)


if __name__ == "__main__":
    main()