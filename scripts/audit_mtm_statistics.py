from pathlib import Path

import h5py
import numpy as np

from src.data.mtm_split import (
    reference_trajectory_split,
)

from src.data.mtm_statistics import (
    compute_reference_mtm_statistics,
)

from src.data.mtm_targets import (
    build_reference_mtm_returns,
)

from src.data.trajectories import (
    find_completed_trajectories,
)


DATASET_PATH = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)


def main() -> None:
    with h5py.File(
        DATASET_PATH,
        "r",
    ) as handle:
        observations = (
            handle["observations"][:]
        )

        actions = (
            handle["actions"][:]
        )

        rewards = (
            handle["rewards"][:]
        )

        terminals = (
            handle["terminals"][:]
            .astype(bool)
        )

        timeouts = (
            handle["timeouts"][:]
            .astype(bool)
        )

    (
        trajectories,
        trailing,
    ) = find_completed_trajectories(
        terminals,
        timeouts,
    )

    split = reference_trajectory_split(
        trajectories,
        train_fraction=0.95,
    )

    returns = build_reference_mtm_returns(
        rewards,
        trajectories,
        max_path_length=1000,
        discount=1.5,
    )

    stats = compute_reference_mtm_statistics(
        observations,
        actions,
        returns,
        split.train_trajectories,
        max_path_length=1000,
    )

    print("=" * 70)
    print("REFERENCE MTM STATISTICS AUDIT")
    print("=" * 70)

    print(
        f"completed trajectories: "
        f"{len(trajectories)}"
    )

    print(
        f"train trajectories: "
        f"{len(split.train_trajectories)}"
    )

    print(
        f"validation trajectories: "
        f"{len(split.validation_trajectories)}"
    )

    print(
        f"trailing transitions: "
        f"{trailing}"
    )

    assert len(trajectories) == 1190
    assert len(
        split.train_trajectories
    ) == 1130

    assert len(
        split.validation_trajectories
    ) == 60

    assert trailing == 5

    print()

    for modality, modality_stats in (
        stats.items()
    ):
        print(
            f"[{modality}]"
        )

        print(
            "  shape mean:",
            modality_stats.mean.shape,
        )

        print(
            "  mean range:",
            float(
                np.min(
                    modality_stats.mean
                )
            ),
            float(
                np.max(
                    modality_stats.mean
                )
            ),
        )

        print(
            "  std range:",
            float(
                np.min(
                    modality_stats.std
                )
            ),
            float(
                np.max(
                    modality_stats.std
                )
            ),
        )

        assert np.isfinite(
            modality_stats.mean
        ).all()

        assert np.isfinite(
            modality_stats.std
        ).all()

        assert np.isfinite(
            modality_stats.min
        ).all()

        assert np.isfinite(
            modality_stats.max
        ).all()

    print()
    print("[PASS] Reference MTM statistics audit")


if __name__ == "__main__":
    main()