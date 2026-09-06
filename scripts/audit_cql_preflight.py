from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from src.baselines.cql.runner import (
    build_components,
)


CLEAN_DATASET = Path(
    "data/raw/walker2d-medium-v2/"
    "walker2d_medium-v2.hdf5"
)


def main():
    components = build_components(
        dataset_path=CLEAN_DATASET,
        seed=0,
        use_gpu=True,
        gpu_id=0,
    )

    replay_buffer = components[
        "replay_buffer"
    ]

    training_view = components[
        "training_view"
    ]

    expected_n = len(
        training_view[
            "observations"
        ]
    )

    assert (
        replay_buffer._size
        == expected_n
    )

    batch = replay_buffer.random_batch(
        256
    )

    print(
        "observations:",
        batch[
            "observations"
        ].shape,
    )

    print(
        "actions:",
        batch[
            "actions"
        ].shape,
    )

    print(
        "next observations:",
        batch[
            "next_observations"
        ].shape,
    )

    assert batch[
        "observations"
    ].shape == (256, 17)

    assert batch[
        "actions"
    ].shape == (256, 6)

    assert batch[
        "next_observations"
    ].shape == (256, 17)

    assert (
        "raw_indices"
        not in batch
    )

    from rlkit.torch.core import (
        np_to_pytorch_batch,
    )

    torch_batch = np_to_pytorch_batch(
        batch
    )

    components[
        "trainer"
    ].train_from_torch(
        torch_batch
    )

    diagnostics = components[
        "trainer"
    ].get_diagnostics()

    for key, value in diagnostics.items():
        array = np.asarray(
            value
        )

        if np.issubdtype(
            array.dtype,
            np.number,
        ):
            if not np.all(
                np.isfinite(array)
            ):
                raise RuntimeError(
                    f"Non-finite diagnostic: {key}"
                )

    policy = components[
        "eval_policy"
    ]

    env = components[
        "env"
    ]

    obs = env.reset()

    if isinstance(obs, tuple):
        obs = obs[0]

    action, _ = policy.get_action(
        obs
    )

    result = env.step(
        action
    )

    assert result is not None

    assert torch.cuda.is_available()

    print(
        "device:",
        components[
            "metadata"
        ][
            "device"
        ],
    )

    print(
        "replay N:",
        replay_buffer._size,
    )

    print(
        "one trainer step: PASS"
    )

    print(
        "one environment step: PASS"
    )

    print(
        "finite diagnostics: PASS"
    )

    print()
    print(
        "CQL PREFLIGHT: PASS"
    )

    env.close()


if __name__ == "__main__":
    main()