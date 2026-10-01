from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from src.methods.rdt_source_dt.data import (
    SourceSequenceDataset,
    load_source_trajectories,
)
from src.methods.rdt_source_dt.model import (
    make_source_compatible_dt,
)


ROOT = Path(
    __file__
).resolve().parents[
    1
]

CONFIG = (
    ROOT
    / "configs"
    / "attack_qualification"
    / "rdt_source_dt_victim.json"
)

DATASET = (
    ROOT
    / "data"
    / "derived"
    / "rdt_source_random_state_corruption"
    / "walker2d-medium-v2"
    / "clean_ratio_0p02.hdf5"
)

OUTPUT = (
    ROOT
    / "experiments"
    / "attack_qualification"
    / "rdt_source_dt_victim"
    / "clean_preflight.json"
)

EXPECTED_SHA = (
    "a2b0eedb6083b2d28f4ea0b5ce220dd"
    "6763df8c8152ce8209e98292a359ad401"
)


def sha256_file(
    path: Path,
) -> str:
    digest = (
        hashlib.sha256()
    )

    with path.open(
        "rb"
    ) as handle:
        for chunk in iter(
            lambda: handle.read(
                1024
                * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:
    cfg = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    if (
        cfg[
            "status"
        ]
        != "predeclared"
    ):
        raise RuntimeError(
            "victim protocol is not predeclared"
        )

    if (
        cfg[
            "source"
        ][
            "repository_commit"
        ]
        != "865fb60632153ed7d8a49e7941b675f026480006"
    ):
        raise RuntimeError(
            "source commit changed"
        )

    if (
        sha256_file(
            DATASET
        )
        != EXPECTED_SHA
    ):
        raise RuntimeError(
            "clean dataset SHA mismatch"
        )

    trajectories = (
        load_source_trajectories(
            DATASET,
            expected_num_trajectories=23,
            expected_num_transitions=20_147,
            expected_trailing_transitions=0,
        )
    )

    dataset = (
        SourceSequenceDataset(
            trajectories,
            seq_len=20,
            episode_len=1000,
            reward_scale=0.001,
        )
    )

    model = (
        make_source_compatible_dt()
    )

    if (
        len(
            model.blocks
        )
        != 3
        or model.embedding_dim
        != 128
        or model.seq_len
        != 20
        or model.timestep_embedding.num_embeddings
        != 1020
    ):
        raise RuntimeError(
            "source-compatible model contract mismatch"
        )

    if hasattr(
        model,
        "embedding_dropout",
    ):
        raise RuntimeError(
            "embedding dropout must be disabled"
        )

    if (
        model.predict_dropout.p
        != 0.1
    ):
        raise RuntimeError(
            "prediction dropout changed"
        )

    for block in model.blocks:
        if (
            block.attention.dropout
            != 0.0
            or block.attention_residual_dropout.p
            != 0.1
            or block.mlp[
                -1
            ].p
            != 0.1
        ):
            raise RuntimeError(
                "source dropout contract changed"
            )

    state, action, returns, time_steps, mask = (
        dataset.prepare_sample(
            0,
            trajectories[
                0
            ].length
            - 1,
        )
    )

    if (
        mask[
            0
        ]
        != 1.0
        or not bool(
            (
                mask[
                    1:
                ]
                == 0.0
            ).all()
        )
    ):
        raise RuntimeError(
            "right-padding mask mismatch"
        )

    if not bool(
        (
            state[
                1:
            ]
            == 0.0
        ).all()
    ):
        raise RuntimeError(
            "state right padding changed"
        )

    if not bool(
        (
            action[
                1:
            ]
            == 0.0
        ).all()
    ):
        raise RuntimeError(
            "action right padding changed"
        )

    if not bool(
        (
            returns[
                1:
            ]
            == 0.0
        ).all()
    ):
        raise RuntimeError(
            "RTG right padding changed"
        )

    if (
        time_steps.shape
        != (
            20,
        )
    ):
        raise RuntimeError(
            "timestep shape mismatch"
        )

    result = {
        "status": (
            "PREFLIGHT_PASS"
        ),
        "source_repository_commit": (
            cfg[
                "source"
            ][
                "repository_commit"
            ]
        ),
        "clean_dataset_sha256": (
            EXPECTED_SHA
        ),
        "clean_trajectories": (
            len(
                trajectories
            )
        ),
        "clean_transitions": (
            sum(
                trajectory.length
                for trajectory
                in trajectories
            )
        ),
        "source_model": {
            "embedding_dim": (
                model.embedding_dim
            ),
            "num_layers": (
                len(
                    model.blocks
                )
            ),
            "num_heads": (
                model.blocks[
                    0
                ].attention.num_heads
            ),
            "seq_len": (
                model.seq_len
            ),
            "timestep_embeddings": (
                model.timestep_embedding.num_embeddings
            ),
            "attention_dropout": (
                model.blocks[
                    0
                ].attention.dropout
            ),
            "residual_dropout": (
                model.blocks[
                    0
                ].attention_residual_dropout.p
            ),
            "embedding_dropout": (
                None
            ),
            "prediction_dropout": (
                model.predict_dropout.p
            ),
            "state_normalization": (
                False
            ),
        },
        "batching": {
            "padding_side": (
                "right"
            ),
            "state_padding": (
                0.0
            ),
            "action_padding": (
                0.0
            ),
            "return_padding": (
                0.0
            ),
            "reward_scale": (
                dataset.reward_scale
            ),
        },
        "clean_viability_seeds": [
            0,
            1,
            2,
        ],
        "corrupted_training_allowed": (
            False
        ),
        "torch_version": (
            torch.__version__
        ),
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "=" * 92
    )

    print(
        "SOURCE-COMPATIBLE RDT DT CLEAN-VICTIM PREFLIGHT"
    )

    print(
        "=" * 92
    )

    print(
        "dataset SHA: PASS"
    )

    print(
        "trajectory contract: "
        "23 / 20147 / trailing 0"
    )

    print(
        "source model contract: PASS"
    )

    print(
        "source batching contract: PASS"
    )

    print(
        "clean viability matrix: seeds 0,1,2 only"
    )

    print(
        "corrupted training: BLOCKED until clean viability passes"
    )

    print(
        "PREFLIGHT: PASS"
    )

    print(
        "output ->",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
