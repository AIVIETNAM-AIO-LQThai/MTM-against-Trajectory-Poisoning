from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "attack_qualification"
    / "a2_dt_attack_qualification.json"
)
OUTPUT = (
    ROOT
    / "experiments"
    / "attack_qualification"
    / "a2_dt"
    / "preflight.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with Path(path).open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_config():
    config = json.loads(
        CONFIG.read_text(
            encoding="utf-8"
        )
    )

    if (
        config.get("experiment")
        != "A2_VANILLA_DT_ATTACK_QUALIFICATION"
    ):
        raise ValueError(
            "unexpected A2 config"
        )

    if (
        config.get("status")
        != "PREDECLARED"
    ):
        raise ValueError(
            "A2 status changed"
        )

    if (
        config["attack_seeds"]
        != [10, 11, 12]
    ):
        raise ValueError(
            "attack seeds changed"
        )

    if (
        config["model_seeds"]
        != [0, 1, 2]
    ):
        raise ValueError(
            "model seeds changed"
        )

    return config


def _attribute_dict(obj):
    return {
        str(key): np.asarray(value)
        for key, value
        in obj.attrs.items()
    }


def _assert_attributes_equal(
    clean_obj,
    poison_obj,
    *,
    object_path,
):
    clean_attrs = _attribute_dict(
        clean_obj
    )

    poison_attrs = _attribute_dict(
        poison_obj
    )

    if (
        set(clean_attrs)
        != set(poison_attrs)
    ):
        raise RuntimeError(
            "HDF5 attribute keys changed "
            f"at {object_path}"
        )

    for key in clean_attrs:
        clean_value = clean_attrs[key]
        poison_value = poison_attrs[key]

        if (
            clean_value.shape
            != poison_value.shape
        ):
            raise RuntimeError(
                "HDF5 attribute shape changed "
                f"at {object_path}:{key}"
            )

        if (
            clean_value.dtype
            != poison_value.dtype
        ):
            # String attributes can be represented with
            # equivalent object/string dtypes after HDF5 reads.
            # Fall through to value comparison for those.
            clean_is_string = (
                clean_value.dtype.kind
                in {"O", "S", "U"}
            )
            poison_is_string = (
                poison_value.dtype.kind
                in {"O", "S", "U"}
            )

            if not (
                clean_is_string
                and poison_is_string
            ):
                raise RuntimeError(
                    "HDF5 attribute dtype changed "
                    f"at {object_path}:{key}: "
                    f"{clean_value.dtype} vs "
                    f"{poison_value.dtype}"
                )

        if not np.array_equal(
            clean_value,
            poison_value,
        ):
            raise RuntimeError(
                "HDF5 attribute value changed "
                f"at {object_path}:{key}"
            )


def _object_kind(obj):
    if isinstance(
        obj,
        h5py.Dataset,
    ):
        return "dataset"

    if isinstance(
        obj,
        h5py.Group,
    ):
        return "group"

    return type(obj).__name__


def _collect_object_paths(
    handle,
):
    objects = {
        "/": "group"
    }

    def visitor(name, obj):
        objects[
            "/" + name
        ] = _object_kind(
            obj
        )

    handle.visititems(
        visitor
    )

    return objects


def _assert_reward_only_hdf5_difference(
    clean_path: Path,
    poison_path: Path,
    *,
    reward_dataset_path: str = "/rewards",
):
    with h5py.File(
        clean_path,
        "r",
    ) as clean, h5py.File(
        poison_path,
        "r",
    ) as poison:
        clean_objects = (
            _collect_object_paths(
                clean
            )
        )

        poison_objects = (
            _collect_object_paths(
                poison
            )
        )

        if (
            clean_objects
            != poison_objects
        ):
            clean_only = sorted(
                set(clean_objects)
                - set(poison_objects)
            )

            poison_only = sorted(
                set(poison_objects)
                - set(clean_objects)
            )

            kind_mismatch = sorted(
                path
                for path
                in (
                    set(clean_objects)
                    & set(poison_objects)
                )
                if (
                    clean_objects[path]
                    != poison_objects[path]
                )
            )

            raise RuntimeError(
                "HDF5 hierarchy changed: "
                f"clean_only={clean_only}, "
                f"poison_only={poison_only}, "
                f"kind_mismatch={kind_mismatch}"
            )

        _assert_attributes_equal(
            clean,
            poison,
            object_path="/",
        )

        if (
            reward_dataset_path
            not in clean_objects
        ):
            raise RuntimeError(
                "reward dataset missing from "
                f"clean HDF5: {reward_dataset_path}"
            )

        if (
            clean_objects[
                reward_dataset_path
            ]
            != "dataset"
        ):
            raise RuntimeError(
                "reward path is not an HDF5 "
                "dataset"
            )

        reward_changed = False

        for object_path, kind in (
            clean_objects.items()
        ):
            if object_path == "/":
                continue

            hdf5_name = (
                object_path.lstrip("/")
            )

            clean_obj = clean[
                hdf5_name
            ]

            poison_obj = poison[
                hdf5_name
            ]

            _assert_attributes_equal(
                clean_obj,
                poison_obj,
                object_path=(
                    object_path
                ),
            )

            if kind == "group":
                continue

            if (
                clean_obj.shape
                != poison_obj.shape
            ):
                raise RuntimeError(
                    "dataset shape changed: "
                    f"{object_path}"
                )

            if (
                clean_obj.dtype
                != poison_obj.dtype
            ):
                raise RuntimeError(
                    "dataset dtype changed: "
                    f"{object_path}: "
                    f"{clean_obj.dtype} vs "
                    f"{poison_obj.dtype}"
                )

            clean_data = clean_obj[()]
            poison_data = (
                poison_obj[()]
            )

            if (
                object_path
                == reward_dataset_path
            ):
                reward_changed = (
                    not np.array_equal(
                        clean_data,
                        poison_data,
                    )
                )

                continue

            if not np.array_equal(
                clean_data,
                poison_data,
            ):
                raise RuntimeError(
                    "non-reward dataset differs: "
                    f"{object_path}"
                )

        if not reward_changed:
            raise RuntimeError(
                "reward dataset did not change"
            )


def main():
    config = load_config()

    clean_path = (
        ROOT
        / config[
            "clean_dataset"
        ]
    )

    if not clean_path.exists():
        raise FileNotFoundError(
            clean_path
        )

    actual_clean_sha = (
        sha256_file(
            clean_path
        )
    )

    if (
        actual_clean_sha
        != config[
            "clean_dataset_sha256"
        ]
    ):
        raise RuntimeError(
            "clean dataset SHA mismatch"
        )

    metadata_root = (
        ROOT
        / config[
            "paths"
        ][
            "a1_metadata_root"
        ]
    )

    poison_root = (
        ROOT
        / config[
            "paths"
        ][
            "a1_poison_root"
        ]
    )

    rows = []

    for attack_seed in (
        config[
            "attack_seeds"
        ]
    ):
        metadata_path = (
            metadata_root
            / (
                f"attack_seed_"
                f"{attack_seed}.json"
            )
        )

        poison_path = (
            poison_root
            / (
                f"attack_seed_"
                f"{attack_seed}.hdf5"
            )
        )

        if not metadata_path.exists():
            raise FileNotFoundError(
                metadata_path
            )

        if not poison_path.exists():
            raise FileNotFoundError(
                poison_path
            )

        metadata = json.loads(
            metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            int(
                metadata[
                    "attack_seed"
                ]
            )
            != attack_seed
        ):
            raise RuntimeError(
                "metadata attack seed "
                "mismatch"
            )

        poison_sha = sha256_file(
            poison_path
        )

        if (
            poison_sha
            != metadata[
                "poisoned_dataset_sha256"
            ]
        ):
            raise RuntimeError(
                "poison SHA mismatch for "
                f"seed {attack_seed}"
            )

        integrity = metadata[
            "integrity"
        ]

        required_true = [
            "non_reward_arrays_identical",
            "trailing_fragment_unchanged",
            "selected_within_candidate_pool",
            "unselected_rewards_identical",
        ]

        for key in required_true:
            if (
                integrity.get(key)
                is not True
            ):
                raise RuntimeError(
                    "A1 integrity failed: "
                    f"seed={attack_seed}, "
                    f"{key}"
                )

        if (
            float(
                integrity[
                    "budget_utilization"
                ]
            )
            < 0.98
        ):
            raise RuntimeError(
                "A1 budget utilization "
                "below frozen minimum"
            )

        if (
            float(
                integrity[
                    "max_selected_target_return_error"
                ]
            )
            > 1e-2
        ):
            raise RuntimeError(
                "A1 target return error "
                "too large"
            )

        _assert_reward_only_hdf5_difference(
            clean_path,
            poison_path,
            reward_dataset_path=(
                "/rewards"
            ),
        )

        rows.append(
            {
                "attack_seed": (
                    attack_seed
                ),
                "poisoned_dataset": str(
                    poison_path
                ),
                "poisoned_dataset_sha256": (
                    poison_sha
                ),
                "selected_trajectory_count": int(
                    integrity[
                        "selected_trajectory_count"
                    ]
                ),
                "actual_transition_budget": int(
                    integrity[
                        "actual_transition_budget"
                    ]
                ),
                "requested_transition_budget": int(
                    integrity[
                        "requested_transition_budget"
                    ]
                ),
                "budget_utilization": float(
                    integrity[
                        "budget_utilization"
                    ]
                ),
                "max_selected_target_return_error": float(
                    integrity[
                        "max_selected_target_return_error"
                    ]
                ),
            }
        )

    output = {
        "experiment": (
            config["experiment"]
        ),
        "status": (
            "PREFLIGHT_PASS"
        ),
        "clean_dataset_sha256": (
            actual_clean_sha
        ),
        "attack_rows": rows,
        "matrix": {
            "clean_runs": 3,
            "poisoned_runs": 9,
            "total_training_runs": 12,
            "model_seeds": (
                config[
                    "model_seeds"
                ]
            ),
            "attack_seeds": (
                config[
                    "attack_seeds"
                ]
            ),
        },
    }

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT.write_text(
        json.dumps(
            output,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "A2 PREFLIGHT PASS"
    )

    print(
        json.dumps(
            output["matrix"],
            indent=2,
        )
    )

    for row in rows:
        print(
            f"attack_seed="
            f"{row['attack_seed']} "
            f"budget="
            f"{row['actual_transition_budget']}/"
            f"{row['requested_transition_budget']} "
            f"util="
            f"{row['budget_utilization']:.6f} "
            f"target_err="
            f"{row['max_selected_target_return_error']:.3e}"
        )

    print(
        "output ->",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
