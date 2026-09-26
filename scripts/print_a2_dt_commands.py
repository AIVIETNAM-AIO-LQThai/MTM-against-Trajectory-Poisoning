from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CLEAN = "data/raw/walker2d-medium-v2/walker2d_medium-v2.hdf5"
OUT = "experiments/attack_qualification/a2_dt"


def train_command(dataset, condition, rho, attack_seed, seed, output_dir):
    return (
        "python -m scripts.train_dt_stress "
        f"--dataset {dataset} "
        f"--condition {condition} "
        f"--rho {rho} "
        f"--attack-seed {attack_seed} "
        f"--seed {seed} "
        "--num-updates 100000 "
        f"--output-dir {output_dir}"
    )


def eval_command(checkpoint, condition, rho, attack_seed, output_dir):
    return (
        "python -m scripts.evaluate_dt_stress "
        f"--checkpoint {checkpoint} "
        f"--condition {condition} "
        f"--rho {rho} "
        f"--attack-seed {attack_seed} "
        "--target-return 5000 "
        "--num-episodes 100 "
        "--eval-seed-base 30000 "
        f"--output-dir {output_dir}"
    )


def main():
    print("# A2 CLEAN TRAINING")
    for seed in (0, 1, 2):
        base = f"{OUT}/clean/model_seed_{seed}"
        print(train_command(CLEAN, "a2_clean", 0.0, -1, seed, base + "/train"))

    print()
    print("# A2 CLEAN EVALUATION")
    for seed in (0, 1, 2):
        base = f"{OUT}/clean/model_seed_{seed}"
        ckpt = base + "/train/checkpoints/checkpoint_step_100000.pt"
        print(eval_command(ckpt, "a2_clean", 0.0, -1, base + "/eval"))

    print()
    print("# A2 POISONED TRAINING")
    for attack_seed in (10, 11, 12):
        dataset = (
            "data/poisoned/rtg_inflation/walker2d-medium-v2/"
            f"attack_seed_{attack_seed}.hdf5"
        )
        for seed in (0, 1, 2):
            base = f"{OUT}/poison/attack_seed_{attack_seed}/model_seed_{seed}"
            print(
                train_command(
                    dataset,
                    "rtg_inflation",
                    0.05,
                    attack_seed,
                    seed,
                    base + "/train",
                )
            )

    print()
    print("# A2 POISONED EVALUATION")
    for attack_seed in (10, 11, 12):
        for seed in (0, 1, 2):
            base = f"{OUT}/poison/attack_seed_{attack_seed}/model_seed_{seed}"
            ckpt = base + "/train/checkpoints/checkpoint_step_100000.pt"
            print(
                eval_command(
                    ckpt,
                    "rtg_inflation",
                    0.05,
                    attack_seed,
                    base + "/eval",
                )
            )


if __name__ == "__main__":
    main()
