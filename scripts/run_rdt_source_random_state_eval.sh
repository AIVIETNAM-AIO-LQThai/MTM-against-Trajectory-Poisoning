#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: run_rdt_source_random_state_eval.sh <repo-root-wsl>"
  exit 2
fi

REPO="$1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dt-reference

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/quoc_thai/.mujoco/mujoco200/bin"

cd "$REPO"

ROOT="experiments/attack_qualification/rdt_source_random_state_corruption"
NORMALIZATION="data/metadata/rdt_source_random_state_corruption/walker2d-medium-v2/clean_ratio_0p02_normalization.npz"

for model_seed in 0 1 2; do
  run_root="${ROOT}/clean/model_seed_${model_seed}"
  checkpoint="${run_root}/train/checkpoints/checkpoint_step_100000.pt"
  eval_dir="${run_root}/eval"
  summary="${eval_dir}/summary.json"

  echo
  echo "======================================================================"
  echo "EVAL CLEAN model=${model_seed}"
  echo "======================================================================"

  if [[ -f "${summary}" ]]; then
    echo "Evaluation summary already exists -> SKIP"
    continue
  fi

  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing final checkpoint: ${checkpoint}"
    exit 1
  fi

  if [[ -d "${eval_dir}" ]]; then
    echo "Partial eval directory exists without summary -> REMOVE"
    rm -rf "${eval_dir}"
  fi

  python -m scripts.evaluate_dt_stress \
    --checkpoint "${checkpoint}" \
    --normalization "${NORMALIZATION}" \
    --condition rdt_source_clean_ratio_0p02 \
    --rho 0.0 \
    --attack-seed 1234 \
    --target-return 5000 \
    --num-episodes 100 \
    --eval-seed-base 30000 \
    --output-dir "${eval_dir}"
done

for corruption_seed in 2023 2024 2025; do
  for model_seed in 0 1 2; do
    run_root="${ROOT}/corrupted/corruption_seed_${corruption_seed}/model_seed_${model_seed}"
    checkpoint="${run_root}/train/checkpoints/checkpoint_step_100000.pt"
    eval_dir="${run_root}/eval"
    summary="${eval_dir}/summary.json"

    echo
    echo "======================================================================"
    echo "EVAL CORRUPTED corruption=${corruption_seed} model=${model_seed}"
    echo "======================================================================"

    if [[ -f "${summary}" ]]; then
      echo "Evaluation summary already exists -> SKIP"
      continue
    fi

    if [[ ! -f "${checkpoint}" ]]; then
      echo "Missing final checkpoint: ${checkpoint}"
      exit 1
    fi

    if [[ -d "${eval_dir}" ]]; then
      echo "Partial eval directory exists without summary -> REMOVE"
      rm -rf "${eval_dir}"
    fi

    python -m scripts.evaluate_dt_stress \
      --checkpoint "${checkpoint}" \
      --normalization "${NORMALIZATION}" \
      --condition rdt_source_random_state_corruption_ratio_0p02 \
      --rho 0.30 \
      --attack-seed "${corruption_seed}" \
      --target-return 5000 \
      --num-episodes 100 \
      --eval-seed-base 30000 \
      --output-dir "${eval_dir}"
  done
done

echo
echo "Evaluation matrix complete."
