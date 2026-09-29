#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: run_state_matched_action_eval.sh <repo-root-wsl>"
  exit 2
fi

REPO="$1"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dt-reference

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/home/quoc_thai/.mujoco/mujoco200/bin"

cd "$REPO"

ROOT="experiments/attack_qualification/state_matched_action_substitution"

for attack_seed in 30 31 32; do
  for model_seed in 0 1 2; do
    run_root="${ROOT}/poison/attack_seed_${attack_seed}/model_seed_${model_seed}"
    checkpoint="${run_root}/train/checkpoints/checkpoint_step_100000.pt"
    eval_dir="${run_root}/eval"
    summary="${eval_dir}/summary.json"

    echo
    echo "======================================================================"
    echo "EVAL attack=${attack_seed} model=${model_seed}"
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
      --condition state_matched_low_return_action_substitution \
      --rho 0.05 \
      --attack-seed "${attack_seed}" \
      --target-return 5000 \
      --num-episodes 100 \
      --eval-seed-base 30000 \
      --output-dir "${eval_dir}"
  done
done

echo
echo "Evaluation matrix complete."
