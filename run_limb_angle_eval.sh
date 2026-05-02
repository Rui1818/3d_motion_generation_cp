#!/bin/bash
#SBATCH --job-name=limb_angle_eval
#SBATCH --output=logs/limb_angle_eval_%j.out
#SBATCH --error=logs/limb_angle_eval_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# ── Model list ────────────────────────────────────────────────────────────────
SAVE_DIRS=(
    "final_training/window/config9"
    "final_training/window/config12"
    "final_training/transformer/config2"
    "final_training/transformer/config5"
    "final_training/transformer/config12"
    "final_training/dcttransformer/config1"
    "final_training/dcttransformer/config5"
    "final_training/dctmlp/config1"
    "final_training/dctmlp/config5"
    "final_training/weightmlp/config2"
)

# ── Common settings ───────────────────────────────────────────────────────────
DATASET_PATH="final_dataset"
NUM_FOLDS=5
SEED=10
CHECKPOINT="best"

# ── Run all models sequentially ───────────────────────────────────────────────
echo "SLURM job: $SLURM_JOB_ID  |  Node: $SLURMD_NODENAME  |  Start: $(date)"

for SAVE_DIR in "${SAVE_DIRS[@]}"; do
    echo "======================================================================"
    echo "Evaluating: $SAVE_DIR  ($(date))"
    echo "======================================================================"

    python limb_angle_crossval_eval.py \
        --save_dir "$SAVE_DIR" \
        --dataset_path "$DATASET_PATH" \
        --num_folds "$NUM_FOLDS" \
        --seed "$SEED" \
        --checkpoint "$CHECKPOINT"

    if [ $? -ne 0 ]; then
        echo "  !!! FAILED: $SAVE_DIR !!!"
    else
        echo "  Done: $SAVE_DIR"
    fi
done

echo "======================================================================"
echo "All evaluations finished  ($(date))"
echo "======================================================================"
