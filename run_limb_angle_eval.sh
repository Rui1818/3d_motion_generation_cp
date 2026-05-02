#!/bin/bash
#SBATCH --job-name=limb_angle_eval
#SBATCH --output=logs/limb_angle_eval_%A_%a.out   # stdout  (%A = job id, %a = array index)
#SBATCH --error=logs/limb_angle_eval_%A_%a.err    # stderr
#SBATCH --array=0-9%1                             # one task per model; %1 = run sequentially
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# ── Model list ────────────────────────────────────────────────────────────────
# Add one entry per model save_dir (must contain fold_0/, fold_1/, etc.)
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
CHECKPOINT="best"   # "best", "latest", or "both"

# ── Pick this task's save_dir via array index ─────────────────────────────────
SAVE_DIR="${SAVE_DIRS[$SLURM_ARRAY_TASK_ID]}"

if [ -z "$SAVE_DIR" ]; then
    echo "No SAVE_DIR for array index $SLURM_ARRAY_TASK_ID — exiting."
    exit 1
fi

echo "======================================================================"
echo "SLURM job:   $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node:        $SLURMD_NODENAME"
echo "Save dir:    $SAVE_DIR"
echo "Dataset:     $DATASET_PATH"
echo "Checkpoint:  $CHECKPOINT"
echo "Start time:  $(date)"
echo "======================================================================"

mkdir -p logs

python limb_angle_crossval_eval.py \
    --save_dir "$SAVE_DIR" \
    --dataset_path "$DATASET_PATH" \
    --num_folds "$NUM_FOLDS" \
    --seed "$SEED" \
    --checkpoint "$CHECKPOINT"

STATUS=$?
echo "======================================================================"
echo "Finished: $SAVE_DIR  (exit $STATUS)  $(date)"
echo "======================================================================"
exit $STATUS
