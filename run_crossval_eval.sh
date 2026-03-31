#!/bin/bash

# Cross-validation evaluation runner.
# Runs gait_crossval_eval.py for each model save_dir listed below.

DATASET_PATH="final_dataset"
NUM_FOLDS=5
SEED=10

# Optional: path to autoencoder checkpoint for FID computation.
# Leave empty to skip FID.
AUTOENCODER_PATH=""

# ── Helper ────────────────────────────────────────────────────────────────────

run_eval() {
    local save_dir=$1
    echo ""
    echo "======================================================================"
    echo "Evaluating: $save_dir"
    echo "======================================================================"

    if [ ! -d "$save_dir" ]; then
        echo "  SKIP: directory not found."
        return
    fi

    local cmd="python gait_crossval_eval.py \
        --save_dir \"$save_dir\" \
        --dataset_path \"$DATASET_PATH\" \
        --num_folds $NUM_FOLDS \
        --seed $SEED"

    if [ -n "$AUTOENCODER_PATH" ]; then
        cmd="$cmd --autoencoder_path \"$AUTOENCODER_PATH\""
    fi

    eval $cmd

    if [ $? -ne 0 ]; then
        echo "  !!! Evaluation FAILED for $save_dir !!!"
    else
        echo "  Evaluation complete for $save_dir"
    fi
}

# ── Model list ────────────────────────────────────────────────────────────────
# Add or remove save_dirs here. Each should contain fold_0/, fold_1/, etc.

run_eval "final_training/full/config1"
run_eval "final_training/full/config2"
run_eval "final_training/full/config3"
run_eval "final_training/full/config4"
run_eval "final_training/full/config5"
run_eval "final_training/full/config6"
run_eval "final_training/full/config7"
run_eval "final_training/full/config8"
run_eval "final_training/full/config9"
run_eval "final_training/full/config10"
run_eval "final_training/full/config11"
run_eval "final_training/full/config12"
run_eval "final_training/full/config13"
run_eval "final_training/full/config14"
run_eval "final_training/full/config15"

run_eval "final_training/window/config1"
run_eval "final_training/window/config2"
run_eval "final_training/window/config4"
run_eval "final_training/window/config6"
run_eval "final_training/window/config9"
run_eval "final_training/window/config10"
run_eval "final_training/window/config12"
run_eval "final_training/window/config13"
run_eval "final_training/window/config14"
run_eval "final_training/window/config15"
run_eval "final_training/window/config16"

echo ""
echo "======================================================================"
echo "All evaluations finished."
echo "======================================================================"
