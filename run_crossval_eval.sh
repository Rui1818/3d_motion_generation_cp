#!/bin/bash

# Cross-validation evaluation runner.
# Runs gait_crossval_eval.py for each model save_dir listed below.

DATASET_PATH="final_dataset"
NUM_FOLDS=5
SEED=10

# Optional: path to autoencoder checkpoint for FID computation.
# Leave empty to skip FID.

# ── Helper ────────────────────────────────────────────────────────────────────

run_eval() {
    local save_dir=$1
    local ae=$2
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
        --seed $SEED
        --autoencoder_path \"$ae\""

    eval $cmd

    if [ $? -ne 0 ]; then
        echo "  !!! Evaluation FAILED for $save_dir !!!"
    else
        echo "  Evaluation complete for $save_dir"
    fi
}

# ── Model list ────────────────────────────────────────────────────────────────
# Add or remove save_dirs here. Each should contain fold_0/, fold_1/, etc.

#run_eval "final_training/full/config1"
#run_eval "final_training/full/config2"
#run_eval "final_training/full/config3"
#run_eval "final_training/full/config4"
#run_eval "final_training/full/config5"
#run_eval "final_training/full/config6"
#run_eval "final_training/full/config7"
#run_eval "final_training/full/config8"
#run_eval "final_training/full/config9"
#run_eval "final_training/full/config10"
#run_eval "final_training/full/config11"
#run_eval "final_training/full/config12"

#run_eval "final_training/window/config1" checkpoints/experiments/w30openpose/best_autoencoder.pt
#run_eval "final_training/window/config4" checkpoints/experiments/w60openpose/best_autoencoder.pt
#run_eval "final_training/window/config6" checkpoints/experiments/w60openpose/best_autoencoder.pt
#run_eval "final_training/window/config9" checkpoints/experiments/w30rot/best_autoencoder.pt
#run_eval "final_training/window/config10" checkpoints/experiments/w60rot/best_autoencoder.pt
#run_eval "final_training/window/config12" checkpoints/experiments/w60rot/best_autoencoder.pt
#run_eval "final_training/window/config13" checkpoints/experiments/w60openpose/best_autoencoder.pt
#run_eval "final_training/window/config14" checkpoints/experiments/w60rot/best_autoencoder.pt
#run_eval "final_training/window/config15" checkpoints/experiments/w30openpose/best_autoencoder.pt
#run_eval "final_training/window/config16" checkpoints/experiments/w30rot/best_autoencoder.pt


run_eval "my_training/transformer/config1" checkpoints/experiments/w30openpose/best_autoencoder.pt
run_eval "my_training/transformer/config2" checkpoints/experiments/w30openpose/best_autoencoder.pt
run_eval "my_training/transformer/config3" checkpoints/experiments/w60openpose/best_autoencoder.pt
run_eval "my_training/transformer/config4" checkpoints/experiments/w60openpose/best_autoencoder.pt
run_eval "my_training/transformer/config5" checkpoints/experiments/w30rot/best_autoencoder.pt
run_eval "my_training/transformer/config6" checkpoints/experiments/w30rot/best_autoencoder.pt
run_eval "my_training/transformer/config7" checkpoints/experiments/w60rot/best_autoencoder.pt
run_eval "my_training/transformer/config8" checkpoints/experiments/w60rot/best_autoencoder.pt
run_eval "my_training/transformer/config9" checkpoints/experiments/w60openpose/best_autoencoder.pt
run_eval "my_training/transformer/config10" checkpoints/experiments/w60rot/best_autoencoder.pt
run_eval "my_training/transformer/config11" checkpoints/experiments/w30openpose/best_autoencoder.pt
#run_eval "my_training/transformer/config12" checkpoints/experiments/w30rot/best_autoencoder.pt

echo ""
echo "======================================================================"
echo "All evaluations finished."
echo "======================================================================"
