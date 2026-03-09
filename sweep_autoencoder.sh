#!/bin/bash
# Grid search over key autoencoder hyperparameters.
# Each config trains independently and saves to its own subdirectory.
# Run with: bash sweep_autoencoder.sh

DATASET_PATH="mydataset"
KEYPOINTTYPE="6d"       # choices: 6d | openpose | smpl
EPOCHS=200
BASE_SAVE="./checkpoints/autoencoder_sweep"

run() {
    local window=$1
    local latent=$2
    local hidden=$3
    local dropout=$4
    local repeat=$5
    local lr=$6
    local kptype=$7
    local name="w${window}_l${latent}_h${hidden}_d${dropout}_r${repeat}_lr${lr}_kp${kptype}"
    local save_dir="${BASE_SAVE}/${name}"

    echo "============================================"
    echo "Config: $name"
    echo "============================================"

    python train_autoencoder.py \
        --dataset_path        "$DATASET_PATH" \
        --keypointtype        "$kptype" \
        --save_dir            "$save_dir" \
        --input_motion_length "$window" \
        --latent_dim          "$latent" \
        --hidden_dim          "$hidden" \
        --dropout             "$dropout" \
        --repeat_times        "$repeat" \
        --epochs              "$EPOCHS" \
        --lr                  "$lr" \
        --batch_size          16 \
        --weight_decay        1e-4
}

#            window  latent  hidden  dropout  repeat  lr      keypointtype
# ── Baseline ───────────────────────────────────────────────────────────────
run            60     128     256     0.1      4    1e-4    6d
run            60     128     512     0.1      4    1e-4    6d
run            60     128     256     0.1      4    3e-4    6d
run            60      64     256     0.1      4    1e-4    6d
run            60     256     512     0.1      4    1e-4    6d

run            60     128     256     0.1      4    1e-4    openpose
run            60     128     512     0.1      4    1e-4    openpose
run            60     128     256     0.1      4    3e-4    openpose
run            60      64     256     0.1      4    1e-4    openpose
run            60     256     512     0.1      4    1e-4    openpose

run            30     128     256     0.1      10    1e-4    6d
run            30     128     512     0.1      10    1e-4    6d
run            30     128     256     0.1      10    3e-4    6d
run            30      64     256     0.1      10    1e-4    6d
run            30     256     512     0.1      10    1e-4    6d

run            30     128     256     0.1      10    1e-4    openpose
run            30     128     512     0.1      10    1e-4    openpose
run            30     128     256     0.1      10    3e-4    openpose
run            30      64     256     0.1      10    1e-4    openpose
run            30     256     512     0.1      10    1e-4    openpose

echo "============================================"
echo "All configs done. Compare with tensorboard:"
echo "  tensorboard --logdir ${BASE_SAVE}"
echo "============================================"
