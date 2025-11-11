#!/bin/bash

# --- Static Parameters ---
# Settings that are the same for all experiments
DATASET_PATH="observations"
DATASET_NAME="gait"
BATCH_SIZE=2
SAVE_INTERVAL=1000
LOG_INTERVAL=100
DEVICE=0
NUM_WORKERS=2
ARCH="diffusion_DiffMLP"
MOTION_LENGTH=296

# --- Helper Function ---
# This function runs a single training experiment
# It takes arguments for the parameters we want to change
run_training() {
    # $1: Save Directory (e.g., "my_training/config1")
    # $2: Latent Dimension (e.g., 128)
    # $3: Num Layers (e.g., 8)
    # $4: Weight Decay (e.g., 1e-4)

    local save_dir=$1
    local latent_dim=$2
    local layers=$3
    local weight_decay=$4
    local num_steps=$5
    local lr=$6

    echo "--- Starting Training: $save_dir ---"
    echo "    Latent Dim: $latent_dim, Layers: $layers, Weight Decay: $weight_decay, Steps: $num_steps, LR: $lr"

    python gait_train.py \
        --save_dir "$save_dir" \
        --dataset_path "$DATASET_PATH" \
        --dataset "$DATASET_NAME" \
        --weight_decay "$weight_decay" \
        --batch_size "$BATCH_SIZE" \
        --latent_dim "$latent_dim" \
        --save_interval "$SAVE_INTERVAL" \
        --log_interval "$LOG_INTERVAL" \
        --device "$DEVICE" \
        --num_workers "$NUM_WORKERS" \
        --arch "$ARCH" \
        --layers "$layers" \
        --num_steps "$num_steps" \
        --input_motion_length "$MOTION_LENGTH" \
        --lr "$lr" \
        --overwrite

    # Check if training was successful
    if [ $? -ne 0 ]; then
        echo "--- !!! Training FAILED for $save_dir !!! ---"
    else
        echo "--- Training Complete for $save_dir ---"
    fi
}

# --- Experiment Definitions ---
# Now, we just call the function for each configuration we want to test.
# The script will run these one after another.

#run_training "my_training/config1" 128 8 1e-4 20000 2e-4
#run_training "my_training/config2" 256 8 1e-4 20000 2e-4
#run_training "my_training/config3" 512 8 1e-4 20000 2e-4
#run_training "my_training/config4" 128 6 1e-4 20000 2e-4
#run_training "my_training/config5" 256 6 1e-4 20000 2e-4
run_training "my_training/config6" 512 8 1e-4 60000 2e-4
#run_training "my_training/config7" 128 8 1e-4 20000 2e-4
#run_training "my_training/config8" 256 8 1e-4 60000 2e-4
run_training "my_training/config9" 256 8 1e-4 60000 2e-4

# Config 4: Add more configs as you like...
# run_training "my_training/config4_..." ... ... ...

echo "--- All Experiments Finished ---"