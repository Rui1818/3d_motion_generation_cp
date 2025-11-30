#!/bin/bash

# --- Static Parameters ---
# Settings that are the same for all experiments
DATASET_PATH="mydataset"
DATASET_NAME="gait"
BATCH_SIZE=8
SAVE_INTERVAL=5000
LOG_INTERVAL=1000
DEVICE=0
NUM_WORKERS=4
ARCH="diffusion_DiffMLP"
MOTION_LENGTH=240

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
    local keypointtype=$7
    local cond_mask_prob=$8
    local lambda_rot_vel=$9
    local lambda_transl_vel=${10}
    local motionnfeatures=${11}

    echo "--- Starting Training: $save_dir ---"
    echo "Latent Dim: $latent_dim, Layers: $layers, Weight Decay: $weight_decay, Steps: $num_steps, LR: $lr"

    python gait_train.py \
        --save_dir "$save_dir" \
        --dataset_path "$DATASET_PATH" \
        --input_motion_length "$MOTION_LENGTH" \
        --dataset "$DATASET_NAME" \
        --save_interval "$SAVE_INTERVAL" \
        --batch_size "$BATCH_SIZE" \
        --log_interval "$LOG_INTERVAL" \
        --num_workers "$NUM_WORKERS" \
        --device "$DEVICE" \
        --arch "$ARCH" \
        --latent_dim "$latent_dim" \
        --layers "$layers" \
        --weight_decay "$weight_decay" \
        --num_steps "$num_steps" \
        --lr "$lr" \
        --keypointtype "$keypointtype" \
        --cond_mask_prob "$cond_mask_prob" \
        --lambda_rot_vel "$lambda_rot_vel" \
        --lambda_transl_vel "$lambda_transl_vel" \
        --motion_nfeat "$motionnfeatures" \
        --sparse_dim "$motionnfeatures" \
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
    local save_dir=$1
    local latent_dim=$2
    local layers=$3
    local weight_decay=$4
    local num_steps=$5
    local lr=$6
    local keypointtype=$7
    local cond_mask_prob=$8
    local lambda_rot_vel=$9
    local lambda_transl_vel=${10}
    local motionnfeatures=${11}

# Openpose keypoints

run_training "my_training/config1" 256 6 1e-4 500000 2e-4 openpose 0 0 0 69
run_training "my_training/config2" 512 6 1e-4 500000 2e-4 openpose 0 0 0 69
run_training "my_training/config3" 256 8 1e-4 500000 2e-4 openpose 0 0 0 69
run_training "my_training/config4" 512 8 1e-4 500000 2e-4 openpose 0 0 0 69
run_training "my_training/config5" 256 4 1e-4 500000 2e-4 openpose 0 0 0 69
run_training "my_training/config6" 512 4 1e-4 500000 2e-4 openpose 0 0 0 69
#velocity
run_training "my_training/config7" 256 6 1e-4 500000 2e-4 openpose 0 0.5 0 69
run_training "my_training/config8" 512 6 1e-4 500000 2e-4 openpose 0 0.5 0 69
run_training "my_training/config9" 256 8 1e-4 500000 2e-4 openpose 0 0.5 0 69
run_training "my_training/config10" 512 8 1e-4 500000 2e-4 openpose 0 0.5 0 69
run_training "my_training/config11" 256 4 1e-4 500000 2e-4 openpose 0 0.5 0 69
run_training "my_training/config12" 512 4 1e-4 500000 2e-4 openpose 0 0.5 0 69

# 6d rotations
run_training "my_training/config13" 256 6 1e-4 500000 2e-4 6d 0 0 0 135
run_training "my_training/config14" 512 6 1e-4 500000 2e-4 6d 0 0 0 135
run_training "my_training/config15" 256 8 1e-4 500000 2e-4 6d 0 0 0 135
run_training "my_training/config16" 512 8 1e-4 500000 2e-4 6d 0 0 0 135
run_training "my_training/config17" 256 4 1e-4 500000 2e-4 6d 0 0 0 135
run_training "my_training/config18" 512 4 1e-4 500000 2e-4 6d 0 0 0 135
#velocity
run_training "my_training/config19" 256 6 1e-4 500000 2e-4 6d 0 0.5 0.1 135
run_training "my_training/config20" 512 6 1e-4 500000 2e-4 6d 0 0.5 0.1 135
run_training "my_training/config21" 256 8 1e-4 500000 2e-4 6d 0 0.5 0.1 135
run_training "my_training/config22" 512 8 1e-4 500000 2e-4 6d 0 0.5 0.1 135
run_training "my_training/config23" 256 8 1e-4 500000 2e-4 6d 0 0.5 0.1 135
run_training "my_training/config24" 512 8 1e-4 500000 2e-4 6d 0 0.5 0.1 135

# Config 4: Add more configs as you like...
# run_training "my_training/config4_..." ... ... ...

echo "--- All Experiments Finished ---"