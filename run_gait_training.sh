#!/bin/bash

# --- Static Parameters ---
# Settings that are the same for all experiments
DATASET_PATH="mydataset"
DATASET_NAME="gait"
BATCH_SIZE=8
SAVE_INTERVAL=200
LOG_INTERVAL=100
DEVICE=0
NUM_WORKERS=4
ARCH="diffusion_DiffMLP"

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
    local lr_anneal=${12}
    local motion_length=${13}
    local loss_func=${14}

    echo "--- Starting Training: $save_dir ---"
    echo "Latent Dim: $latent_dim, Layers: $layers, Weight Decay: $weight_decay, Steps: $num_steps, LR: $lr"

    python gait_train.py \
        --save_dir "$save_dir" \
        --dataset_path "$DATASET_PATH" \
        --input_motion_length "$motion_length" \
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
        --lr_anneal_steps "$lr_anneal" \
        --loss_func "$loss_func" \
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

# Openpose keypoints
#v2 is with lr annealing after 18000 steps and dataset with cartesian product augmentation
#run_training "my_training/config2_v2" 512 8 1e-4 160000 2e-4 openpose 0 0 0 69 18000 240 mse

#velocity
#run_training "my_training/config3_v2" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 18000 240

#cond_mask_prob 0.1
#run_training "my_training/config5_v2" 512 12 1e-4 160000 2e-4 openpose 0.1 0 0 69 18000 240
#run_training "my_training/config7_v2" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 18000 240

# 6d rotations
#run_training "my_training/config8_v2" 256 8 1e-4 160000 2e-4 6d 0 0 0 135 18000 240
#run_training "my_training/config9_v2" 512 8 1e-4 160000 2e-4 6d 0 0 0 135 18000 240
#velocity
#run_training "my_training/config14_v2" 256 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 18000 240

#velocity + cond_mask_prob 0.1
#run_training "my_training/config15_v2" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 18000 240
#run_training "my_training/config16_v2" 256 12 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 18000 240
#run_training "my_training/config25_v2" 512 12 1e-4 160000 2e-4 openpose 0 0 0 69 18000 240
#run_training "my_training/config26_v2" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 18000 240
#run_training "my_training/config28_v2" 512 12 1e-4 160000 2e-4 6d 0 0 0 135 18000 240
#run_training "my_training/config29_v2" 512 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 18000 240

#softdtw training
#run_training "my_training/config_sdtw1_v2" 512 8 1e-4 160000 2e-4 openpose 0 0 0 69 20000 240 softdtw
#run_training "my_training/config_sdtw2_v2" 512 12 1e-4 160000 2e-4 openpose 0 0 0 69 20000 240 softdtw
#run_training "my_training/config_sdtw3_v2" 512 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 20000 240 softdtw
#run_training "my_training/config_sdtw4_v2" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 20000 240 softdtw

#
#local save_dir=$1
#local latent_dim=$2
#local layers=$3
#local weight_decay=$4
#local num_steps=$5
#local lr=$6
#local keypointtype=$7
#local cond_mask_prob=$8
#local lambda_rot_vel=$9
#local lambda_transl_vel=${10}
#local motionnfeatures=${11}
#local lr_anneal=${12}
#local motion_length=${13}

#short motion windows
#anneal factor 20
#run_training "my_training/config_window1_new" 512 8 1e-4 160000 2e-4 openpose 0 0 0 69 32000 30
#run_training "my_training/config_window2_new" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 32000 30
#run_training "my_training/config_window3_new" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 32000 30
#run_training "my_training/config_window4_new" 512 8 1e-4 160000 2e-4 openpose 0 0 0.5 69 32000 60
#run_training "my_training/config_window5_new" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 32000 60
#run_training "my_training/config_window6_new" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 32000 60
#run_training "my_training/config_window7_new" 512 8 1e-4 160000 2e-4 6d 0 0 0 135 32000 30
#run_training "my_training/config_window8_new" 512 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 32000 30
#run_training "my_training/config_window9_new" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 32000 30
#run_training "my_training/config_window10_new" 512 8 1e-4 160000 2e-4 6d 0 0 0 135 32000 60
#run_training "my_training/config_window11_new" 512 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 32000 60
#run_training "my_training/config_window12_new" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 32000 60

#softdtw with short windows
#run_training "my_training/config_window13" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 0 60
#run_training "my_training/config_window14" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 0 60


#short motion windows
#with the adjusted sampling (aligned)
#anneal factor 20
#run_training "my_training/config_window1_v2" 512 8 1e-4 160000 2e-4 openpose 0 0 0 69 0 30 mse
#run_training "my_training/config_window2_v2" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 0 30 mse
#run_training "my_training/config_window3_v2" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 0 30 mse
#run_training "my_training/config_window4_v2" 512 8 1e-4 160000 2e-4 openpose 0 0 0.5 69 0 60 mse
#run_training "my_training/config_window5_v2" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 0 60 mse
#run_training "my_training/config_window6_v2" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 0 60 mse
#run_training "my_training/config_window7_v2" 512 8 1e-4 160000 2e-4 6d 0 0 0 135 0 30 mse
#run_training "my_training/config_window8_v2" 512 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 0 30 mse
#run_training "my_training/config_window9_v2" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 0 30 mse
#run_training "my_training/config_window10_v2" 512 8 1e-4 160000 2e-4 6d 0 0 0 135 0 60 mse
#run_training "my_training/config_window11_v2" 512 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 0 60 mse
#run_training "my_training/config_window12_v2" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 0 60 mse

#softdtw with short windows
#run_training "my_training/config_window13_v2" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 0 60 softdtw
#run_training "my_training/config_window14_v2" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 0 60 softdtw
#run_training "my_training/config_window15_v2" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 0 30 softdtw
#run_training "my_training/config_window16_v2" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 0 30 softdtw

# Config 4: Add more configs as you like...
# run_training "my_training/config4_..." ... ... ...

echo "--- All Experiments Finished ---"