#!/bin/bash

# --- Static Parameters ---
# Settings that are the same for all experiments
DATASET_PATH="final_dataset"
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
    local use_dct=${15}

    echo "--- Starting Training: $save_dir ---"
    echo "Latent Dim: $latent_dim, Layers: $layers, Weight Decay: $weight_decay, Steps: $num_steps, LR: $lr"
    echo "Keypoint Type: $keypointtype, Cond Mask Prob: $cond_mask_prob, Lambda Rot Vel: $lambda_rot_vel, Lambda Transl Vel: $lambda_transl_vel"
    echo "Motion N Features: $motionnfeatures, LR Anneal: $lr_anneal, Motion Length: $motion_length, Loss Func: $loss_func, Use DCT: $use_dct"

    python gait_crossval.py \
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
        ${use_dct:+--use_dct} \
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
run_training "final_training/full/config1" 512 8 1e-4 160000 2e-4 openpose 0 0 0 69 18000 240 mse

#velocity
run_training "final_training/full/config2" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 18000 240 mse

#cond_mask_prob 0.1
run_training "final_training/full/config3" 512 12 1e-4 160000 2e-4 openpose 0.1 0 0 69 18000 240 mse
run_training "final_training/full/config4" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 18000 240 mse

# 6d rotations
run_training "final_training/full/config5" 256 8 1e-4 160000 2e-4 6d 0 0 0 135 18000 240 mse
run_training "final_training/full/config6" 512 8 1e-4 160000 2e-4 6d 0 0 0 135 18000 240 mse
#velocity
run_training "final_training/full/config7" 256 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 18000 240 mse

#velocity + cond_mask_prob 0.1
run_training "final_training/full/config8" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 18000 240 mse
run_training "final_training/full/config9" 256 12 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 18000 240 mse
run_training "final_training/full/config10" 512 12 1e-4 160000 2e-4 openpose 0 0 0 69 18000 240 mse
run_training "final_training/full/config11" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 18000 240 mse

#softdtw training
run_training "final_training/full/config12" 512 8 1e-4 160000 2e-4 openpose 0 0 0 69 20000 240 softdtw
run_training "final_training/full/config13" 512 12 1e-4 160000 2e-4 openpose 0 0 0 69 20000 240 softdtw
run_training "final_training/full/config14" 512 8 1e-4 160000 2e-4 6d 0 0.5 0.5 135 20000 240 softdtw
run_training "final_training/full/config15" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 20000 240 softdtw

#short motion windows
run_training "final_training/window/config1" 512 8 1e-4 160000 2e-4 openpose 0 0 0 69 32000 30 mse
run_training "final_training/window/config2" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 32000 30 mse
run_training "final_training/window/config4" 512 8 1e-4 160000 2e-4 openpose 0 0 0.5 69 32000 60 mse
run_training "final_training/window/config6" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 32000 60 mse
run_training "final_training/window/config9" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 32000 30 mse
run_training "final_training/window/config10" 512 8 1e-4 160000 2e-4 6d 0 0 0 135 32000 60 mse
run_training "final_training/window/config12" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 32000 60 mse

#softdtw with short windows
run_training "final_training/window/config13" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 0 60 softdtw
run_training "final_training/window/config14" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 0 60 softdtw
run_training "final_training/window/config15" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 0 30 softdtw
run_training "final_training/window/config16" 512 8 1e-4 160000 2e-4 6d 0.1 0.5 0.5 135 0 30 softdtw




#transformers 
#run_training "my_training/config_transformer1_v2" 512 8 1e-4 200000 2e-4 openpose 0 0 0 69 30000 30 mse
#run_training "my_training/config_transformer2_v2" 512 8 1e-4 200000 2e-4 openpose 0 1 0 69 30000 30 mse
#run_training "my_training/config_transformer3_v2" 512 8 1e-4 200000 2e-4 openpose 0.1 1 0 69 30000 30 mse
#run_training "my_training/config_transformer4_v2" 512 8 1e-4 200000 2e-4 openpose 0 0 0.5 69 30000 60 mse
#run_training "my_training/config_transformer5_v2" 512 8 1e-4 200000 2e-4 openpose 0 1 0 69 30000 60 mse
#run_training "my_training/config_transformer6_v2" 512 8 1e-4 200000 2e-4 openpose 0.1 1 0 69 30000 60 mse
#run_training "my_training/config_transformer7_v2" 512 8 1e-4 200000 2e-4 6d 0 0 0 135 30000 30 mse
#run_training "my_training/config_transformer8_v2" 512 8 1e-4 200000 2e-4 6d 0 0.5 0.5 135 30000 30 mse
#run_training "my_training/config_transformer9_v2" 512 8 1e-4 200000 2e-4 6d 0.05 0.5 0.5 135 30000 30 mse
#run_training "my_training/config_transformer10_v2" 512 8 1e-4 200000 2e-4 6d 0 0 0 135 30000 60 mse
#run_training "my_training/config_transformer11_v2" 512 8 1e-4 200000 2e-4 6d 0 0.1 0.5 135 30000 60 mse
#run_training "my_training/config_transformer12_v2" 512 8 1e-4 200000 2e-4 6d 0.05 0.1 0.5 135 30000 60 mse
#run_training "my_training/config_transformer13_v2" 512 8 1e-4 200000 2e-4 openpose 0.1 1 0 69 30000 60 softdtw
#run_training "my_training/config_transformer14_v2" 512 8 1e-4 200000 2e-4 6d 0.1 0.5 0.5 135 30000 60 softdtw
#run_training "my_training/config_transformer15_v2" 512 8 1e-4 200000 2e-4 openpose 0.1 1 0 69 30000 30 softdtw
#run_training "my_training/config_transformer16_v2" 512 8 1e-4 200000 2e-4 6d 0.1 0.5 0.5 135 30000 30 softdtw

#run_training "my_training/config_dctmlp" 512 8 1e-4 170000 2e-4 openpose 0 0 0 69 30000 30 mse True
#run_training "my_training/config_dctmlp2" 512 8 1e-4 170000 2e-4 openpose 0 1 0 69 30000 30 mse True
#run_training "my_training/config_dctmlp3" 512 8 1e-4 170000 2e-4 openpose 0.1 1 0 69 30000 30 mse True
#run_training "my_training/config_dctmlp4" 512 8 1e-4 170000 2e-4 openpose 0 0 0.5 69 30000 60 mse True
#run_training "my_training/config_dctmlp5" 512 8 1e-4 170000 2e-4 openpose 0 1 0 69 30000 60 mse True
#run_training "my_training/config_dctmlp6" 512 8 1e-4 170000 2e-4 openpose 0.1 1 0 69 30000 60 mse True
#run_training "my_training/config_dctmlp7" 512 8 1e-4 170000 2e-4 6d 0 0 0 135 30000 30 mse True
#run_training "my_training/config_dctmlp8" 512 8 1e-4 170000 2e-4 6d 0 0.5 0.5 135 30000 30 mse True
#run_training "my_training/config_dctmlp9" 512 8 1e-4 170000 2e-4 6d 0.05 0.5 0.5 135 30000 30 mse True
#run_training "my_training/config_dctmlp10" 512 8 1e-4 170000 2e-4 6d 0.05 0.1 0.5 135 30000 60 mse True
run_training "my_training/config_dctmlp11" 512 8 1e-4 200000 2e-4 6d 0 0.1 0.5 135 30000 60 mse True
run_training "my_training/config_dctmlp12" 512 8 1e-4 200000 2e-4 6d 0.05 0.1 0.5 135 30000 60 mse True
run_training "my_training/config_dctmlp13" 512 8 1e-4 170000 2e-4 openpose 0.1 1 0 69 30000 60 softdtw True
run_training "my_training/config_dctmlp14" 512 8 1e-4 170000 2e-4 6d 0.1 0.5 0.5 135 30000 60 softdtw True
run_training "my_training/config_dctmlp15" 512 8 1e-4 170000 2e-4 openpose 0.1 1 0 69 30000 30 softdtw True
run_training "my_training/config_dctmlp16" 512 8 1e-4 170000 2e-4 6d 0.1 0.5 0.5 135 30000 30 softdtw True

# Config 4: Add more configs as you like...
# run_training "my_training/config4_..." ... ... ...

echo "--- All Experiments Finished ---"