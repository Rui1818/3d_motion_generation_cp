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
    local lambda_transl=${16}
    local lambda_rot=${17}

    echo "--- Starting Training: $save_dir ---"
    echo "Latent Dim: $latent_dim, Layers: $layers, Weight Decay: $weight_decay, Steps: $num_steps, LR: $lr"
    echo "Keypoint Type: $keypointtype, Cond Mask Prob: $cond_mask_prob, Lambda Rot Vel: $lambda_rot_vel, Lambda Transl Vel: $lambda_transl_vel"
    echo "Motion N Features: $motionnfeatures, LR Anneal: $lr_anneal, Motion Length: $motion_length, Loss Func: $loss_func, Use DCT: $use_dct"
    echo "Lambda Transl: $lambda_transl, Lambda Rot: $lambda_rot"

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
        --lambda_transl "${lambda_transl:-1.0}" \
        --lambda_rot "${lambda_rot:-1.0}" \
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
#run_training "final_training/full/config1" 512 8 1e-4 160000 2e-4 openpose 0 0 0 69 18000 240 mse

#velocity
#run_training "final_training/full/config2" 512 8 1e-4 160000 2e-4 openpose 0 1 0 69 18000 240 mse

#cond_mask_prob 0.1
#run_training "final_training/full/config3" 512 12 1e-4 160000 2e-4 openpose 0.1 0 0 69 18000 240 mse
#run_training "final_training/full/config4" 512 8 1e-4 160000 2e-4 openpose 0.1 1 0 69 18000 240 mse

# 6d rotations
#run_training "final_training/full/config5" 256 8 1e-4 160000 2e-4 6d 0 0 0 135 18000 240 mse
#run_training "final_training/full/config6" 512 8 1e-4 160000 2e-4 6d 0 0 0 135 18000 240 mse
#velocity
#run_training "final_training/full/config7" 256 8 1e-4 80000 2e-4 6d 0 0.5 0.5 135 18000 240 mse

#velocity + cond_mask_prob 0.1
#run_training "final_training/full/config8" 512 8 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 18000 240 mse
#run_training "final_training/full/config9" 256 12 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 18000 240 mse
#run_training "final_training/full/config10" 512 12 1e-4 80000 2e-4 openpose 0 0 0 69 18000 240 mse
#run_training "final_training/full/config11" 512 8 1e-4 80000 2e-4 openpose 0 1 0 69 18000 240 mse

#softdtw training
#run_training "final_training/full/config12" 512 8 1e-4 80000 2e-4 openpose 0 0 0 69 20000 240 softdtw
#run_training "final_training/full/config13" 512 12 1e-4 80000 2e-4 openpose 0 0 0 69 20000 240 softdtw

#torunnnn

#run_training "final_training/full/config15" 512 8 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 20000 240 softdtw

#short motion windows
#run_training "final_training/window/config1" 512 8 1e-4 80000 2e-4 openpose 0 0 0 69 32000 30 mse
#run_training "final_training/window/config2" 512 8 1e-4 80000 2e-4 openpose 0 1 0 69 32000 30 mse
#run_training "final_training/window/config4" 512 8 1e-4 80000 2e-4 openpose 0 0 0.5 69 32000 60 mse
#run_training "final_training/window/config6" 512 8 1e-4 80000 2e-4 openpose 0.1 1 0 69 32000 60 mse
#run_training "final_training/window/config9" 512 8 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 32000 30 mse
#run_training "final_training/window/config10" 512 8 1e-4 80000 2e-4 6d 0 0 0 135 32000 60 mse
#run_training "final_training/window/config12" 512 8 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 32000 60 mse

#softdtw with short windows
#run_training "final_training/window/config13" 512 8 1e-4 80000 2e-4 openpose 0.1 1 0 69 0 60 softdtw
#run_training "final_training/window/config14" 512 8 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 0 60 softdtw
#run_training "final_training/window/config15" 512 8 1e-4 80000 2e-4 openpose 0.1 1 0 69 0 30 softdtw
#run_training "final_training/window/config16" 512 8 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 0 30 softdtw






#run_training "final_training/dctmlp/config1" 512 8 1e-4 80000 2e-4 openpose 0 0 0 69 30000 30 mse True
#run_training "final_training/dctmlp/config2" 512 8 1e-4 80000 2e-4 openpose 0.1 1 0 69 30000 30 mse True
#run_training "final_training/dctmlp/config3" 512 8 1e-4 80000 2e-4 openpose 0 0 0.5 69 30000 60 mse True
#run_training "final_training/dctmlp/config4" 512 8 1e-4 80000 2e-4 openpose 0.1 1 0 69 30000 60 mse True
#run_training "final_training/dctmlp/config5" 512 8 1e-4 80000 2e-4 6d 0 0 0 135 30000 30 mse True
#run_training "final_training/dctmlp/config6" 512 8 1e-4 80000 2e-4 6d 0 0.5 0.5 135 30000 30 mse True
#run_training "final_training/dctmlp/config7" 512 8 1e-4 80000 2e-4 6d 0.05 0.1 0.5 135 30000 60 mse True
#run_training "final_training/dctmlp/config8" 512 8 1e-4 80000 2e-4 6d 0 0.1 0.5 135 30000 60 mse True
#run_training "final_training/dctmlp/config9" 512 8 1e-4 80000 2e-4 openpose 0.1 1 0 69 30000 60 softdtw True
#run_training "final_training/dctmlp/config10" 512 8 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 30000 60 softdtw True
#run_training "final_training/dctmlp/config11" 512 8 1e-4 80000 2e-4 openpose 0.1 1 0 69 30000 30 softdtw True
#run_training "final_training/dctmlp/config12" 512 8 1e-4 80000 2e-4 6d 0.1 0.5 0.5 135 30000 30 softdtw True

#run_training "final_training/full/config14" 512 8 1e-4 80000 2e-4 6d 0 0.5 0.5 135 20000 240 softdtw

#transformers 
#run_training "my_training/transformer/config1" 512 8 1e-4 70000 2e-4 openpose 0 0 0 69 30000 30 mse
#run_training "my_training/transformer/config2" 512 8 1e-4 70000 2e-4 openpose 0 1 0 69 30000 30 mse
#run_training "my_training/transformer/config3" 512 8 1e-4 70000 2e-4 openpose 0 0 0.5 69 30000 60 mse
#run_training "my_training/transformer/config4" 512 8 1e-4 70000 2e-4 openpose 0.1 1 0 69 30000 60 mse
#run_training "my_training/transformer/config5" 512 8 1e-4 70000 2e-4 6d 0 0 0 135 30000 30 mse
#run_training "my_training/transformer/config6" 512 8 1e-4 70000 2e-4 6d 0 0.5 0.5 135 30000 30 mse
#run_training "my_training/transformer/config7" 512 8 1e-4 70000 2e-4 6d 0 0.1 0.5 135 30000 60 mse
#run_training "my_training/transformer/config8" 512 8 1e-4 70000 2e-4 6d 0.05 0.1 0.5 135 30000 60 mse
#run_training "my_training/transformer/config9" 512 8 1e-4 70000 2e-4 openpose 0.1 1 0 69 30000 60 softdtw
#run_training "my_training/transformer/config10" 512 8 1e-4 70000 2e-4 6d 0.1 0.5 0.5 135 30000 60 softdtw
#run_training "my_training/transformer/config11" 512 8 1e-4 70000 2e-4 openpose 0.1 1 0 69 30000 30 softdtw
#run_training "my_training/transformer/config12" 512 8 1e-4 70000 2e-4 6d 0.1 0.5 0.5 135 30000 30 softdtw


#run_training "final_training/dcttransformer/config1" 512 8 1e-4 70000 2e-4 openpose 0 0 0 69 30000 30 mse True
#run_training "final_training/dcttransformer/config2" 512 8 1e-4 70000 2e-4 openpose 0.1 1 0 69 30000 30 mse True
#run_training "final_training/dcttransformer/config3" 512 8 1e-4 70000 2e-4 openpose 0 0 0.5 69 30000 60 mse True
#run_training "final_training/dcttransformer/config4" 512 8 1e-4 70000 2e-4 openpose 0.1 1 0 69 30000 60 mse True
#run_training "final_training/dcttransformer/config5" 512 8 1e-4 70000 2e-4 6d 0 0 0 135 30000 30 mse True
#run_training "final_training/dcttransformer/config6" 512 8 1e-4 70000 2e-4 6d 0 0.5 0.5 135 30000 30 mse True
#run_training "final_training/dcttransformer/config7" 512 8 1e-4 70000 2e-4 6d 0.05 0.1 0.5 135 30000 60 mse True
#run_training "final_training/dcttransformer/config8" 512 8 1e-4 70000 2e-4 6d 0 0.1 0.5 135 30000 60 mse True
#run_training "final_training/dcttransformer/config9" 512 8 1e-4 70000 2e-4 openpose 0.1 1 0 69 30000 60 softdtw True
#run_training "final_training/dcttransformer/config10" 512 8 1e-4 70000 2e-4 6d 0.1 0.5 0.5 135 30000 60 softdtw True
#run_training "final_training/dcttransformer/config11" 512 8 1e-4 70000 2e-4 openpose 0.1 1 0 69 30000 30 softdtw True
#run_training "final_training/dcttransformer/config12" 512 8 1e-4 70000 2e-4 6d 0.1 0.5 0.5 135 30000 30 softdtw True


run_training "final_training/weighttransformer/config1" 512 8 1e-4 70000 2e-4 6d 0 0 0 135 30000 30 mse "" 1.0 1.0
run_training "final_training/weighttransformer/config2" 512 8 1e-4 70000 2e-4 6d 0 0 0 135 30000 30 mse "" 5.0 1.0
run_training "final_training/weighttransformer/config3" 512 8 1e-4 70000 2e-4 6d 0 0 0 135 30000 30 mse "" 1.0 0.5
run_training "final_training/weighttransformer/config4" 512 8 1e-4 70000 2e-4 6d 0 0 0 135 30000 30 mse "" 5.0 0.5


echo "--- All Experiments Finished ---"