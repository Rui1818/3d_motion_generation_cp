#!/bin/bash

# ================= CONFIGURATION =================

# 1. List your configuration folder names here
CONFIGS=("newconfig1" "newconfig2")

# 2. List your model checkpoint names here (without the .pt extension)
# The script assumes these specific filenames exist inside every config folder
MODELS=("model000080002" "model000090000" "model000100000")

# 3. Common settings
INPUT_LENGTH=240
KEYPOINT_TYPE="openpose"
BASE_TRAINING_DIR="my_training"
BASE_RESULTS_DIR="results"

# =================================================

# Loop through every configuration
for config in "${CONFIGS[@]}"; do
    
    # Loop through every model checkpoint
    for model in "${MODELS[@]}"; do
        
        # Construct the specific paths
        # Input: my_training/config/model.pt
        FULL_MODEL_PATH="${BASE_TRAINING_DIR}/${config}/${model}.pt"
        
        # Output: results/config/model
        OUTPUT_PATH="${BASE_RESULTS_DIR}/${config}/${model}"

        echo "------------------------------------------------"
        echo "Processing: $config | $model"
        echo "Input:  $FULL_MODEL_PATH"
        echo "Output: $OUTPUT_PATH"
        
        # Optional: Create output directory explicitly if python script doesn't do it
        # mkdir -p "$OUTPUT_PATH"

        # Execute the Python command
        python gait_generate.py \
            --model_path "$FULL_MODEL_PATH" \
            --input_motion_length "$INPUT_LENGTH" \
            --keypointtype "$KEYPOINT_TYPE" \
            --output_dir "$OUTPUT_PATH"
            
    done
done

echo "------------------------------------------------"
echo "Batch processing complete."