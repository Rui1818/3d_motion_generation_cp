#!/bin/bash

# ================= CONFIGURATION =================

# 1. Automatically find all configuration folders in the training directory.
#    This removes the need to list them manually, and excludes 'overfit_run'.
#CONFIGS=($(find my_training -mindepth 1 -maxdepth 1 -not -path "my_training/overfit_run" -type d -exec basename {} \;))
CONFIGS=("config14" "config15" "config16" "config17" "config18" "config19" "config20" "config21" "config22" "config23" "config24")
# 2. List your model checkpoint names here (without the .pt extension)
# The script assumes these specific filenames exist inside every config folder
MODELS=("model000080002" "model000070213" "model000059813" "model000050713" "model000045513" "model000040313" "model000035113" "model000029913" "model000024713" "model000019513" "model000014313")

# 3. Common settings
INPUT_LENGTH=240
KEYPOINT_TYPE="6d"
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