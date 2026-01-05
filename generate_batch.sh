#!/bin/bash

# ================= CONFIGURATION =================

# 1. Automatically find all configuration folders in the training directory.
#CONFIGS=($(find my_training -mindepth 1 -maxdepth 1 -not -path "my_training/overfit_run" -type d -exec basename {} \;))
#openpose
#CONFIGS=("config1_new" "config2_new" "config3_new" "config4_new" "config5_new" "config6_new" "config7_new" "config25_new" "config26_new" "config27_new")
#CONFIGS=("config_sdtw1" "config_sdtw2")
#CONFIGS=("config8_new" "config9_new" "config12_new" "config14_new" "config15_new" "config16_new" "config28_new" "config29_new" "config30_new")
#CONFIGS=("config_sdtw3")

#CONFIGS=("config_window1" "config_window2" "config_window3" "config_window7" "config_window8" "config_window9")
CONFIGS=("config_window4" "config_window5" "config_window6" "config_window10" "config_window11" "config_window12")
# 2. List your model checkpoint names here (without the .pt extension)
# The script assumes these specific filenames exist inside every config folder
#MODELS=("model000042263" "model000026013")
#MODELS=("model000019513" "model000026013" "model000032513" "model000065013")
MODELS=("model000054281" "model000077531" "model000116281" "model000160022")

# 3. Common settings
INPUT_LENGTH=60
BASE_TRAINING_DIR="my_training"
BASE_RESULTS_DIR="results/window60"

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
            --output_dir "$OUTPUT_PATH" \
            
    done
done

echo "------------------------------------------------"
echo "Batch processing complete."