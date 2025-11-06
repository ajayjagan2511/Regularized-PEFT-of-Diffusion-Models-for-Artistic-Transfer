#!/bin/bash

echo "Starting FID image generation process..."

# --- Configuration ---
# Path to your LoRA model trained WITHOUT regularization
export LORA_BASELINE_PATH="./outputs/lora_baseline_monet/checkpoint-1000/"

# Path to your LoRA model trained WITH regularization
export LORA_REG_PATH="./outputs/lora_paired_reg_monet/checkpoint-1000/"

# Number of images to generate for each model
export NUM_IMAGES=250

# --- 1. Generate images for the BASE SDXL model ---
echo "Generating images for BASE model..."
python src/generate_for_fid.py \
    --model_type base \
    --output_dir ./outputs/fid_images_base \
    --num_images $NUM_IMAGES

# --- 2. Generate images for the LoRA BASELINE model ---
echo "Generating images for LoRA BASELINE model..."
python src/generate_for_fid.py \
    --model_type lora \
    --lora_weights_path $LORA_BASELINE_PATH \
    --output_dir ./outputs/fid_images_lora_baseline \
    --num_images $NUM_IMAGES

# --- 3. Generate images for the REGULARIZED LoRA model ---
echo "Generating images for REGULARIZED LoRA model..."
python src/generate_for_fid.py \
    --model_type reg_lora \
    --lora_weights_path $LORA_REG_PATH \
    --output_dir ./outputs/fid_images_lora_regularized \
    --num_images $NUM_IMAGES

echo "All image sets have been generated."
echo "You can now run the torch-fidelity commands to calculate the scores."