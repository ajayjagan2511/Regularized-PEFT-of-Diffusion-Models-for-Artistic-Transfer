#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Fixed Parameters ---
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
VAE_MODEL="madebyollin/sdxl-vae-fp16-fix"
DATA_PARENT_DIR="/home/shivamsinghal/LTC/GEN/data"

# Train Folders
MONET_DIR_NAME="monet_style"
ORIGINAL_DIR_NAME="monet_style_original"

# Validation Folders
MONET_VAL_DIR_NAME="monet_style_val"
ORIGINAL_VAL_DIR_NAME="monet_style_original_val"

# --- WANDB SETUP ---
PROJECT_NAME="cvpr-regularized-sweep-v2" 

# --- Hyperparameter Search Space ---
# FFT requires much lower learning rates than LoRA
LEARNING_RATES=(1e-6 5e-6) 

# --- Main Loop ---
for lr in "${LEARNING_RATES[@]}"; do
    
    # 1. Generate a unique name
    RUN_NAME="FFT_lr_${lr}_baseline_batch_4"
    OUTPUT_DIR="./outputs/${RUN_NAME}"

    echo "================================================="
    echo "STARTING RUN: ${RUN_NAME}"
    echo "Project: ${PROJECT_NAME}"
    echo "Outputting to: ${OUTPUT_DIR}"
    echo "================================================="

    # 2. Execute Training
    # Using gradient_accumulation_steps=4 to simulate larger batches on single GPU
    CUDA_VISIBLE_DEVICES='0' accelerate launch src/train_full_finetune.py \
    --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
    --pretrained_vae_model_name_or_path="${VAE_MODEL}" \
    --data_parent_dir="${DATA_PARENT_DIR}" \
    --monet_dir_name="${MONET_DIR_NAME}" \
    --original_dir_name="${ORIGINAL_DIR_NAME}" \
    --monet_val_dir_name="${MONET_VAL_DIR_NAME}" \
    --original_val_dir_name="${ORIGINAL_VAL_DIR_NAME}" \
    --output_dir="${OUTPUT_DIR}" \
    --instance_prompt="sksmonet, a painting in the style of Monet" \
    --resolution=1024 \
    --train_batch_size=4 \
    --gradient_accumulation_steps=1 \
    --learning_rate="${lr}" \
    --num_epochs=20 \
    --save_every_epoch=10 \
    --mixed_precision="fp16" \
    --seed=42 \
    --wandb_project="${PROJECT_NAME}" \
    --wandb_run_name="${RUN_NAME}"

    echo "-------------------------------------------------"
    echo "FINISHED RUN: ${RUN_NAME}"
    echo "-------------------------------------------------"

done

echo "🎉 Full Fine-Tuning sweep completed! 🎉"