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

# Validation Folders (Leave empty string if not creating them yet)
MONET_VAL_DIR_NAME="monet_style_val"
ORIGINAL_VAL_DIR_NAME="monet_style_original_val"

# --- WANDB SETUP ---
PROJECT_NAME="cvpr-regularized-sweep-v2" 

# --- Hyperparameter Search Space ---
LEARNING_RATES=(1e-4 1e-5)
RANKS=(16)
LAMBDAS=(0.0 10)

# --- Main Loop ---
for lr in "${LEARNING_RATES[@]}"; do
    for rank in "${RANKS[@]}"; do
        for lambda_val in "${LAMBDAS[@]}"; do
        
            # 1. Generate a unique name
            RUN_NAME="lr_${lr}_lambda_${lambda_val}_rank_${rank}_reg"
            OUTPUT_DIR="./outputs/${RUN_NAME}"

            echo "================================================="
            echo "STARTING RUN: ${RUN_NAME}"
            echo "Project: ${PROJECT_NAME}"
            echo "Outputting to: ${OUTPUT_DIR}"
            echo "================================================="

            # 2. Execute Training
            CUDA_VISIBLE_DEVICES='0' accelerate launch src/train_lora_reg.py \
            --pretrained_model_name_or_path="${PRETRAINED_MODEL}" \
            --pretrained_vae_model_name_or_path="${VAE_MODEL}" \
            --data_parent_dir="${DATA_PARENT_DIR}" \
            --monet_dir_name="${MONET_DIR_NAME}" \
            --original_dir_name="${ORIGINAL_DIR_NAME}" \
            --monet_val_dir_name="${MONET_VAL_DIR_NAME}" \
            --original_val_dir_name="${ORIGINAL_VAL_DIR_NAME}" \
            --output_dir="${OUTPUT_DIR}" \
            --instance_prompt="sksmonet, a painting in the style of Monet" \
            --style_prompt="a painting in the style of Monet" \
            --prior_prompt="a photo of a landscape" \
            --resolution=1024 \
            --train_batch_size=4 \
            --gradient_accumulation_steps=1 \
            --learning_rate="${lr}" \
            --lora_rank="${rank}" \
            --lambda_reg="${lambda_val}" \
            --num_epochs=20 \
            --save_every_epoch=10 \
            --use_regularization \
            --ema_beta=0.98 \
            --mixed_precision="fp16" \
            --seed=42 \
            --wandb_project="${PROJECT_NAME}" \
            --wandb_run_name="${RUN_NAME}"

            echo "-------------------------------------------------"
            echo "FINISHED RUN: ${RUN_NAME}"
            echo "-------------------------------------------------"
        
        done
    done
done

echo "🎉 All hyperparameter sweep runs completed! 🎉"