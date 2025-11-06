#!/bin/bash

CUDA_VISIBLE_DEVICES='0'

# --- SHARED VARIABLES ---
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"

# --- DATASET CONFIGURATION ---
# Assumes your data is in ./data/monet_style and ./data/monet_style_original
export DATA_PARENT_DIR="./data" 
export MONET_DIR_NAME="monet_style"
export ORIGINAL_DIR_NAME="monet_style_original"

# --- PROMPT CONFIGURATION ---
# IMPORTANT: Use a unique, rare token for your style
export INSTANCE_PROMPT="a painting in the style of monet"
export PRIOR_PROMPT="a photo of some landscape"

# --- EXPERIMENT: PAIRED REGULARIZED LORA ---
accelerate launch --num_processes=1 ./src/train_lora_reg.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --data_parent_dir=$DATA_PARENT_DIR \
  --monet_dir_name=$MONET_DIR_NAME \
  --original_dir_name=$ORIGINAL_DIR_NAME \
  --output_dir="./outputs/lora_paired_reg_monet" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --prior_prompt="$PRIOR_PROMPT" \
  --resolution=1024 \
  --train_batch_size=4 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --max_train_steps=1000 \
  --lora_rank=32 \
  --use_regularization \
  --lambda_reg=1.0

echo "Training complete. Model saved to ./outputs/lora_paired_reg_monet"