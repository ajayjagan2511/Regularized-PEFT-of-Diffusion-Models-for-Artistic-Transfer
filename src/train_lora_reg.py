# src/train_lora_reg.py

import argparse
import itertools
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import is_xformers_available
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# --- NEW: Import our custom dataset ---
from dataset import PairedImageDataset

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Paired regularization training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    
    # --- MODIFIED: Arguments for paired dataset ---
    parser.add_argument("--data_parent_dir", type=str, required=True, help="Parent folder containing the image directories.")
    parser.add_argument("--monet_dir_name", type=str, required=True, help="Name of the folder with Monet images.")
    parser.add_argument("--original_dir_name", type=str, required=True, help="Name of the folder with original photos.")
    
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True, help="The prompt with the unique identifier for the style.")
    
    # --- MODIFIED: Simplified regularization prompt ---
    parser.add_argument("--prior_prompt", type=str, default="a photo of a landscape", help="The generic prompt for the original images.")
    parser.add_argument("--use_regularization", action="store_true", help="Enable paired-image regularization loss.")
    parser.add_argument("--lambda_reg", type=float, default=1.0, help="Weight for the regularization loss.")

    # --- Standard Training Arguments ---
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--lora_rank", type=int, default=16)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb"
    )

    accelerator.init_trackers(project_name="cvpr-regularized-diffusion-paired", config=vars(args))
    set_seed(args.seed)

    # --- 1. LOAD MODELS (Same as before) ---
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant="fp16")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, torch_dtype=torch.float16)
    unet_train = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant="fp16")

    if args.use_regularization:
        unet_frozen = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant="fp16")
        unet_frozen.requires_grad_(False)
        logger.info("Loaded frozen UNet for regularization.")
    else:
        unet_frozen = None

    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    
    unet_lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet_train = get_peft_model(unet_train, unet_lora_config)

    # --- 2. SETUP OPTIMIZER (Same as before) ---
    optimizer = torch.optim.AdamW(unet_train.parameters(), lr=args.learning_rate)

    # --- 3. NEW: LOAD PAIRED DATASET ---
    train_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    monet_data_path = os.path.join(args.data_parent_dir, args.monet_dir_name)
    original_data_path = os.path.join(args.data_parent_dir, args.original_dir_name)

    train_dataset = PairedImageDataset(
        monet_dir=monet_data_path,
        original_dir=original_data_path,
        transform=train_transforms
    )
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    logger.info(f"Loaded {len(train_dataset)} paired images.")

    # --- 4. ACCELERATE EVERYTHING ---
    if args.use_regularization:
        unet_train, unet_frozen, optimizer, train_dataloader = accelerator.prepare(unet_train, unet_frozen, optimizer, train_dataloader)
    else:
        unet_train, optimizer, train_dataloader = accelerator.prepare(unet_train, optimizer, train_dataloader)

    weight_dtype = torch.float16
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    if unet_frozen:
        unet_frozen.to(accelerator.device, dtype=weight_dtype)

    # --- 5. REWRITTEN TRAINING LOOP ---
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    def compute_text_embeddings(prompt, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two):
        with torch.no_grad():
            token_ids_one = tokenizer_one(prompt, padding="max_length", max_length=tokenizer_one.model_max_length, truncation=True, return_tensors="pt").input_ids
            token_ids_two = tokenizer_two(prompt, padding="max_length", max_length=tokenizer_two.model_max_length, truncation=True, return_tensors="pt").input_ids
            prompt_embeds_one = text_encoder_one(token_ids_one.to(accelerator.device), output_hidden_states=True)
            prompt_embeds_two = text_encoder_two(token_ids_two.to(accelerator.device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds_two[0]
            prompt_embeds = torch.cat([prompt_embeds_one.hidden_states[-2], prompt_embeds_two.hidden_states[-2]], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    # Pre-compute text embeddings once
    style_prompt_embeds, style_pooled_embeds = compute_text_embeddings(args.instance_prompt, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two)
    if args.use_regularization:
        prior_prompt_embeds, prior_pooled_embeds = compute_text_embeddings(args.prior_prompt, tokenizer_one, tokenizer_two, text_encoder_one, text_encoder_two)

    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), desc="Steps")

    for epoch in range(math.ceil(args.max_train_steps / len(train_dataloader))):
        unet_train.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet_train):
                # --- Get Paired Data ---
                monet_pixel_values = batch["monet_pixel_values"].to(dtype=weight_dtype)
                original_pixel_values = batch["original_pixel_values"].to(dtype=weight_dtype)

                # --- VAE Encoding ---
                latents_monet = vae.encode(monet_pixel_values).latent_dist.sample() * vae.config.scaling_factor
                latents_original = vae.encode(original_pixel_values).latent_dist.sample() * vae.config.scaling_factor

                # --- Shared Noise and Timesteps ---
                noise = torch.randn_like(latents_monet)
                bsz = latents_monet.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents_monet.device).long()
                
                noisy_latents_monet = noise_scheduler.add_noise(latents_monet, noise, timesteps)
                noisy_latents_original = noise_scheduler.add_noise(latents_original, noise, timesteps)

                # --- Shared Added Conditions ---
                add_time_ids = torch.tensor([[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]], device=accelerator.device, dtype=weight_dtype)
                add_time_ids = add_time_ids.repeat(bsz, 1)

                # --- A. Fine-Tuning Loss (L_ft) on Monet Image ---
                style_unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": style_pooled_embeds.repeat(bsz, 1)}
                model_pred_ft = unet_train(noisy_latents_monet, timesteps, style_prompt_embeds.repeat(bsz, 1, 1), added_cond_kwargs=style_unet_added_conditions).sample
                loss_ft = F.mse_loss(model_pred_ft.float(), noise.float(), reduction="mean")
                total_loss = loss_ft

                # --- B. Regularization Loss (L_reg) on Original Image ---
                if args.use_regularization:
                    prior_unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": prior_pooled_embeds.repeat(bsz, 1)}
                    
                    with torch.no_grad():
                        target_pred_frozen = unet_frozen(noisy_latents_original, timesteps, prior_prompt_embeds.repeat(bsz, 1, 1), added_cond_kwargs=prior_unet_added_conditions).sample
                    
                    model_pred_reg = unet_train(noisy_latents_original, timesteps, prior_prompt_embeds.repeat(bsz, 1, 1), added_cond_kwargs=prior_unet_added_conditions).sample
                    loss_reg = F.mse_loss(model_pred_reg.float(), target_pred_frozen.float(), reduction="mean")
                    total_loss += args.lambda_reg * loss_reg
                    accelerator.log({"loss_reg": loss_reg.detach().item()}, step=global_step)

                accelerator.log({"loss_ft": loss_ft.detach().item(), "total_loss": total_loss.detach().item()}, step=global_step)
                accelerator.backward(total_loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet_train.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % 500 == 0 and accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    unwrapped_unet = accelerator.unwrap_model(unet_train)
                    lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                    torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.bin"))
            
            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.end_training()
    logger.info("Training finished.")

if __name__ == "__main__":
    main()