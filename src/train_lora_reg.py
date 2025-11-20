# src/train_lora_reg.py

import argparse
import math
import os
import csv
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# Import your custom dataset
from dataset import PairedImageDataset

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Regularized Paired LoRA Training for Style Transfer")
    
    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    
    # Data directories
    parser.add_argument("--data_parent_dir", type=str, required=True, help="Root data folder")
    parser.add_argument("--monet_dir_name", type=str, required=True, help="Train: Styled images folder")
    parser.add_argument("--original_dir_name", type=str, required=True, help="Train: Original images folder")
    
    # Validation directories
    parser.add_argument("--monet_val_dir_name", type=str, default=None, help="Val: Styled images folder")
    parser.add_argument("--original_val_dir_name", type=str, default=None, help="Val: Original images folder")
    
    parser.add_argument("--output_dir", type=str, required=True)

    # Prompts
    parser.add_argument("--instance_prompt", type=str, required=True, help="Student Trigger (e.g. 'sksmonet, a painting...')")
    parser.add_argument("--style_prompt", type=str, default="a painting in the style of Monet", help="Teacher Style Prompt (no trigger)")
    parser.add_argument("--prior_prompt", type=str, default="a photo of a landscape", help="Regularization Prompt (generic)")
    
    # Training Hyperparams
    parser.add_argument("--use_regularization", action="store_true")
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--save_every_epoch", type=int, default=1)
    
    parser.add_argument("--ema_beta", type=float, default=0.99, help="Decay factor for Exponential Moving Average of loss")

    # --- NEW ARGUMENT: SNR Gamma ---
    parser.add_argument("--snr_gamma", type=float, default=5.0, help="Gamma weighting for Min-SNR loss. Set to 0 to disable.")

    # Logging
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="cvpr-regularized-diffusion-paired")

    args = parser.parse_args()
    return args

# --- Helper: Compute Text Embeddings Wrapper ---
def compute_embeddings(prompt, tokenizers, text_encoders, device):
    tokenizer_one, tokenizer_two = tokenizers
    text_encoder_one, text_encoder_two = text_encoders
    
    with torch.no_grad():
        # Tokenizer 1
        text_input_1 = tokenizer_one(prompt, padding="max_length", max_length=tokenizer_one.model_max_length, truncation=True, return_tensors="pt")
        # Tokenizer 2
        text_input_2 = tokenizer_two(prompt, padding="max_length", max_length=tokenizer_two.model_max_length, truncation=True, return_tensors="pt")
        
        # Encoder 1
        prompt_embeds_1 = text_encoder_one(text_input_1.input_ids.to(device), output_hidden_states=True).hidden_states[-2]
        # Encoder 2
        enc_2_out = text_encoder_two(text_input_2.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_2 = enc_2_out.hidden_states[-2]
        pooled_embeds = enc_2_out.text_embeds

        # Concat
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    
    return prompt_embeds, pooled_embeds

# --- Helper: Loss Computation Logic (Updated with SNR) ---
def compute_loss_dict(
    unet_train, unet_frozen, noise_scheduler, 
    batch, vae, 
    embeddings_dict, 
    timesteps, noise, bsz, resolution,
    device, dtype, lambda_reg,
    snr_gamma=0.0 # New Argument
):
    # 1. Prepare Images
    monet_pixels = batch["monet_pixel_values"].to(dtype=dtype)
    orig_pixels = batch["original_pixel_values"].to(dtype=dtype)

    # 2. Encode to Latents
    latents_style = vae.encode(monet_pixels).latent_dist.sample() * vae.config.scaling_factor
    latents_orig = vae.encode(orig_pixels).latent_dist.sample() * vae.config.scaling_factor

    # 3. Add Noise
    noisy_latents_style = noise_scheduler.add_noise(latents_style, noise, timesteps)
    noisy_latents_orig = noise_scheduler.add_noise(latents_orig, noise, timesteps)

    # 4. Prepare Conditions
    add_time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]], device=device, dtype=dtype).repeat(bsz, 1)

    # --- CALCULATE SNR WEIGHTS (Min-SNR Gamma) ---
    if snr_gamma > 0:
        # Compute Signal-to-Noise Ratio
        alphas = noise_scheduler.alphas_cumprod.to(device)
        alpha_t = alphas[timesteps]
        
        # SNR = Signal / Noise = alpha_t / (1 - alpha_t)
        snr = alpha_t / (1 - alpha_t)
        
        # Min-SNR Weighting: min(SNR, gamma) / SNR
        # This down-weights high-noise steps (structure) to focus on low-noise steps (texture)
        mse_loss_weights = torch.stack([snr, snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
    else:
        mse_loss_weights = torch.ones_like(timesteps).float()

    # --- MAIN LOSS ---
    teacher_cond = {
        "time_ids": add_time_ids, 
        "text_embeds": embeddings_dict["teacher_style_pooled"].repeat(bsz, 1)
    }
    student_cond = {
        "time_ids": add_time_ids,
        "text_embeds": embeddings_dict["student_trigger_pooled"].repeat(bsz, 1)
    }

    with torch.no_grad():
        target_noise = unet_frozen(
            noisy_latents_style, timesteps, 
            embeddings_dict["teacher_style"].repeat(bsz, 1, 1), 
            added_cond_kwargs=teacher_cond
        ).sample

    pred_noise = unet_train(
        noisy_latents_orig, timesteps, 
        embeddings_dict["student_trigger"].repeat(bsz, 1, 1), 
        added_cond_kwargs=student_cond
    ).sample

    # Calculate Raw MSE (per element) -> Average Spatial -> Apply Weight -> Average Batch
    loss_ft_raw = F.mse_loss(pred_noise.float(), target_noise.float(), reduction="none")
    loss_ft_raw = loss_ft_raw.mean(dim=[1, 2, 3]) # Average over C, H, W
    loss_ft = (loss_ft_raw * mse_loss_weights).mean()
    
    # --- REGULARIZATION LOSS ---
    loss_reg = torch.tensor(0.0, device=device)
    if lambda_reg > 0:
        reg_cond = {
            "time_ids": add_time_ids,
            "text_embeds": embeddings_dict["reg_pooled"].repeat(bsz, 1)
        }
        
        with torch.no_grad():
            reg_target_noise = unet_frozen(
                noisy_latents_orig, timesteps, 
                embeddings_dict["reg_embeds"].repeat(bsz, 1, 1), 
                added_cond_kwargs=reg_cond
            ).sample
        
        # Same weighting strategy for regularization
        loss_reg_raw = F.mse_loss(pred_noise.float(), reg_target_noise.float(), reduction="none")
        loss_reg_raw = loss_reg_raw.mean(dim=[1, 2, 3])
        loss_reg = (loss_reg_raw * mse_loss_weights).mean()

    return loss_ft, loss_reg

def update_ema(old_val, new_val, beta):
    if old_val is None:
        return new_val
    return old_val * beta + new_val * (1 - beta)

def main():
    args = parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb"
    )
    
    run_name = args.wandb_run_name or f"ep{args.num_epochs}_lr{args.learning_rate}_reg{args.lambda_reg}"
    accelerator.init_trackers(
        project_name=args.wandb_project,
        config=vars(args),
        init_kwargs={"wandb": {"name": run_name}}
    )
    set_seed(args.seed)

    # --- Load Models ---
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_enc_1 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
    text_enc_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant="fp16")
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, torch_dtype=torch.float16)
    
    unet_train = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant="fp16")
    unet_frozen = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant="fp16")
    
    vae.requires_grad_(False)
    text_enc_1.requires_grad_(False)
    text_enc_2.requires_grad_(False)
    unet_frozen.requires_grad_(False)
    
    unet_lora_config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=["to_q", "to_k", "to_v", "to_out.0"])
    unet_train = get_peft_model(unet_train, unet_lora_config)
    unet_train.print_trainable_parameters()

    optimizer = torch.optim.AdamW(unet_train.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer)

    # --- Data Loading ---
    train_tfms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = PairedImageDataset(
        monet_dir=os.path.join(args.data_parent_dir, args.monet_dir_name),
        original_dir=os.path.join(args.data_parent_dir, args.original_dir_name),
        transform=train_tfms
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    val_loader = None
    if args.monet_val_dir_name and args.original_val_dir_name:
        val_dataset = PairedImageDataset(
            monet_dir=os.path.join(args.data_parent_dir, args.monet_val_dir_name),
            original_dir=os.path.join(args.data_parent_dir, args.original_val_dir_name),
            transform=train_tfms
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=2)
        logger.info(f"Validation set loaded: {len(val_dataset)} pairs")

    # --- Prepare ---
    unet_train, optimizer, train_loader, lr_scheduler = accelerator.prepare(unet_train, optimizer, train_loader, lr_scheduler)
    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=dtype)
    text_enc_1.to(accelerator.device, dtype=dtype)
    text_enc_2.to(accelerator.device, dtype=dtype)
    unet_frozen.to(accelerator.device, dtype=dtype)

    # --- Pre-compute Embeddings ---
    tokenizers = (tokenizer_1, tokenizer_2)
    text_encoders = (text_enc_1, text_enc_2)
    
    logger.info("Pre-computing text embeddings...")
    s_emb, s_pool = compute_embeddings(args.instance_prompt, tokenizers, text_encoders, accelerator.device)
    t_emb, t_pool = compute_embeddings(args.style_prompt, tokenizers, text_encoders, accelerator.device)
    r_emb, r_pool = compute_embeddings(args.prior_prompt, tokenizers, text_encoders, accelerator.device)
    
    embeddings_dict = {
        "student_trigger": s_emb, "student_trigger_pooled": s_pool,
        "teacher_style": t_emb, "teacher_style_pooled": t_pool,
        "reg_embeds": r_emb, "reg_pooled": r_pool
    }

    # --- Logging CSV Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = open(os.path.join(args.output_dir, "metrics.csv"), "w", newline="")
    writer = csv.writer(metrics_file)
    writer.writerow([
        "epoch", "timestamp", 
        "train_loss_total", "train_loss_ft", "train_loss_reg", 
        "train_loss_total_ema", "train_loss_ft_ema", "train_loss_reg_ema",
        "val_loss_total", "val_loss_ft", "val_loss_reg"
    ])

    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # --- EMA State Variables ---
    ema_losses = {"total": None, "ft": None, "reg": None}

    logger.info(f"Starting Training... SNR Gamma: {args.snr_gamma}")

    # --- Training Loop ---
    for epoch in range(1, args.num_epochs + 1):
        unet_train.train()
        epoch_loss_ft = 0.0
        epoch_loss_reg = 0.0
        epoch_batches = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            with accelerator.accumulate(unet_train):
                bsz = batch["monet_pixel_values"].shape[0]
                noise = torch.randn((bsz, 4, args.resolution//8, args.resolution//8), device=accelerator.device, dtype=dtype)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()

                loss_ft, loss_reg = compute_loss_dict(
                    unet_train, unet_frozen, noise_scheduler,
                    batch, vae, embeddings_dict, timesteps, noise, bsz, args.resolution,
                    accelerator.device, dtype, args.lambda_reg,
                    snr_gamma=args.snr_gamma # <--- PASSED HERE
                )
                
                total_loss = loss_ft + (args.lambda_reg * loss_reg)
                
                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet_train.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Update Running Epoch Sums
                epoch_loss_ft += loss_ft.item()
                epoch_loss_reg += loss_reg.item()
                epoch_batches += 1
                
                # --- UPDATE EMA ---
                ema_losses["total"] = update_ema(ema_losses["total"], total_loss.item(), args.ema_beta)
                ema_losses["ft"] = update_ema(ema_losses["ft"], loss_ft.item(), args.ema_beta)
                ema_losses["reg"] = update_ema(ema_losses["reg"], loss_reg.item(), args.ema_beta)

                # Log Step (Including EMA)
                accelerator.log({
                    "step_loss": total_loss.item(),
                    "step_loss_ema": ema_losses["total"]
                }, step=(epoch-1)*len(train_loader)+step)

        avg_train_ft = epoch_loss_ft / epoch_batches
        avg_train_reg = epoch_loss_reg / epoch_batches
        avg_train_total = avg_train_ft + (args.lambda_reg * avg_train_reg)

        # --- Validation Loop ---
        avg_val_ft = 0.0
        avg_val_reg = 0.0
        avg_val_total = 0.0
        
        if val_loader:
            unet_train.eval()
            val_loss_ft = 0.0
            val_loss_reg = 0.0
            val_batches = 0
            
            for v_batch in tqdm(val_loader, desc="Validation"):
                with torch.no_grad():
                    bsz = v_batch["monet_pixel_values"].shape[0]
                    noise = torch.randn((bsz, 4, args.resolution//8, args.resolution//8), device=accelerator.device, dtype=dtype)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
                    
                    v_loss_ft, v_loss_reg = compute_loss_dict(
                        unet_train, unet_frozen, noise_scheduler,
                        v_batch, vae, embeddings_dict, timesteps, noise, bsz, args.resolution,
                        accelerator.device, dtype, args.lambda_reg,
                        snr_gamma=args.snr_gamma # <--- PASSED HERE
                    )
                    val_loss_ft += v_loss_ft.item()
                    val_loss_reg += v_loss_reg.item()
                    val_batches += 1
            
            avg_val_ft = val_loss_ft / val_batches
            avg_val_reg = val_loss_reg / val_batches
            avg_val_total = avg_val_ft + (args.lambda_reg * avg_val_reg)

        # --- End of Epoch Logging ---
        logger.info(f"EPOCH {epoch} Results:")
        logger.info(f"Train Loss: {avg_train_total:.4f} (EMA: {ema_losses['total']:.4f})")
        
        accelerator.log({
            "epoch": epoch,
            "train/total_loss": avg_train_total,
            "train/ft_loss": avg_train_ft,
            "train/reg_loss": avg_train_reg,
            "train/total_loss_ema": ema_losses["total"],
            "train/ft_loss_ema": ema_losses["ft"],
            "train/reg_loss_ema": ema_losses["reg"],
            "val/total_loss": avg_val_total,
            "val/ft_loss": avg_val_ft,
            "val/reg_loss": avg_val_reg,
        })
        
        writer.writerow([
            epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            avg_train_total, avg_train_ft, avg_train_reg,
            ema_losses["total"], ema_losses["ft"], ema_losses["reg"],
            avg_val_total, avg_val_ft, avg_val_reg
        ])
        metrics_file.flush()

        if epoch % args.save_every_epoch == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(save_path, exist_ok=True)
            unwrapped_unet = accelerator.unwrap_model(unet_train)
            lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
            torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.bin"))
            logger.info(f"Saved checkpoint to {save_path}")

    metrics_file.close()
    accelerator.end_training()

if __name__ == "__main__":
    main()