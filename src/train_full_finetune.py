# src/train_full_finetune.py

import argparse
import os
import csv
import math
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

# Import your custom dataset
from dataset import PairedImageDataset

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Full Fine-Tuning (FFT) for SDXL")
    
    # Model paths
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    
    # Data directories
    parser.add_argument("--data_parent_dir", type=str, required=True)
    parser.add_argument("--monet_dir_name", type=str, required=True)
    # We keep the original dir arg to reuse the dataset class, though strictly not needed for pure FFT logic
    parser.add_argument("--original_dir_name", type=str, required=True) 
    
    # Validation directories
    parser.add_argument("--monet_val_dir_name", type=str, default=None)
    parser.add_argument("--original_val_dir_name", type=str, default=None)
    
    parser.add_argument("--output_dir", type=str, required=True)

    # Prompts
    parser.add_argument("--instance_prompt", type=str, required=True, help="Training prompt (e.g. 'sksmonet, a painting...')")
    
    # Training Hyperparams
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-6) # Lower LR for FFT
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--save_every_epoch", type=int, default=1)
    parser.add_argument("--snr_gamma", type=float, default=0.0)

    # Logging
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="cvpr-full-finetune")

    args = parser.parse_args()
    return args

def compute_embeddings(prompt, tokenizers, text_encoders, device):
    tokenizer_one, tokenizer_two = tokenizers
    text_encoder_one, text_encoder_two = text_encoders
    
    with torch.no_grad():
        text_input_1 = tokenizer_one(prompt, padding="max_length", max_length=tokenizer_one.model_max_length, truncation=True, return_tensors="pt")
        text_input_2 = tokenizer_two(prompt, padding="max_length", max_length=tokenizer_two.model_max_length, truncation=True, return_tensors="pt")
        
        prompt_embeds_1 = text_encoder_one(text_input_1.input_ids.to(device), output_hidden_states=True).hidden_states[-2]
        enc_2_out = text_encoder_two(text_input_2.input_ids.to(device), output_hidden_states=True)
        prompt_embeds_2 = enc_2_out.hidden_states[-2]
        pooled_embeds = enc_2_out.text_embeds

        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=-1)
    
    return prompt_embeds, pooled_embeds

def main():
    args = parse_args()
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb"
    )
    
    run_name = args.wandb_run_name or f"FFT_ep{args.num_epochs}_lr{args.learning_rate}"
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
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Load UNet for Full Fine-Tuning
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant="fp16")

    # Freeze VAE and Text Encoders
    vae.requires_grad_(False)
    text_enc_1.requires_grad_(False)
    text_enc_2.requires_grad_(False)
    
    # Enable Gradient Checkpointing (CRITICAL for FFT on SDXL)
    unet.enable_gradient_checkpointing()
    unet.train()

    # Optimizer for full UNet
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer)

    # --- Data Loading ---
    train_tfms = transforms.Compose([
        transforms.Resize(args.resolution),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Reuse PairedDataset but we will only use the 'monet' keys
    train_dataset = PairedImageDataset(
        monet_dir=os.path.join(args.data_parent_dir, args.monet_dir_name),
        original_dir=os.path.join(args.data_parent_dir, args.original_dir_name), # Loaded but unused
        transform=train_tfms
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    val_loader = None
    if args.monet_val_dir_name:
        val_dataset = PairedImageDataset(
            monet_dir=os.path.join(args.data_parent_dir, args.monet_val_dir_name),
            original_dir=os.path.join(args.data_parent_dir, args.original_val_dir_name),
            transform=train_tfms
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=2)

    # --- Prepare ---
    unet, optimizer, train_loader, lr_scheduler = accelerator.prepare(unet, optimizer, train_loader, lr_scheduler)
    if val_loader:
        val_loader = accelerator.prepare(val_loader)

    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=dtype)
    text_enc_1.to(accelerator.device, dtype=dtype)
    text_enc_2.to(accelerator.device, dtype=dtype)

    # --- Pre-compute Embeddings ---
    tokenizers = (tokenizer_1, tokenizer_2)
    text_encoders = (text_enc_1, text_enc_2)
    
    logger.info("Pre-computing text embeddings...")
    # For FFT Style Transfer, we only train on the Style Prompt
    instance_emb, instance_pool = compute_embeddings(args.instance_prompt, tokenizers, text_encoders, accelerator.device)

    # --- Logging Setup ---
    os.makedirs(args.output_dir, exist_ok=True)
    metrics_file = open(os.path.join(args.output_dir, "metrics.csv"), "w", newline="")
    writer = csv.writer(metrics_file)
    writer.writerow(["epoch", "timestamp", "train_loss", "val_loss"])

    logger.info("Starting Full Fine-Tuning...")

    for epoch in range(1, args.num_epochs + 1):
        unet.train()
        epoch_loss = 0.0
        epoch_batches = 0
        
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Train")):
            with accelerator.accumulate(unet):
                # 1. Latents
                monet_pixels = batch["monet_pixel_values"].to(dtype=dtype)
                bsz = monet_pixels.shape[0]
                
                latents = vae.encode(monet_pixels).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 2. Conditioning
                add_time_ids = torch.tensor([[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]], device=accelerator.device, dtype=dtype).repeat(bsz, 1)
                
                added_cond_kwargs = {
                    "time_ids": add_time_ids,
                    "text_embeds": instance_pool.repeat(bsz, 1)
                }
                prompt_embeds = instance_emb.repeat(bsz, 1, 1)

                # 3. Forward
                model_pred = unet(
                    noisy_latents, timesteps, 
                    prompt_embeds, 
                    added_cond_kwargs=added_cond_kwargs
                ).sample

                # 4. Loss
                if args.snr_gamma > 0:
                    alphas = noise_scheduler.alphas_cumprod.to(accelerator.device)
                    alpha_t = alphas[timesteps]
                    snr = alpha_t / (1 - alpha_t)
                    mse_loss_weights = torch.stack([snr, args.snr_gamma * torch.ones_like(snr)], dim=1).min(dim=1)[0] / snr
                    loss_raw = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                    loss_raw = loss_raw.mean(dim=[1, 2, 3])
                    loss = (loss_raw * mse_loss_weights).mean()
                else:
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                epoch_batches += 1
                
                accelerator.log({"step_loss": loss.item()}, step=(epoch-1)*len(train_loader)+step)

        avg_train_loss = epoch_loss / epoch_batches

        # --- Validation ---
        avg_val_loss = 0.0
        if val_loader:
            unet.eval()
            val_loss_sum = 0.0
            val_batches = 0
            
            for v_batch in tqdm(val_loader, desc="Validation"):
                with torch.no_grad():
                    monet_pixels = v_batch["monet_pixel_values"].to(dtype=dtype)
                    bsz = monet_pixels.shape[0]
                    
                    latents = vae.encode(monet_pixels).latent_dist.sample() * vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=accelerator.device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                    
                    add_time_ids = torch.tensor([[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]], device=accelerator.device, dtype=dtype).repeat(bsz, 1)
                    added_cond_kwargs = {"time_ids": add_time_ids, "text_embeds": instance_pool.repeat(bsz, 1)}
                    prompt_embeds = instance_emb.repeat(bsz, 1, 1)

                    model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample
                    
                    v_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    val_loss_sum += v_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss_sum / val_batches

        # --- Logging ---
        logger.info(f"EPOCH {epoch}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        accelerator.log({
            "epoch": epoch,
            "train/loss": avg_train_loss,
            "val/loss": avg_val_loss
        })
        
        writer.writerow([epoch, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), avg_train_loss, avg_val_loss])
        metrics_file.flush()

        # Save
        if epoch % args.save_every_epoch == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
            os.makedirs(save_path, exist_ok=True)
            # For FFT, we save the whole UNet
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(os.path.join(save_path, "unet"))
            logger.info(f"Saved Full UNet to {save_path}")

    metrics_file.close()
    accelerator.end_training()

if __name__ == "__main__":
    main()