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
from datasets import load_dataset
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import is_xformers_available
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from dataset import PairedImageDataset

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--pretrained_vae_model_name_or_path", type=str, default="madebyollin/sdxl-vae-fp16-fix")
    parser.add_argument("--instance_data_dir", type=str, required=True, help="A folder containing the training data.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--instance_prompt", type=str, required=True, help="The prompt with identifier.")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--lora_rank", type=int, default=16)
    
    # --- OUR NOVEL ARGUMENTS ---
    parser.add_argument("--use_regularization", action="store_true", help="Enable dual-model regularization loss.")
    parser.add_argument("--lambda_reg", type=float, default=1.0, help="Weight for the regularization loss.")
    parser.add_argument("--prior_prompt_list", type=str, default="a cat,a dog,a car,a landscape", help="Comma-separated list of prior prompts for regularization.")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb" # or "tensorboard"
    )

    # Initialize wandb
    accelerator.init_trackers(
        project_name="cvpr-regularized-diffusion",
        config=vars(args)
    )

    set_seed(args.seed)

    # --- 1. LOAD MODELS ---
    # Load tokenizers
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    
    # Load text encoders
    text_encoder_one = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", variant="fp16")
    text_encoder_two = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2", variant="fp16")

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.pretrained_vae_model_name_or_path, torch_dtype=torch.float16)

    # Load UNet (this is the one we will train)
    unet_train = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant="fp16")
    
    # --- OUR MODIFICATION: Load the FROZEN UNet for regularization ---
    if args.use_regularization:
        unet_frozen = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet", variant="fp16")
        unet_frozen.requires_grad_(False)
        logger.info("Loaded frozen UNet for regularization.")
    # --- END MODIFICATION ---

    # Freeze VAE and text encoders
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Add LoRA adapters to the trainable UNet
    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    unet_train = get_peft_model(unet_train, unet_lora_config)
    
    if is_xformers_available():
        unet_train.enable_xformers_memory_efficient_attention()

    # --- 2. SETUP OPTIMIZER ---
    trainable_params = list(filter(lambda p: p.requires_grad, unet_train.parameters()))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
    )

    # --- 3. LOAD DATASET ---
    # Create a simple dataset from a folder of images
    dataset = load_dataset(
        "imagefolder",
        data_dir=args.instance_data_dir,
    )

    # Preprocessing (We have already preprocessed the images size 1024x1024)
    train_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples["image"]]
        examples["pixel_values"] = [train_transforms(image) for image in images]

        # Tokenize the prompt once
        input_ids_one, input_ids_two = tokenize_prompt(args.instance_prompt)
        
        # Get the actual tensors out of the batch-of-1 dimension
        prompt_ids_one = input_ids_one[0]
        prompt_ids_two = input_ids_two[0]

        # Create a list of tokenized prompts, one for each image
        num_images = len(images)
        examples["input_ids"] = [prompt_ids_one] * num_images
        examples["input_ids_2"] = [prompt_ids_two] * num_images
        
        return examples

    # Function to tokenize prompts for SDXL
    def tokenize_prompt(prompt):
        text_inputs = tokenizer_one(
            prompt,
            padding="max_length",
            max_length=tokenizer_one.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_inputs_2 = tokenizer_two(
            prompt,
            padding="max_length",
            max_length=tokenizer_two.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids, text_inputs_2.input_ids

    # train_dataset = dataset["train"].with_transform(preprocess_train)
    train_dataset = dataset["train"].map(
        function=preprocess_train,
        batched=True,
        remove_columns=["image"], # This is important!
    )
    train_dataset.set_format(type="torch", columns=["pixel_values", "input_ids", "input_ids_2"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )
    
    # --- PREPARE FOR REGULARIZATION ---
    if args.use_regularization:
        prior_prompts = args.prior_prompt_list.split(',')
        logger.info(f"Using prior prompts for regularization: {prior_prompts}")

    # --- 4. ACCELERATE EVERYTHING ---
    if args.use_regularization:
        unet_train, unet_frozen, optimizer, train_dataloader = accelerator.prepare(
            unet_train, unet_frozen, optimizer, train_dataloader
        )
    else:
        unet_train, optimizer, train_dataloader = accelerator.prepare(
            unet_train, optimizer, train_dataloader
        )

    # Move models to device
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else torch.float32
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    if args.use_regularization:
        unet_frozen.to(accelerator.device, dtype=weight_dtype)

    # --- 5. THE TRAINING LOOP ---
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    
    # Function to compute text embeddings
    def compute_text_embeddings(prompt, text_encoder_one, text_encoder_two):
        with torch.no_grad():
            token_ids_one, token_ids_two = tokenize_prompt(prompt)
            prompt_embeds_one = text_encoder_one(token_ids_one.to(accelerator.device), output_hidden_states=True)
            prompt_embeds_two = text_encoder_two(token_ids_two.to(accelerator.device), output_hidden_states=True)
            
            pooled_prompt_embeds = prompt_embeds_two[0]
            prompt_embeds = torch.cat([prompt_embeds_one.hidden_states[-2], prompt_embeds_two.hidden_states[-2]], dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    global_step = 0
    progress_bar = tqdm(range(global_step, args.max_train_steps), desc="Steps")

    for epoch in range(math.ceil(args.max_train_steps / len(train_dataloader))):
        unet_train.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet_train):
                # --- A. Fine-Tuning Loss (L_ft) ---
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                
                # Encode images to latents
                latents = vae.encode(pixel_values).latent_dist.sample() * vae.config.scaling_factor
                
                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Create a batch of prompts, not just one, to match the image batch size
                style_prompts = [args.instance_prompt] * bsz
                prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(style_prompts, text_encoder_one, text_encoder_two)
                add_text_embeds = pooled_prompt_embeds
                
                # Add time ids as required by SDXL
                add_time_ids = torch.tensor([[args.resolution, args.resolution, 0, 0, args.resolution, args.resolution]], device=accelerator.device)
                add_time_ids = add_time_ids.repeat(bsz, 1)

                unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": add_text_embeds}

                # Predict noise
                model_pred_ft = unet_train(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample
                loss_ft = F.mse_loss(model_pred_ft.float(), noise.float(), reduction="mean")
                
                total_loss = loss_ft

                # --- B. Regularization Loss (L_reg) ---
                if args.use_regularization:
                    # Choose a random prior prompt
                    prior_prompt = random.choice(prior_prompts)
                    
                    # Get text embeddings for the prior prompt
                    prior_prompt_embeds, prior_pooled_embeds = compute_text_embeddings(prior_prompt, text_encoder_one, text_encoder_two)
                    prior_add_text_embeds = prior_pooled_embeds
                    prior_unet_added_conditions = {"time_ids": add_time_ids, "text_embeds": prior_add_text_embeds}
                    
                    # Get predictions from both models
                    with torch.no_grad():
                        target_pred_frozen = unet_frozen(noisy_latents, timesteps, prior_prompt_embeds, added_cond_kwargs=prior_unet_added_conditions).sample
                    
                    model_pred_reg = unet_train(noisy_latents, timesteps, prior_prompt_embeds, added_cond_kwargs=prior_unet_added_conditions).sample
                    
                    loss_reg = F.mse_loss(model_pred_reg.float(), target_pred_frozen.float(), reduction="mean")
                    
                    # Combine losses
                    total_loss += args.lambda_reg * loss_reg
                    accelerator.log({"loss_reg": loss_reg.detach().item()}, step=global_step)

                accelerator.log({"loss_ft": loss_ft.detach().item(), "total_loss": total_loss.detach().item()}, step=global_step)

                accelerator.backward(total_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)
                
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if global_step % 500 == 0: # Save a checkpoint every 500 steps
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    
                    # Also save the LoRA weights separately for easy inference
                    unwrapped_unet = accelerator.unwrap_model(unet_train)
                    lora_state_dict = get_peft_model_state_dict(unwrapped_unet)
                    torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.bin"))


            if global_step >= args.max_train_steps:
                break
    
    accelerator.end_training()
    logger.info("Training finished.")

if __name__ == "__main__":
    main()