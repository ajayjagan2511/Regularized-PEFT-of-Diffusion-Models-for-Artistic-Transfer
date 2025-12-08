import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.image import StructuralSimilarityIndexMeasure, FrechetInceptionDistance
from huggingface_hub import snapshot_download
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL, UNet2DConditionModel
from PIL import Image
from tqdm import tqdm
import warnings
import random
import numpy as np
import copy
import gc
import json
import csv
from datetime import datetime

# Suppress warnings
# warnings.filterwarnings("ignore")

# ==========================================
#               CONFIGURATION
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. DEFINITIONS
TRIGGER_STR = "sks monet style"
GENERIC_STR = "a painting in the style of Monet"
FALLBACK_PROMPT = "sksmonet style, a photo of a landscape"
REPORT_OUTPUT_DIR = "/home/shivamsinghal/LTC/GEN/data/eval_results"

# 2. DATASETS
DATASET_CONFIGS = {
    "Validation": {
        "orig_dir": "/home/shivamsinghal/LTC/GEN/data/monet_style_original_val",
        "gt_dir":   "/home/shivamsinghal/LTC/GEN/data/monet_style_val",
        "output_root": "/home/shivamsinghal/LTC/GEN/data/eval_results/val"
    },
    "Test": {
        "orig_dir": "/home/shivamsinghal/LTC/GEN/data/monet_style_test",
        "gt_dir":   "/home/shivamsinghal/LTC/GEN/data/monet_style_original_test",
        "output_root": "/home/shivamsinghal/LTC/GEN/data/eval_results/test"
    }
}

# 3. MODELS TO EVALUATE
MODEL_CONFIGS = {
    "Base_SDXL": {
        "type": "base",
        "path": None 
    },
    "LoRA_Regularized_lr_1e-4_lambda_1_epoch_10": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-4_lambda_1.0_rank_16_reg/checkpoint-epoch-10/pytorch_lora_weights.bin"
    },
    "LoRA_Regularized_lr_1e-4_lambda_10_epoch_10": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-4_lambda_10.0_rank_16_reg/checkpoint-epoch-10/pytorch_lora_weights.bin"
    },
    "LoRA_Regularized_lr_1e-5_lambda_1_epoch_10": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-5_lambda_1.0_rank_16_reg/checkpoint-epoch-10/pytorch_lora_weights.bin"
    },
    "LoRA_Regularized_lr_1e-5_lambda_10_epoch_10": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-5_lambda_10.0_rank_16_reg/checkpoint-epoch-10/pytorch_lora_weights.bin"
    },
    "LoRA_Regularized_lr_1e-4_lambda_1_epoch_20": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-4_lambda_1.0_rank_16_reg/checkpoint-epoch-20/pytorch_lora_weights.bin"
    },
    "LoRA_Regularized_lr_1e-4_lambda_10_epoch_20": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-4_lambda_10.0_rank_16_reg/checkpoint-epoch-20/pytorch_lora_weights.bin"
    },
    "LoRA_Regularized_lr_1e-5_lambda_1_epoch_20": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-5_lambda_1.0_rank_16_reg/checkpoint-epoch-20/pytorch_lora_weights.bin"
    },
    "LoRA_Regularized_lr_1e-5_lambda_10_epoch_20": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-5_lambda_10.0_rank_16_reg/checkpoint-epoch-20/pytorch_lora_weights.bin"
    },
    "LoRA_No_Reg_lr_1e-5_epoch_10": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-5_lambda_0.0_rank_16_reg/checkpoint-epoch-10/pytorch_lora_weights.bin"
    },
    "LoRA_No_Reg_lr_1e-4_epoch_10": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-4_lambda_0.0_rank_16_reg/checkpoint-epoch-10/pytorch_lora_weights.bin"
    },
    "LoRA_No_Reg_lr_1e-5_epoch_20": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-5_lambda_0.0_rank_16_reg/checkpoint-epoch-20/pytorch_lora_weights.bin"
    },
    "LoRA_No_Reg_lr_1e-4_epoch_20": {
        "type": "lora",
        "repo_id": "batshiv/sdxl-vae-fp16-fix-regularized", 
        "filename": "lr_1e-4_lambda_0.0_rank_16_reg/checkpoint-epoch-20/pytorch_lora_weights.bin"
    },
    "FFT_Model_lr_1e-6_epoch_10": {
        "type": "fft",
        "path": "/home/shivamsinghal/LTC/GEN/outputs/FFT_lr_5e-6_baseline_batch_8/checkpoint-epoch-10" 
    },
    "FFT_Model_lr_5e-6_epoch_10": {
        "type": "fft",
        "path": "/home/shivamsinghal/LTC/GEN/outputs/FFT_lr_1e-6_baseline_batch_8/checkpoint-epoch-10" 
    },
    "FFT_Model_lr_1e-6_epoch_20": {
        "type": "fft",
        "path": "/home/shivamsinghal/LTC/GEN/outputs/FFT_lr_5e-6_baseline_batch_8/checkpoint-epoch-20" 
    },
    "FFT_Model_lr_5e-6_epoch_20": {
        "type": "fft",
        "path": "/home/shivamsinghal/LTC/GEN/outputs/FFT_lr_1e-6_baseline_batch_8/checkpoint-epoch-20" 
    },
    "FFT_Model_lr_1e-6_epoch_10_batch_4": {
        "type": "fft",
        "path": "/home/shivamsinghal/LTC/GEN/outputs/FFT_lr_5e-6_baseline_batch_4/checkpoint-epoch-10" 
    },
    "FFT_Model_lr_5e-6_epoch_10_batch_4": {
        "type": "fft",
        "path": "/home/shivamsinghal/LTC/GEN/outputs/FFT_lr_1e-6_baseline_batch_4/checkpoint-epoch-10" 
    },
    "FFT_Model_lr_1e-6_epoch_20_batch_4": {
        "type": "fft",
        "path": "/home/shivamsinghal/LTC/GEN/outputs/FFT_lr_5e-6_baseline_batch_4/checkpoint-epoch-20" 
    },
    "FFT_Model_lr_5e-6_epoch_20_batch_4": {
        "type": "fft",
        "path": "/home/shivamsinghal/LTC/GEN/outputs/FFT_lr_1e-6_baseline_batch_4/checkpoint-epoch-20" 
    }
}


# ==========================================
#             METRIC CLASSES
# ==========================================

class GramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(DEVICE).eval()
        self.indices = [0, 5, 10, 19, 28]
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def forward(self, pred_pil, gt_pil):
        pred = self.transform(pred_pil).unsqueeze(0).to(DEVICE)
        gt = self.transform(gt_pil).unsqueeze(0).to(DEVICE)
        loss = 0.0
        with torch.no_grad():
            for i, layer in enumerate(self.vgg):
                pred, gt = layer(pred), layer(gt)
                if i in self.indices:
                    b, c, h, w = pred.shape
                    p_gram = torch.mm(pred.view(c, h*w), pred.view(c, h*w).t()) / (c*h*w)
                    g_gram = torch.mm(gt.view(c, h*w), gt.view(c, h*w).t()) / (c*h*w)
                    loss += F.mse_loss(p_gram, g_gram)
        return loss.item()

class CLIPMetric(nn.Module):
    def __init__(self):
        super().__init__()
        # Use LAION model to avoid 'weights_only=True' security error with standard OpenAI model
        model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
        self.model = CLIPModel.from_pretrained(model_id, use_safetensors=True).to(DEVICE).eval()
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.cosine = nn.CosineSimilarity(dim=1)

    def forward(self, pred_pil, gt_pil):
        inputs = self.processor(images=[pred_pil, gt_pil], return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            embeds = self.model.get_image_features(**inputs)
        embeds = F.normalize(embeds, p=2, dim=1)
        return (1.0 - self.cosine(embeds[0].unsqueeze(0), embeds[1].unsqueeze(0))).item()

def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==========================================
#          MODEL LOADING LOGIC
# ==========================================

def get_lora_path(config):
    if config.get("path"):
        return config["path"]
    
    print(f"Downloading LoRA from {config['repo_id']}...")
    try:
        file_path = snapshot_download(
            repo_id=config["repo_id"],
            allow_patterns=[config["filename"]],
            local_dir="hf_artifacts",
            local_dir_use_symlinks=False
        )
        return os.path.join(file_path, config["filename"])
    except Exception as e:
        print(f"Error downloading LoRA: {e}")
        return None

def prepare_pipeline_for_model(base_pipe, model_name, config, original_unet):
    print(f"--- Preparing Pipeline for: {model_name} ({config['type']}) ---")
    base_pipe.unload_lora_weights()
    base_pipe.unet = original_unet 
    
    if config["type"] == "base":
        return base_pipe, False, True

    elif config["type"] == "lora":
        lora_path = get_lora_path(config)
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA from: {lora_path}")
            try:
                state_dict = torch.load(lora_path, map_location="cpu")
                new_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("base_model.model."):
                        k = k.replace("base_model.model.", "")
                    if not k.startswith("unet.") and not k.startswith("text_encoder."):
                        new_key = f"unet.{k}"
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                
                base_pipe.load_lora_weights(new_state_dict, adapter_name="active_adapter")
                base_pipe.set_adapters(["active_adapter"])
                print("✅ LoRA loaded successfully.")
                return base_pipe, True, True
            except Exception as e:
                print(f"❌ Failed to load LoRA: {e}")
                return base_pipe, True, False
        else:
            print(f"CRITICAL: LoRA path not found: {lora_path}")
            return base_pipe, True, False

    elif config["type"] == "fft":
        fft_path = config["path"]
        if not os.path.exists(fft_path):
            print(f"CRITICAL: FFT path not found: {fft_path}")
            return base_pipe, False, False
            
        print(f"Loading Fine-Tuned UNet from {fft_path}...")
        try:
            try:
                fft_unet = UNet2DConditionModel.from_pretrained(fft_path, subfolder="unet", torch_dtype=torch.float16, use_safetensors=True)
            except:
                try:
                    fft_unet = UNet2DConditionModel.from_pretrained(fft_path, subfolder="unet", torch_dtype=torch.float16, use_safetensors=False)
                except:
                    try:
                        fft_unet = UNet2DConditionModel.from_pretrained(fft_path, torch_dtype=torch.float16, use_safetensors=True)
                    except:
                        fft_unet = UNet2DConditionModel.from_pretrained(fft_path, torch_dtype=torch.float16, use_safetensors=False)

            fft_unet.to(DEVICE)
            base_pipe.unet = fft_unet
            return base_pipe, True, True
        except Exception as e:
            print(f"❌ Error loading FFT model: {e}")
            return base_pipe, True, False

    return base_pipe, False, False

# ==========================================
#          EVALUATION LOGIC
# ==========================================

def evaluate_dataset(pipeline, dataset_name, dataset_config, model_name, is_trained_model):
    orig_dir = dataset_config["orig_dir"]
    gt_dir = dataset_config["gt_dir"]
    out_dir = os.path.join(dataset_config["output_root"], model_name)
    os.makedirs(out_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(orig_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        return {}

    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    ssim_transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    fid = FrechetInceptionDistance(feature=64).to(DEVICE)
    fid_transform = transforms.Compose([transforms.Resize((299, 299)), transforms.PILToTensor()])
    gram_calc = GramLoss()
    clip_calc = CLIPMetric()

    scores = {"ssim": [], "gram": [], "clip": []}

    print(f"Generating {len(image_files)} images for {model_name} on {dataset_name}...")

    for filename in tqdm(image_files):
        try:
            orig_path = os.path.join(orig_dir, filename)
            gt_path = os.path.join(gt_dir, filename)
            txt_path = os.path.join(orig_dir, os.path.splitext(filename)[0] + ".txt")
            
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    raw_prompt = f.read().strip()
            else:
                raw_prompt = FALLBACK_PROMPT

            if is_trained_model:
                final_prompt = raw_prompt
            else:
                final_prompt = raw_prompt.replace(TRIGGER_STR, GENERIC_STR).replace("sks", "").strip()

            if len(final_prompt) > 300:
                final_prompt = final_prompt[:300]

            orig_pil = Image.open(orig_path).convert("RGB").resize((1024, 1024))
            gt_pil = Image.open(gt_path).convert("RGB").resize((1024, 1024))

            seed_everything(42)
            gen_pil = pipeline(
                prompt=final_prompt,
                image=orig_pil,
                strength=0.75,
                guidance_scale=7.5,
                num_inference_steps=30
            ).images[0]

            gen_pil.save(os.path.join(out_dir, filename))

            real_tensor = fid_transform(gt_pil).unsqueeze(0).to(DEVICE)
            fake_tensor = fid_transform(gen_pil).unsqueeze(0).to(DEVICE)
            fid.update(real_tensor, real=True)
            fid.update(fake_tensor, real=False)
            
            orig_tensor = ssim_transform(orig_pil).unsqueeze(0).to(DEVICE)
            gen_tensor = ssim_transform(gen_pil).unsqueeze(0).to(DEVICE)
            
            scores["ssim"].append(ssim(gen_tensor, orig_tensor).item())
            scores["gram"].append(gram_calc(gen_pil, gt_pil))
            scores["clip"].append(clip_calc(gen_pil, gt_pil))

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print("Computing FID...")
    try:
        fid_score = fid.compute().item()
    except:
        fid_score = 999.0

    return {
        "FID": fid_score,
        "SSIM": np.mean(scores["ssim"]),
        "Gram": np.mean(scores["gram"]),
        "CLIP": np.mean(scores["clip"])
    }

# ==========================================
#          SAVE HELPER
# ==========================================

def save_results(all_results, output_dir, timestamp):
    """Saves the current state of results to JSON and CSV."""
    # JSON
    json_path = os.path.join(output_dir, f"eval_report_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # CSV
    csv_path = os.path.join(output_dir, f"eval_report_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Model", "FID", "SSIM", "Gram", "CLIP"])
        
        for ds_name, models_res in all_results.items():
            for model_name, metrics in models_res.items():
                if not metrics: continue
                writer.writerow([
                    ds_name, model_name, 
                    f"{metrics['FID']:.4f}", f"{metrics['SSIM']:.4f}", 
                    f"{metrics['Gram']:.4f}", f"{metrics['CLIP']:.4f}"
                ])
    
    print(f"✅ Results updated in {output_dir}")

# ==========================================
#               MAIN LOOP
# ==========================================

def main():
    print("Loading Base SDXL Pipeline...")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    base_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True
    ).to(DEVICE)
    
    original_unet = base_pipe.unet
    
    # Generate Timestamp once for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    all_results = {}

    for ds_name, ds_config in DATASET_CONFIGS.items():
        print(f"\n\n{'='*60}\nEVALUATING DATASET: {ds_name}\n{'='*60}")
        all_results[ds_name] = {}
        
        for model_name, model_config in MODEL_CONFIGS.items():
            pipe, is_trained, success = prepare_pipeline_for_model(base_pipe, model_name, model_config, original_unet)
            
            if not success:
                print(f"⛔ SKIPPING {model_name} due to loading error.")
                continue

            metrics = evaluate_dataset(pipe, ds_name, ds_config, model_name, is_trained)
            all_results[ds_name][model_name] = metrics
            
            # --- SAVE PROGRESS ---
            save_results(all_results, REPORT_OUTPUT_DIR, timestamp)
            
            torch.cuda.empty_cache()
            gc.collect()

    print("\n\n" + "="*80)
    print("FINAL EVALUATION REPORT")
    print("="*80)
    print(f"{'Dataset':<10} | {'Model':<30} | {'FID (↓)':<10} | {'SSIM (↑)':<10} | {'Gram (↓)':<10} | {'CLIP (↓)':<10}")
    print("-" * 95)

    for ds_name, models_res in all_results.items():
        for model_name, metrics in models_res.items():
            if not metrics: continue
            print(f"{ds_name:<10} | {model_name:<30} | {metrics['FID']:<10.4f} | {metrics['SSIM']:<10.4f} | {metrics['Gram']:<10.4f} | {metrics['CLIP']:<10.4f}")

if __name__ == "__main__":
    main()