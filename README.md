# Regularized Parameter-Efficient Fine-Tuning (Reg-PEFT) for SDXL

[![SDXL](https://img.shields.io/badge/Model-SDXL%201.0-blue)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

This repository contains the official implementation for the paper **"Regularized Parameter-Efficient Fine-Tuning of Diffusion Models for Artistic Transfer"**.

We propose a Dual-Model "Teacher-Student" framework for fine-tuning Stable Diffusion XL (SDXL) on artistic styles. By utilizing a frozen teacher anchor and a **Signal-to-Noise Ratio (SNR) weighted regularization loss**, we achieve high-fidelity style transfer while preventing catastrophic forgetting of structural priors.

---

## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/regdiffusion-peft.git
cd regdiffusion-peft
```

Install dependencies (this project relies on `diffusers`, `accelerate`, `peft`, and `torchmetrics` for evaluation):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # Adjust for your CUDA version
pip install -U diffusers accelerate transformers peft wandb bitsandbytes torchmetrics hf_transfer huggingface_hub
```

Configure Accelerate (Mixed Precision FP16 is recommended):

```bash
accelerate config
```

---

## 📊 Data Preparation

The training pipeline requires **paired data** (Content Image ↔ Style Image).

Structure your data directory as follows:

```text
/path/to/data/
├── monet_style/           # Stylized images (Target)
│   ├── image_01.jpg
│   └── ...
└── monet_style_original/  # Content/Structure images (Source)
    ├── image_01.jpg       # Must match filenames in style folder
    └── ...
```

Pre-process images using the provided script to center-crop and resize images to **1024×1024** (standard SDXL resolution):

```bash
# Edit src/preprocess_images.py to point to your directories first
python src/preprocess_images.py
```

---

## 🚀 Replicating Experiments

We provide shell scripts to reproduce the experiments from the paper (Standard LoRA, Regularized LoRA, and Full Fine-Tuning).

### 1. Regularized LoRA (Proposed Method) & Standard LoRA

The script `scripts/run_training.sh` is set up to run a hyperparameter sweep over learning rates and regularization strengths (`λ`).

- **Standard LoRA Baseline:** `λ = 0.0`  
- **Regularized LoRA (Ours):** `λ = 10.0` (Best reported configuration)

To run the experiments:

1. Open `scripts/run_training.sh`.
2. Update the `DATA_PARENT_DIR` variable to point to your data folder.
3. Execute the script:

```bash
chmod +x scripts/run_training.sh
./scripts/run_training.sh
```

### 2. Full Fine-Tuning (Baseline)

To compare against Full Fine-Tuning (FFT), use the provided FFT script. Note that FFT requires significantly more VRAM and uses lower learning rates.

1. Open `scripts/run_fft.sh`.
2. Update `DATA_PARENT_DIR`.
3. Execute:

```bash
chmod +x scripts/run_fft.sh
./scripts/run_fft.sh
```

---

## 📈 Evaluation & Benchmarking

We provide a comprehensive evaluation script (`src/evaluate.py`) that calculates:

- **FID** – Realism  
- **SSIM** – Structural Preservation  
- **Gram Matrix Loss** – Style Consistency  
- **CLIP Score** – Semantic Alignment  

The evaluation pipeline uses `StableDiffusionXLImg2ImgPipeline` to assess how well the model applies style to a validation content image.

### 1. Configuration

Before running, edit the **CONFIGURATION** section at the top of `src/evaluate.py`:

```python
# src/evaluate.py

# 1. Set your local data paths
DATASET_CONFIGS = {
    "Validation": {
        "orig_dir": "/path/to/your/monet_style_original_val",
        "gt_dir":   "/path/to/your/monet_style_val",
        "output_root": "./eval_results/val"
    },
    # Add more dataset splits if needed...
}

# 2. Define the models you want to benchmark
MODEL_CONFIGS = {
    "Base_SDXL": {
        "type": "base",
        "path": None
    },

    # For LoRA (can load from HuggingFace Hub or local path)
    "My_Reg_LoRA": {
        "type": "lora",
        "path": "outputs/lr_1e-4_lambda_10_reg/checkpoint-epoch-20/pytorch_lora_weights.bin",
        # OR use "repo_id" and "filename" for HF Hub download, e.g.:
        # "repo_id": "username/my-reg-lora",
        # "filename": "pytorch_lora_weights.bin",
    },

    # For Full Fine-Tuning
    "FFT_Model": {
        "type": "fft",
        "path": "outputs/FFT_run/checkpoint-epoch-20"
    }
}
```

Make sure the `MODEL_CONFIGS` dictionary matches:

- The filesystem layout produced by your training scripts, **and/or**
- Any published checkpoints on the HuggingFace Hub if you use `repo_id` and `filename`.

### 2. Run Evaluation

Once configured, run the script:

```bash
python src/evaluate.py
```

### 3. Output

The script will generate:

- **Generated Images** – Saved in the `output_root` directory for each dataset configuration, for visual inspection.
- **CSV Report** – A `eval_report_TIMESTAMP.csv` file containing FID, SSIM, Gram, and CLIP scores for all models.
- **JSON Report** – A detailed JSON file with full metric dumps for downstream analysis.

---

## 💻 Inference (Quick Start)

You can load the trained LoRA adapters using `diffusers` in a standalone script:

```python
import torch
from diffusers import DiffusionPipeline

# 1. Load Base SDXL
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
).to("cuda")

# 2. Load Your Trained Adapter
adapter_path = "./outputs/lr_1e-4_lambda_10_rank_16_reg/checkpoint-epoch-20/pytorch_lora_weights.bin"
pipe.load_lora_weights(adapter_path, adapter_name="monet")

# 3. Fuse & Scale (Optional but recommended)
pipe.set_adapters(["monet"], adapter_weights=[1.0])

# 4. Generate
prompt = "sksmonet, a painting in the style of Monet, a house by the river"
image = pipe(prompt=prompt, num_inference_steps=30).images[0]
image.save("result.png")
```

---

## 📝 Hyperparameters

Based on the ablation study in the paper, the optimal configurations are:

| Method                  | LR   | Rank | Lambda (`λ`) | SNR Gamma |
|-------------------------|------|------|--------------|-----------|
| Standard LoRA           | 1e-4 | 16   | 0.0          | 5.0       |
| Regularized LoRA (Ours) | 1e-4 | 16   | 10.0         | 5.0       |

> **Note:** We utilize Min-SNR weighting (`--snr_gamma=5.0`) in all LoRA experiments to stabilize training.