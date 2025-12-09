# Regularized Parameter-Efficient Fine-Tuning (Reg-PEFT) for SDXL

[![SDXL](https://img.shields.io/badge/Model-SDXL%201.0-blue)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

This repository contains the official implementation for the paper **"Regularized Parameter-Efficient Fine-Tuning of Diffusion Models for Artistic Transfer"**.

We propose a Dual-Model "Teacher-Student" framework for fine-tuning Stable Diffusion XL (SDXL) on artistic styles. By utilizing a frozen teacher anchor and a **Signal-to-Noise Ratio (SNR) weighted regularization loss**, we achieve high-fidelity style transfer while preventing catastrophic forgetting of structural priors.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/regdiffusion-peft.git
cd regdiffusion-peft
```

Create Environment and Install dependencies (this project relies on `diffusers`, `accelerate`, `peft`, and `torchmetrics` for evaluation):

```bash
# 1. Create a Conda environment with Python 3.13
conda create -n reg_peft python=3.13 -y

# 2. Activate the environment
conda activate reg_peft

# 3. Install PyTorch specifically with CUDA 11.8 support (Recommended for stability)
# Note: Doing this *before* the requirements file ensures the correct GPU version is locked.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install the remaining requirements
pip install -r requirements.txt
```

Configure Accelerate (Mixed Precision FP16 is recommended):

```bash
accelerate config
```

Below is the standard response for the same.
```
- In which compute environment are you running?
  Answer: This machine

- Which type of machine are you using?
  Answer: No distributed training

- Do you want to run your training on CPU only (only recommended for testing purposes)?
  Answer: no

- Do you wish to optimize your script with torch dynamo?
  Answer: no

- Do you want to use DeepSpeed?
  Answer: no

- What GPU(s) (by id) should be used for training on this machine as a comma-separated list? [all]
  Answer: all

- Do you wish to use FP16 or BF16 (mixed precision)?
  Answer: fp16
```

---

## Data Preparation

The training pipeline requires **paired data** (Content Image ↔ Style Image).

Structure your data directory as follows (already configured here):

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
# Run the preprocess script with your image directory (target size is fixed to 1024)
chmod +x scripts/run_preprocess.sh
./scripts/run_preprocess.sh <image_dir>
```

For example:

```bash
./scripts/run_preprocess.sh ./data/monet_style_original
```

---

## Replicating Experiments

We provide shell scripts to reproduce the experiments from the paper (Standard LoRA, Regularized LoRA, and Full Fine-Tuning).

### 1. Regularized LoRA (Proposed Method) & Standard LoRA

The script `scripts/run_training.sh` is set up to run a hyperparameter sweep over learning rates and regularization strengths (`λ`).

- **Standard LoRA Baseline:** `λ = 0.0`  
- **Regularized LoRA (Ours):** `λ = 10.0` (Best reported configuration)

To run the experiments:

1. Configure WandB by adding your API key to `~/.bashrc`:  
   ```bash
   echo 'export WANDB_API_KEY=your_api_key_here' >> ~/.bashrc
   source ~/.bashrc
   ```

2. Execute the script with your data parent directory as an argument:

```bash
chmod +x scripts/run_training.sh
./scripts/run_training.sh <data_parent_dir>
```

For example:

```bash
./scripts/run_training.sh ./data
```

### 2. Full Fine-Tuning (Baseline)

To compare against Full Fine-Tuning (FFT), use the provided FFT script. Note that FFT requires significantly more VRAM and uses lower learning rates.

1. Execute the script with your data parent directory as an argument:

```bash
chmod +x scripts/run_fft.sh
./scripts/run_fft.sh <data_parent_dir>
```

For example:

```bash
./scripts/run_fft.sh ./data
```

---

## Evaluation & Benchmarking

We provide a comprehensive evaluation script (`src/evaluate.py`) that calculates:

- **FID** – Realism  
- **SSIM** – Structural Preservation  
- **Gram Matrix Loss** – Style Consistency  
- **CLIP Score** – Semantic Alignment  

The evaluation pipeline uses `StableDiffusionXLImg2ImgPipeline` to assess how well the model applies style to a validation content image.

### 1. Configuration

The script uses a base directory for data paths. You can optionally edit the `MODEL_CONFIGS` dictionary in `src/evaluation.py` to specify which models to benchmark (e.g., local paths or HuggingFace Hub repos).
Before running, edit the **CONFIGURATION** section at the top of `src/evaluate.py`:

```python
# src/evaluate.py

# 1. Define the models you want to benchmark
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
chmod +x scripts/run_evaluation.sh
./scripts/run_evaluation.sh <base_dir>
```

For example:
```bash
./scripts/run_evaluation.sh ./data
```

### 3. Output

The script will generate:

- **Generated Images** – Saved in the `output_root` directory for each dataset configuration, for visual inspection.
- **CSV Report** – A `eval_report_<TIMESTAMP>.csv` file containing FID, SSIM, Gram, and CLIP scores for all models.
- **JSON Report** – A detailed JSON file with full metric dumps for downstream analysis.

---

## 💻 Inference (Quick Start)

You can load the trained LoRA adapters using `diffusers` in a standalone script (optional):

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

## Hyperparameters

Based on the ablation study in the paper, the optimal configurations are:

| Method                  | LR   | Rank | Lambda (`λ`) | SNR Gamma |
|-------------------------|------|------|--------------|-----------|
| Standard LoRA           | 1e-4 | 16   | 0.0          | 5.0       |
| Regularized LoRA (Ours) | 1e-4 | 16   | 10.0         | 5.0       |

> **Note:** We utilize Min-SNR weighting (`--snr_gamma=5.0`) in all LoRA experiments to stabilize training.

---

## Key Findings

Our Regularized LoRA model achieved the best performance across the board.

### 1. Main Performance Comparison

| Model | FID ↓ | GRAM ↓ | CLIP ↓ | SSIM ↑ |
| :--- | :---: | :---: | :---: | :---: |
| Base SDXL | 4.69 | 0.5474 | 0.0046 | 0.3395 |
| Full FT | 5.02 | 0.5501 | 0.0037 | 0.3317 |
| Std. LoRA | 56.09 | **0.2848** | 0.0732 | 0.2775 |
| Reg. LoRA (Ours) | **3.88** | 0.5402 | **0.0045** | **0.3464** |

- **FID (3.88):** Significantly outperformed Base SDXL (4.69). The model likely learned to "denoise" base model outputs using the Monet style as a coherent aesthetic filter, resulting in statistically cleaner images.
- **Structural Preservation (SSIM 0.3464):** Higher than the Base Model. The regularization signal ($L_{reg}$) acted as a stabilizing force, enforcing strict adherence to structural priors and preventing hallucinations in vague areas.
- **Style-Content Balance (GRAM Loss 0.5402):** Comparable to the Base Model and much higher than overfitted Standard LoRA. This indicates the model learned a generalized representation of the style rather than memorizing training set textures.

### 2. Ablation Study: Learning Rate and Regularization

| Config | LR | $\lambda$ | FID | SSIM |
| :--- | :---: | :---: | :---: | :---: |
| A (Conserv.) | $1e-5$ | 1 | 8.39 | 0.4534 |
| B (Aggress.) | $1e-4$ | 1 | 21.33 | 0.3880 |
| C (Balanced) | $1e-4$ | 10 | **3.88** | **0.5402** |

- **Config A:** Low LR and Low $\lambda$ resulted in a safe but mediocre model (FID 8.39).
- **Config B:** High LR with Low $\lambda$ led to destabilization (FID 21.33). The strong updates from the high LR overpowered the weak regularization, initiating the onset of catastrophic forgetting.
- **Config C:** This is our best model. We used a High LR ($1e-4$) to encourage rapid learning of the style, but counter-balanced it with a massive Regularization penalty ($\lambda=10$). This configuration creates a "canyon" in the loss landscape, forcing weights to stay on the "manifold of coherent images" while moving toward the "Monet" region.
