# regdiffusion-peft
Dual-model regularized parameter-efficient fine-tuning (LoRA/SaRA) of SDXL for high-fidelity artistic style transfer while preserving base model priors.

# SDXL — VAE FP16 Fix (Regularized) & Monet LoRA

**Hugging Face:** [`batshiv/sdxl-vae-fp16-fix-regularized`](https://huggingface.co/batshiv/sdxl-vae-fp16-fix-regularized)  
This repo hosts training artifacts and example images. Large binaries live on the Hub (not GitHub).

---

## Contents

```
batshiv/sdxl-vae-fp16-fix-regularized
├── lora_baseline_monet/
│   └── checkpoint-500/
│       ├── model.safetensors         # model state (at this step; may vary by trainer)
│       ├── pytorch_lora_weights.bin  # LoRA adapter weights
│       ├── optimizer.bin             # optimizer state (resume training)
│       ├── random_states_0.pkl       # RNG state (deterministic resume)
│       └── scaler.pt                 # AMP scaler (fp16 training)
└── evaluation_images/
    ├── evaluation_grid.png
    ├── prompt_00.png
    ├── prompt_01.png
    ├── prompt_02.png
    ├── prompt_03.png
    ├── prompt_04.png
    └── prompt_05.png
```

> If more checkpoints/folders are added later, they’ll appear alongside the paths above.

---

## Install

```bash
pip install -U huggingface_hub diffusers accelerate
# Plus PyTorch that matches your CUDA/CPU setup:
# See https://pytorch.org/get-started
```

If the repo is **private**, set your token:

```bash
export HF_TOKEN=hf_********************************
```

---

## Use case A — Inference with the Monet LoRA (SDXL)

```python
import torch
from diffusers import DiffusionPipeline

# Base SDXL (change to your preferred base/refiner if needed)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA directly from the Hub repo
pipe.load_lora_weights(
    "batshiv/sdxl-vae-fp16-fix-regularized",
    weight_name="lora_baseline_monet/checkpoint-500/pytorch_lora_weights.bin",
)

# Optional: fuse & scale LoRA strength
pipe.fuse_lora(lora_scale=0.8)

image = pipe(prompt="a Monet-style landscape at dusk, soft brush strokes, pastel color palette").images[0]
image.save("sample_monet.png")
```

---

## Use case B — Resume training from a checkpoint

```python
from huggingface_hub import snapshot_download
import os

ckpt_root = snapshot_download(
    repo_id="batshiv/sdxl-vae-fp16-fix-regularized",
    allow_patterns=["lora_baseline_monet/checkpoint-500/**"],
    local_dir="hf_artifacts",
    local_dir_use_symlinks=False
)

resume_dir = os.path.join(ckpt_root, "lora_baseline_monet/checkpoint-500")
print("Checkpoint path:", resume_dir)
# Feed `resume_dir` to your trainer's --resume / --checkpoint arg.
```

---

## Use case C — Grab evaluation images (for reports)

```python
from huggingface_hub import snapshot_download
import glob, os

imgs_root = snapshot_download(
    repo_id="batshiv/sdxl-vae-fp16-fix-regularized",
    allow_patterns=["evaluation_images/*.png"],
    local_dir="hf_artifacts",
    local_dir_use_symlinks=False
)

for p in sorted(glob.glob(os.path.join(imgs_root, "evaluation_images/*.png"))):
    print("Found image:", p)
```

---

## Reproducibility

Pin an exact revision (commit SHA or tag):

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="batshiv/sdxl-vae-fp16-fix-regularized",
    revision="<commit-sha-or-tag>",
    allow_patterns=["lora_baseline_monet/checkpoint-500/**"],
    local_dir="hf_artifacts",
    local_dir_use_symlinks=False
)
```

---

## Notes

- Large binaries (`*.bin`, `*.safetensors`) live on the Hub; keep them out of your GitHub repo via `.gitignore`.
- If you later add a separately exported VAE (e.g., saved with `save_pretrained()`), place it under `vae/` and load with:

```python
from diffusers import AutoencoderKL
from torch import float16

pipe.vae = AutoencoderKL.from_pretrained(
    "batshiv/sdxl-vae-fp16-fix-regularized",
    subfolder="vae",
    torch_dtype=float16,
)
```
