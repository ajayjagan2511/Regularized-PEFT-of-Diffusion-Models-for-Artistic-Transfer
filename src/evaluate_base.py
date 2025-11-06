# src/evaluate_base.py

import torch
from diffusers import DiffusionPipeline
import argparse
import os
from PIL import Image

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def main():
    parser = argparse.ArgumentParser()
    # We only need an output directory for the base model
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    # --- NO LoRA WEIGHTS ARE LOADED ---
    # This is the key difference. We are using the pure, original SDXL model.
    print("Using the original SDXL 1.0 base model.")

    # Prompts using the generic "Monet" concept that the base model knows
    prompts = [
        "a photo of a cat in the style of monet",
        "an astronaut riding a horse on mars, in the style of monet",
        "a bowl of fruit on a table, in the style of monet",
        "a futuristic city skyline at night, in the style of monet",
        # Test for catastrophic forgetting (should NOT apply the style)
        "a sports car",
        "a dog",
    ]

    images = []
    for i, prompt in enumerate(prompts):
        print(f"Generating: {prompt}")
        image = pipe(prompt=prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        images.append(image)
        image.save(os.path.join(args.output_dir, f"prompt_{i:02d}.png"))
        
    grid = image_grid(images, rows=2, cols=3)
    grid.save(os.path.join(args.output_dir, "evaluation_grid_BASE.png"))
    print(f"Saved evaluation grid to {os.path.join(args.output_dir, 'evaluation_grid_BASE.png')}")

if __name__ == "__main__":
    main()