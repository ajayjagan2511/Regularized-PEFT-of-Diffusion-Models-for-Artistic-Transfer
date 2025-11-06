# src/generate_for_fid.py (Updated for comparability)

import torch
from diffusers import DiffusionPipeline
import argparse
import os
from tqdm import tqdm

# A diverse list of ~250 prompts to test the models' general knowledge.
PROMPT_LIST = [
    "a majestic lion in the savannah", "a futuristic cityscape at night", "a tranquil forest stream", 
    "an astronaut floating in space", "a classic red sports car", "a quiet library filled with books",
    "a bustling farmers market", "a serene beach at sunset", "a knight in shining armor",
    "a robot playing chess", "a steaming cup of coffee", "a detailed world map",
    "a vintage steam train", "a mysterious abandoned mansion", "a close-up of a sunflower",
    "a dragon perched on a mountain", "a wizard casting a spell", "a pirate ship on the open sea",
    "a delicious-looking pizza", "a modern architectural building", "a hummingbird in mid-flight",
    "a snowy mountain peak", "a portrait of a wise old person", "a child's colorful drawing",
    "a bowl of fresh fruit", "a classical greek statue", "a neon-lit alleyway",
    "a powerful waterfall", "a galaxy with swirling stars", "a microscope showing cells",
    "a plate of sushi", "a cozy fireplace", "an ancient egyptian pyramid",
    "a guitar leaning against a wall", "a desert landscape with cacti", "a group of hot air balloons",
    "a ballet dancer on stage", "a wolf howling at the moon", "a Formula 1 race car",
    "a peaceful zen garden", "a cyberpunk street scene", "an eagle soaring through the sky",
    "a vinyl record playing", "a coral reef teeming with life", "a glass of red wine",
    "a vintage camera", "a fantasy floating island", "a field of lavender",
    "a busy New York City street", "a plate of pancakes with syrup", "a stained glass window",
    "a close-up of a cat's eye", "a medieval castle", "a scientist in a laboratory",
    "a blooming cherry blossom tree", "a majestic whale breaching the ocean", "a stack of old books",
    "a rustic wooden cabin", "a full moon in a clear night sky", "a tennis player serving the ball",
    "a plate of spaghetti", "a thunderstorm over the plains", "a Ferris wheel at a carnival",
    "a vintage telephone", "a lush jungle with a hidden temple", "a flock of birds migrating",
    "a chess board mid-game", "a set of artist's paintbrushes", "a majestic elephant",
    "a beautiful geode crystal", "a samurai warrior in armor", "a slice of chocolate cake",
    "a Venetian canal with gondolas", "a lighthouse on a rocky shore", "a field of tulips",
    "a roaring fireplace in a cozy room", "a high-speed bullet train", "a butterfly on a flower",
    "a computer motherboard, close-up", "a tranquil pond with lily pads", "a steampunk airship",
    "a scarecrow in a field", "a freshly baked loaf of bread", "a grand piano",
    "a tropical island paradise", "a volcano erupting", "a collection of seashells",
    "a giant redwood tree", "a detective's office, film noir style", "a bowl of ramen",
    "an owl with large eyes", "a vintage bicycle", "a constellation map",
    "a bonsai tree", "a glass of iced tea with lemon", "a dolphin leaping from the water",
    "a violin and bow", "a winding country road", "a set of antique keys",
    "a powerful phoenix rising from ashes", "a colorful chameleon", "a juicy hamburger",
    "a reflection in a puddle", "a windmill in the countryside", "a pair of hiking boots",
    "a glowing mushroom forest", "a cup of hot chocolate with marshmallows", "a red panda",
    "a classic pocket watch", "a Japanese torii gate", "a beehive with bees",
    "a set of colored pencils", "a gecko on a leaf", "a sandcastle on the beach",
    "a cup of green tea", "a fox in the snow", "a vintage motorcycle",
    "a spiderweb with dewdrops", "a rustic barn", "a telescope pointed at the stars",
    "a squirrel eating a nut", "a glass of orange juice", "a typewriter",
    "a waterfall in a lush canyon", "a penguin family", "a hot dog with mustard",
    "a compass pointing north", "a lantern in the dark", "a koala in a eucalyptus tree",
    "a sand dune in the desert", "a bowl of cereal", "a vintage globe",
    "a sea turtle swimming", "a classic acoustic guitar", "a cobblestone street",
]

def main():
    parser = argparse.ArgumentParser(description="Generate images for FID evaluation.")
    parser.add_argument("--model_type", type=str, required=True, choices=["base", "lora", "reg_lora"])
    parser.add_argument("--lora_weights_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=250)
    args = parser.parse_args()

    if args.model_type in ["lora", "reg_lora"] and args.lora_weights_path is None:
        raise ValueError("--lora_weights_path is required.")

    os.makedirs(args.output_dir, exist_ok=True)

    prompts_to_generate = PROMPT_LIST[:args.num_images]
    print(f"Using the first {len(prompts_to_generate)} prompts from the fixed list for generation.")

    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, variant="fp16")
    pipe.to("cuda")

    style_token = ""
    if args.model_type == "base":
        print("Using SDXL 1.0 Base Model.")
        style_token = ", in the style of Monet"
    elif args.model_type in ["lora", "reg_lora"]:
        print(f"Loading LoRA weights from: {args.lora_weights_path}")
        pipe.load_lora_weights(args.lora_weights_path, weight_name="pytorch_lora_weights.bin")
        style_token = ", in the style of sks_monet"

    print(f"Generating {len(prompts_to_generate)} images and saving to {args.output_dir}...")
    for i, base_prompt in enumerate(tqdm(prompts_to_generate)): # Iterate through the fixed list
        full_prompt = base_prompt + style_token
        image = pipe(prompt=full_prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        image.save(os.path.join(args.output_dir, f"image_{i:04d}.png"))
    
    print("Image generation complete.")

if __name__ == "__main__":
    main()