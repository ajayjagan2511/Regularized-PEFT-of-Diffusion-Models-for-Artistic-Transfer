import os
import argparse
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def clean_caption(text):
    # Remove chatty introductions common in VLMs
    prefixes = [
        "The image shows", "The image depicts", "The image features", 
        "In this image", "There is", "A picture of", "An image of",
        "This is an image of"
    ]
    for prefix in prefixes:
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
            # Capitalize first letter if string is not empty
            if text:
                text = text[0].upper() + text[1:]
    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", type=str, required=True)
    parser.add_argument("--style_dir", type=str, required=True)
    parser.add_argument("--style_prefix", type=str, default="sksmonet style")
    args = parser.parse_args()

    # 1. Load Model (Optimized for H100)
    print("Loading Qwen2-VL-7B...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # 2. Get Images
    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    files = [f for f in os.listdir(args.original_dir) if os.path.splitext(f)[1].lower() in exts]
    files.sort()

    print(f"Processing {len(files)} images...")

    # 3. Loop
    for filename in tqdm(files):
        img_path = os.path.join(args.original_dir, filename)
        
        # Prompt specifically for image training captions
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": "Describe the main subject and setting of this image concisely. Do not describe the artistic style, frame, or brushstrokes. Focus only on the physical objects and scene."}
            ]
        }]

        # Inference
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
        
        # Decode
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        caption = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        # Clean and Format
        final_caption = clean_caption(caption)
        styled_caption = f"{args.style_prefix}, {final_caption}"

        # Save
        txt_name = os.path.splitext(filename)[0] + ".txt"
        
        with open(os.path.join(args.original_dir, txt_name), "w") as f:
            f.write(final_caption)
            
        with open(os.path.join(args.style_dir, txt_name), "w") as f:
            f.write(styled_caption)

if __name__ == "__main__":
    main()