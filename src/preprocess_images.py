# preprocess_images.py
import os
import argparse
from PIL import Image

# --- Configuration ---
# TARGET_SIZE and IMAGE_DIR are now command-line arguments

def resize_and_crop(image_path, output_size):
    """
    Resizes an image to fill a square of output_size and crops the center.
    """
    img = Image.open(image_path)
    
    # Don't process non-RGB images like grayscale
    if img.mode != 'RGB':
        print(f"Image {image_path} is in {img.mode} mode, converting to RGB.")
        img = img.convert('RGB')

    # 1. Calculate new dimensions to resize to
    original_width, original_height = img.size
    original_ratio = original_width / original_height
    target_ratio = 1.0 # output_size / output_size

    if original_ratio > target_ratio:
        # Original image is wider than target. Resize based on height.
        new_height = output_size
        new_width = int(new_height * original_ratio)
    else:
        # Original image is taller than or same as target. Resize based on width.
        new_width = output_size
        new_height = int(new_width / original_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)

    # 2. Crop the center
    left = (new_width - output_size) / 2
    top = (new_height - output_size) / 2
    right = (new_width + output_size) / 2
    bottom = (new_height + output_size) / 2

    img = img.crop((left, top, right, bottom))
    
    # 3. Save the image, overwriting the original
    img.save(image_path)
    print(f"Processed {image_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images by resizing and cropping to square.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to preprocess.")
    parser.add_argument("--target_size", type=int, required=True, help="Target size for the square crop.")
    
    args = parser.parse_args()
    
    print(f"Starting to process images in {args.image_dir}...")
    for filename in os.listdir(args.image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(args.image_dir, filename)
            resize_and_crop(file_path, args.target_size)
    print("Processing complete.")