# src/dataset.py

import os
from torch.utils.data import Dataset
from PIL import Image

class PairedImageDataset(Dataset):
    """
    Custom PyTorch Dataset for loading paired images.
    It assumes that the filenames in monet_dir and original_dir are identical.
    
    Args:
        monet_dir (str): Path to the directory containing Monet-stylised images.
        original_dir (str): Path to the directory containing the original photos.
        transform (callable, optional): A function/transform to apply to the images.
    """
    def __init__(self, monet_dir, original_dir, transform=None):
        self.monet_dir = monet_dir
        self.original_dir = original_dir
        self.transform = transform
        
        # Get a sorted list of filenames from the monet directory
        # This ensures that the pairing is consistent
        self.image_files = sorted([f for f in os.listdir(monet_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        
        monet_path = os.path.join(self.monet_dir, filename)
        original_path = os.path.join(self.original_dir, filename)
        
        # Ensure the corresponding original file exists
        if not os.path.exists(original_path):
            raise FileNotFoundError(f"Original image not found for {filename} at {original_path}")

        monet_image = Image.open(monet_path).convert("RGB")
        original_image = Image.open(original_path).convert("RGB")
        
        # Apply transforms if provided
        if self.transform:
            monet_pixel_values = self.transform(monet_image)
            original_pixel_values = self.transform(original_image)
        else:
            monet_pixel_values = monet_image
            original_pixel_values = original_image
            
        return {
            "monet_pixel_values": monet_pixel_values,
            "original_pixel_values": original_pixel_values
        }