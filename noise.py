import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from tqdm import tqdm


image_size = 128
transform = transforms.Compose([
    transforms.Resize((image_size, image_size), antialias=True)
])

class TumorDataSet3Channels(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []

        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.tif') and not f.startswith('._'):
                img_path = os.path.join(data_dir, f)
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.image_files.append(f)
                except Exception as e:
                    print(f"Skipping file {f} due to error: {e}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load images
        jump_size = 3  # Adjust this value for different jump sizes
       # Calculate indices for previous and next images
        prev_idx = max(idx - jump_size, 0)
        next_idx = min(idx + jump_size, len(self.image_files) - 1)
        
         # Load images
        prev_image = self.load_image(prev_idx)
        target_image = self.load_image(idx)
        next_image = self.load_image(next_idx)
#################### Lägg till noise här för LQ, kommentera ut för GT####################
        lambda_value = 0.3  # Adjust this value as needed
        noise = torch.randn(target_image.size()) * lambda_value
        target_image += noise

        # Check dimensions of each image
        if (prev_image.ndim != 3 or target_image.ndim != 3 or next_image.ndim != 3):
            return None

        # Concatenate images along the channel dimension
        combined_image = torch.cat([prev_image, target_image, next_image], dim=0)
        return combined_image, idx
    
    def load_image(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)
        image = np.array(image, dtype=np.float32)

        # Normalize the image for 16-bit depth
        image = image / 65535.0
        image_tensor = torch.tensor(image)

        # Add a channel dimension and apply transformation
        image_tensor = image_tensor.unsqueeze(0)
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor
    
    def get_context_images(self, idx):
        # Load the previous and next images, handle edge cases
        prev_index = max(idx - 1, 0)
        next_index = min(idx + 1, len(self.image_files) - 1)

        prev_image = self.load_image(prev_index)
        next_image = self.load_image(next_index)

        # Concatenate images along the channel dimension
        context_images = torch.cat([prev_image, next_image], dim=0)

        return context_images



def save_concatenated_images(dataset, save_dir, new_size=(128, 128)):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        data = dataset[idx]

        # Skip if data is None
        if data is None:
            continue

        combined_image, _ = data
        combined_image = combined_image.numpy() * 255  # Scale to 8-bit
        combined_image = combined_image.astype(np.uint8)
        combined_image = np.transpose(combined_image, (1, 2, 0))

        try:
            # Convert to PIL Image
            img = Image.fromarray(combined_image)

            # Save image
            img.save(os.path.join(save_dir, f"more_noise_{idx}.png"))
        except Exception as e:
            print(f"Error saving image {idx}: {e}")
        # Usage
data_dir = '128x128_boxes/'
save_dir = 'experiments/gap_3/'
dataset = TumorDataSet3Channels(data_dir, transform=transform)
save_concatenated_images(dataset, save_dir)
