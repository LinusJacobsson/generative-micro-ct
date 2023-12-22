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

    def interpolate(self, prev_image, next_image):
        # Perform linear interpolation
        interpolated = (prev_image + next_image) / 2

        # Squeeze the extra dimension to make it 2D
        interpolated_2d = torch.squeeze(interpolated)

        return interpolated_2d


    def __getitem__(self, idx):
        jump_size = 8  # Set the jump size
        max_index = len(self.image_files) - 1

        # Calculate indices for interpolation
        prev_idx = max(0, idx - jump_size)
        next_idx = min(idx + jump_size, max_index)

        prev_image = self.load_image(prev_idx)
        next_image = self.load_image(next_idx)

        print(f"Shapes before interpolation: prev {prev_image.shape}, next {next_image.shape}")
        
        interpolated_image = self.interpolate(prev_image, next_image)

        print(f"Shape after interpolation: {interpolated_image.shape}")

        if interpolated_image.ndim != 2:
            print(f"Unexpected shape, skipping idx {idx}")
            return None, None  # Return a tuple with None values


        return interpolated_image, idx

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

def save_interpolated_images(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        interpolated_image, _ = dataset[idx]

        if interpolated_image is None:
            continue

        # Convert to numpy array and scale to 8-bit
        interpolated_image = interpolated_image.numpy() * 255
        interpolated_image = interpolated_image.astype(np.uint8)

        # If the image has an extra dimension of size 1, remove it
        if interpolated_image.ndim == 3 and interpolated_image.shape[-1] == 1:
            interpolated_image = np.squeeze(interpolated_image, axis=-1)

        # Save the interpolated image
        img = Image.fromarray(interpolated_image)
        img.save(os.path.join(save_dir, f"interpolated_{idx}.png"))

# Usage
dataset = TumorDataSet3Channels('128x128_boxes/', transform=transform)
save_interpolated_images(dataset, 'experiments/gap_8/interpolated')



import os

def filter_interpolated_images(interpolated_dir, test_gt_dir):
    # Gather indices from test directory
    test_indices = set()
    for filename in os.listdir(test_gt_dir):
        if filename.startswith('more_noise_'):
            index = int(filename.split('_')[-1].split('.')[0])
            test_indices.add(index)

    # Check and remove unnecessary images in interpolated directory
    for filename in os.listdir(interpolated_dir):
        if filename.startswith('interpolated_'):
            index = int(filename.split('_')[-1].split('.')[0])
            if index not in test_indices:
                os.remove(os.path.join(interpolated_dir, filename))
                print(f"Removed: {filename}")

# Specify directories
interpolated_dir = 'experiments/gap_8/interpolated/'
test_gt_dir = 'experiments/gap_8/test/GT/'

# Call function
filter_interpolated_images(interpolated_dir, test_gt_dir)
