# Visualise original images
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class VisualisationConfig:
    image_size = 128
    z_coord = 400
    output_dir = "DDPM_results_128_new/"

config = VisualisationConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_dir = '128x128_boxes/'

class TumorDataSet(Dataset):
    def __init__(self, data_dir, image_size):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif') and not f.startswith('._')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        image = Image.open(img_path)
        image = np.array(image, dtype=np.float32)
        image = image / 65535.0  # Normalize the image
        image_tensor = torch.tensor(image).unsqueeze(0)

        return image_tensor

dataset = TumorDataSet(data_dir, config.image_size)

# Display the image at the specified z_coord
z_image = dataset[config.z_coord].squeeze(0).numpy()
z_image = (z_image * 255).astype(np.uint8)

# Save the visualized image
output_dir = os.path.join(config.output_dir, "original_samples")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"original_image_zcoord_{config.z_coord}.png")
Image.fromarray(z_image).save(output_path)

print("Visualisation completed")