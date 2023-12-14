import os
import numpy as np
import torch
from PIL import Image
from functions import TumorDataSet3Channels  # Import your TumorDataSet class

def save_concatenated_images(dataset, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx in range(len(dataset)):
        combined_image, _ = dataset[idx]

        # Convert the tensor to a numpy array and scale it back to 16-bit
        combined_image = combined_image.numpy() * 65535
        combined_image = combined_image.astype(np.uint16)

        # Convert to PIL Image and save
        img = Image.fromarray(combined_image)
        img.save(os.path.join(save_dir, f"concatenated_{idx}.tif"))

# Usage
data_dir = '/cephyr/users/linusjac/Alvis/whole_tumors/'
save_dir = 'whole_tumors/GT'
dataset = TumorDataSet(data_dir)
save_concatenated_images(dataset, save_dir)
