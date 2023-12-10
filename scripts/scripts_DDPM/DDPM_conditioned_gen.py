# Generation Code

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from pipeline_ddpm_conditioned import DDPMPipelineConditioned
from tqdm.auto import tqdm
import glob
import numpy as np

class GenerationConfig:
    image_size = 128
    eval_batch_size = 9
    num_epochs = 2000
    z_coord = 400
    num_images = 500
    mixed_precision = "fp16"
    output_dir = "DDPM_results_128_new/"
    seed = 0

config = GenerationConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading pre-trained weights
pretrained_model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=2,
    out_channels=1,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

pretrained_model_path = os.path.join(config.output_dir, f"model_epoch_{config.num_epochs}.pt")  # Assuming weights from the last epoch
pretrained_model.load_state_dict(torch.load(pretrained_model_path))

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Normalising z
normalized_z = config.z_coord / (config.num_images - 1)
shifted_z = 2 * normalized_z - 1
z_coord_tensor = torch.tensor(shifted_z, dtype=torch.float).view(-1, 1, 1, 1).to(device)
z_coord_tensor = z_coord_tensor.expand(-1, -1, config.image_size, config.image_size)

# Generate samples
pipeline = DDPMPipelineConditioned(unet=pretrained_model, scheduler=noise_scheduler, z_coord=z_coord_tensor)
images = pipeline(batch_size=config.eval_batch_size, generator=torch.manual_seed(config.seed)).images


def make_grid(images, rows, cols, border_size=5, border_color=0):
    w, h = images[0].size
    grid_w = cols * (w + border_size) - border_size
    grid_h = rows * (h + border_size) - border_size
    grid = Image.new("L", size=(grid_w, grid_h), color=border_color)

    for i, image in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * (w + border_size)
        y = row * (h + border_size)
        grid.paste(image.convert("L"), box=(x, y))

    return grid


image_grid = make_grid(images, rows=3, cols=3)
test_dir = os.path.join(config.output_dir, "generated_samples")
os.makedirs(test_dir, exist_ok=True)
image_grid.save(f"{test_dir}/model_epoch_{config.num_epochs}_zcoord_{config.z_coord}.png")

print("Generation completed")