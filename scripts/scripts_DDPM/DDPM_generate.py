# Generation Code

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from tqdm.auto import tqdm
import glob

class GenerationConfig:
    image_size = 32
    eval_batch_size = 16
    num_epochs = 100
    mixed_precision = "fp16"
    output_dir = "DDPM_results/"
    seed = 0

config = GenerationConfig()

# Loading pre-trained weights
pretrained_model = UNet2DModel(
    sample_size=config.image_size,
    in_channels=1,
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

# Generating new samples
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# Generate samples
pipeline = DDPMPipeline(unet=pretrained_model, scheduler=noise_scheduler)
images = pipeline(batch_size=config.eval_batch_size, generator=torch.manual_seed(config.seed)).images

# Save generated samples
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("L", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image.convert("L"), box=(i % cols * w, i // cols * h))
    return grid

image_grid = make_grid(images, rows=4, cols=4)
test_dir = os.path.join(config.output_dir, "generated_samples")
os.makedirs(test_dir, exist_ok=True)
image_grid.save(f"{test_dir}/model_epoch_{config.num_epochs}.png")