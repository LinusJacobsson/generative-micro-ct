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
import numpy

class GenerationConfig:
    image_size = 32
    eval_batch_size = 16
    num_epochs = 5000
    z_coord = 10810400
    mixed_precision = "fp16"
    output_dir = "DDPM_results/"
    seed = 0

config = GenerationConfig()

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

transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size), antialias=True),
    transforms.Normalize([0.5], [0.5]),
])

def generate_conditioned_sample(model, noise_scheduler, z_coord):
    with torch.no_grad():
        # Generate samples conditioned on a specific z-coordinate
        images = []
        for _ in range(config.eval_batch_size):
            batch_size = 1  # Set batch size to 1 for each sample
            timesteps = torch.LongTensor([50])  # You can adjust this value
            noise1 = torch.randn(batch_size, 1, config.image_size, config.image_size)
            noise2 = torch.randn(batch_size, 1, config.image_size, config.image_size)
            noise = torch.cat([noise1, noise2], dim=1)
            
            # Condition on z-coordinate
            z_coords_tensor = torch.tensor([z_coord], dtype=torch.float32).view(-1, 1, 1, 1)
            z_coords_tensor = z_coords_tensor.expand(-1, 1, config.image_size, config.image_size)
            
            noisy_image = noise_scheduler.add_noise(noise, z_coords_tensor, timesteps)
            generated_image = model(noisy_image, timesteps).sample

            # Apply normalization to the generated image
            generated_image = transform(generated_image)

            # Append the generated image to the list
            images.append(generated_image.squeeze().cpu())


        return images

# Save generated samples
def make_grid(images, rows, cols):

    w, h = images[0].size()
    grid = Image.new("L", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        pil_image = transforms.functional.to_pil_image(image)
        grid.paste(pil_image, box=(i % cols * w, i // cols * h))
    return grid

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
images = generate_conditioned_sample(pretrained_model, noise_scheduler, config.z_coord)

image_grid = make_grid(images, rows=4, cols=4)
test_dir = os.path.join(config.output_dir, "generated_samples")
os.makedirs(test_dir, exist_ok=True)
image_grid.save(f"{test_dir}/model_epoch_{config.num_epochs}_z_coord_{config.z_coord}.png")

print("Generation done")