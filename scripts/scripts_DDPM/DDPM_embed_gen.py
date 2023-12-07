# Generation Code

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DModel
import numpy as np

class GenerationConfig:
    image_size = 32
    eval_batch_size = 16
    num_epochs = 100
    num_z_coords = 5  # Total number of z-coordinates (same as used during training)
    embedding_dim = 20  # Embedding dimension (same as used during training)
    z_coord_to_generate = 3  # Specify the z-coordinate you want to condition on
    mixed_precision = "fp16"
    output_dir = "DDPM_results/"
    seed = 0

config = GenerationConfig()

class CustomUNet2DModel(UNet2DModel):
    def __init__(self, num_z_coords, embedding_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.z_embedding = torch.nn.Embedding(num_z_coords, embedding_dim)
        # Other initializations if necessary

    def forward(self, x, z_indices, timesteps, *args, **kwargs):
        z_embedded = self.z_embedding(z_indices)  # Shape: [batch_size, embedding_dim]
        z_embedded = z_embedded.view(z_embedded.shape[0], z_embedded.shape[1], 1, 1)
        z_embedded = z_embedded.expand(-1, -1, x.shape[2], x.shape[3])
        
        # Concatenate z embedding with the input along the channel dimension
        x_with_z = torch.cat([x, z_embedded], dim=1)
        
        return super().forward(x_with_z, timesteps, *args, **kwargs)

# Loading pre-trained weights
pretrained_model = CustomUNet2DModel(
num_z_coords=config.num_z_coords,
embedding_dim=config.embedding_dim,
sample_size=config.image_size,
in_channels=1,  # Only grayscale channel, embedding will be added separately
# (other parameters)
)
pretrained_model_path = os.path.join(config.output_dir, f"model_epoch_{config.num_epochs}.pt")
pretrained_model.load_state_dict(torch.load(pretrained_model_path))

# Generating new samples
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
pipeline = DDPMPipeline(unet=pretrained_model, scheduler=noise_scheduler)

# Generate z-coordinate embedding
z_indices = torch.tensor([config.z_coord_to_generate], dtype=torch.long)
z_embedding = pretrained_model.z_embedding(z_indices)
z_embedding = z_embedding.view(1, config.embedding_dim, 1, 1)
z_embedding = z_embedding.expand(-1, -1, config.image_size, config.image_size)

# Generate samples conditioned on z-coordinate
images = []
for _ in range(config.eval_batch_size):
    noise = torch.randn(1, 1, config.image_size, config.image_size)
    noisy_image_with_z = torch.cat([noise, z_embedding], dim=1)
    generated_image = pipeline(batch_size=1, generator=torch.manual_seed(config.seed), latents=noisy_image_with_z).images[0]
    images.append(generated_image)

# Save generated samples
def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("L", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image.convert("L"), box=(i % cols * w, i // cols * h))
    return grid

    image_grid = make_grid(images, rows=4, cols=4)
    test_dir = os.path.join(config.output_dir, "conditioned_samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/conditioned_epoch_{config.num_epochs}_z_{config.z_coord_to_generate}.png")
