# Generation Code

import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from diffusers import DDPMScheduler, DDPMPipeline, UNet2DModel
import numpy as np
from pipeline_ddpm_embed import EmbeddedDDPMPipeline

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
        # Calculate the total number of input channels (1 image channel + embedding_dim)
        total_in_channels = 1 + embedding_dim
        # Update the in_channels parameter before calling the base class initializer
        kwargs["in_channels"] = total_in_channels
        super().__init__(*args, **kwargs)
        self.z_embedding = torch.nn.Embedding(num_z_coords, embedding_dim)

    def forward(self, x, z_indices, timesteps, *args, **kwargs):
        # No need to concatenate z_embedding here, as it's already included in x
        return super().forward(x, timesteps, *args, **kwargs)



# Loading pre-trained weights
pretrained_model = CustomUNet2DModel(
    num_z_coords = 5,
    embedding_dim=20,
    sample_size=config.image_size,
    in_channels=21,
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
pretrained_model_path = os.path.join(config.output_dir, f"model_epoch_{config.num_epochs}.pt")
pretrained_model.load_state_dict(torch.load(pretrained_model_path))

# Generating new samples
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
pipeline = EmbeddedDDPMPipeline(unet=pretrained_model, scheduler=noise_scheduler)

# Generate a sample conditioned on a specific z-coordinate
z_coord = 3  # Example z-coordinate
z_indices = torch.tensor([z_coord])  # Convert z_coord to a tensor

# Generate the image using the pipeline with only z_indices (let the pipeline generate the noise)
generated_image = pipeline(
    batch_size=1, 
    generator=torch.manual_seed(0), 
    z_indices=z_indices  # Pass only z_indices for conditional generation
).images[0]

# Check the shape of the generated image
print("Generated image shape:", generated_image.shape)

# Assuming the generated image is grayscale, take the first channel
# Adjust this part if working with RGB images
if generated_image.shape[-1] != 1:
    generated_image = generated_image[..., 0]

# Check the shape after modification
print("Modified image shape:", generated_image.shape)

# Convert to PIL image
generated_image_pil = transforms.functional.to_pil_image(generated_image)
generated_image_pil.save("generated_sample.png")