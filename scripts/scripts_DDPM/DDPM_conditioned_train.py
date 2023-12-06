# Training Code

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from dataclasses import dataclass
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator, notebook_launcher
from tqdm.auto import tqdm
import glob
from pathlib import Path
import re

@dataclass
class TrainingConfig:
    image_size = 32
    train_batch_size = 16
    num_epochs = 1000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 1000
    save_model_epochs = 1000
    mixed_precision = "fp16"
    output_dir = "DDPM_results/"
    overwrite_output_dir = False
    seed = 0

config = TrainingConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Assuming the files are in the current working directory
image_slices = [file for file in os.listdir() if file.endswith('.tif')]
image_slices.sort()
data_dir = 'test_data/'

def extract_z_coordinate(file_name):
    # Adjust this regular expression to correctly capture your specific z-coordinate format
    pattern = re.compile(r'(\d+)(?=\.\w+$)')  # Example pattern to capture digits before file extension

    match = pattern.search(file_name)
    if match:
        return int(match.group(1))
    else:
        print(f"Warning: Z-coordinate not found in filename {file_name}")
        return 0  # Return a default value or handle this case as needed

class TumorDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif') and not f.startswith('._')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('L')  # Convert to grayscale
            image = np.array(image, dtype=np.float32)
            image = image / 65535.0  # Normalize the image
            image_tensor = torch.tensor(image).unsqueeze(0)
            if self.transform:
                image_tensor = self.transform(image_tensor)

            z_coord = idx  # Assign z-coordinate based on the order
            normalized_z = z_coord / (len(self.image_files) - 1)
            shifted_z = 2 * normalized_z - 1
            return image_tensor, shifted_z
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            return None

# Define transformations
transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size), antialias=True),
    transforms.Normalize([0.5], [0.5]),
])

# Create the dataset
dataset = TumorDataSet(data_dir, transform)
print(len(dataset))

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
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
model = model.to(device) # move to GPU
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    total_training_steps = len(train_dataloader) * config.num_epochs
    progress_bar = tqdm(total=total_training_steps, disable=not accelerator.is_local_main_process)

    for epoch in range(config.num_epochs):
        for step, (clean_images, z_coords) in enumerate(train_dataloader):
            clean_images = clean_images.to(device)
            noise = torch.randn_like(clean_images, device=device)  # Move to device

            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (clean_images.size(0),), device=device).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            # Debugging: Print the shapes of tensors
            #print(f"Shape of clean_images: {clean_images.shape}")
            #print(f"Shape of noise1: {noise.shape}")
            #print(f"Shape of noisy_images: {noisy_images.shape}")
            z_coords_tensor = torch.tensor(z_coords, dtype=torch.float).view(-1, 1, 1, 1).to(device)
            z_coords_tensor = (z_coords_tensor / (len(dataset) - 1) * 2 - 1).to(device)
            z_coords_tensor = z_coords_tensor.expand(-1, -1, noisy_images.size(2), noisy_images.size(3))
            #print(f"Shape of z_coords_tensor before expansion: {z_coords_tensor.shape}")

            #print(f"Shape of z_coords_tensor after expansion: {z_coords_tensor.shape}")

            noisy_images_with_z = torch.cat([noisy_images, z_coords_tensor], dim=1)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images_with_z, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                global_step += 1

        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                model_save_path = os.path.join(config.output_dir, f"model_epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), model_save_path)

        progress_bar.update(len(train_dataloader))
        logs = {"epoch": epoch, "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=epoch * len(train_dataloader))

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)