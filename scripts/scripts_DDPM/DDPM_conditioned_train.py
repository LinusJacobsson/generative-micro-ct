# Training Code

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline
from pipeline_ddpm_conditioned import DDPMPipelineConditioned
from dataclasses import dataclass
from diffusers.optimization import get_cosine_schedule_with_warmup
from accelerate import Accelerator, notebook_launcher
from tqdm.auto import tqdm
import glob
from pathlib import Path
import re

@dataclass
class TrainingConfig:
    image_size = 128
    train_batch_size = 16
    num_epochs = 2000
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_model_epochs = 500
    mixed_precision = "fp16"
    output_dir = "DDPM_results_128_new/"
    overwrite_output_dir = False
    seed = 0

config = TrainingConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Assuming the files are in the current working directory
image_slices = [file for file in os.listdir() if file.endswith('.tif')]
image_slices.sort()
data_dir = '128x128_boxes/'

class TumorDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif') and not f.startswith('._')])

        # Extract min and max z-coordinates
        z_coordinates = [self.extract_z_coordinate(file_name) for file_name in self.image_files]
        valid_z_coordinates = [z for z in z_coordinates if z is not None]

        if valid_z_coordinates:
            self.min_z = min(valid_z_coordinates)
            self.max_z = max(valid_z_coordinates)
        else:
            raise ValueError("No valid z-coordinates found in the dataset.")

    def extract_z_coordinate(self, file_name):
        pattern = r'(\d+)\.tif$'
        match = re.search(pattern, file_name)

        if match:
            z_coordinate = int(match.group(1))
            return z_coordinate
        else:
            return None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        while True:
            try:
                img_path = os.path.join(self.data_dir, self.image_files[idx])
                image = Image.open(img_path)
                image = np.array(image, dtype=np.float32)
                image = image / 65535.0  # Normalize the image
                image_tensor = torch.tensor(image).unsqueeze(0)

                if self.transform:
                    image_tensor = self.transform(image_tensor)

                if image_tensor.size() != torch.Size([1, config.image_size, config.image_size]):
                    print(f"Warning: Image {idx} at path {img_path} has unexpected size: {image_tensor.size()}")
                    idx = (idx + 1) % len(self.image_files)
                    continue

                # Extract z-coordinate from filename
                z_coordinate = self.extract_z_coordinate(self.image_files[idx])

                if z_coordinate is not None:
                    # Normalize z-coordinate between -1 and 1 based on the actual range
                    normalized_z = 2 * ((z_coordinate - self.min_z) / (self.max_z - self.min_z)) - 1
                    return image_tensor, normalized_z
                else:
                    print(f"Warning: Could not extract z-coordinate from filename {self.image_files[idx]}")
                    idx = (idx + 1) % len(self.image_files)
                    continue

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

            z_coords_tensor = torch.tensor(z_coords, dtype=torch.float).view(-1, 1, 1, 1).to(device)
            z_coords_tensor = z_coords_tensor.expand(-1, -1, config.image_size, config.image_size)

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

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                model_save_path = os.path.join(config.output_dir, f"model_epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), model_save_path)

        progress_bar.update(len(train_dataloader))
        logs = {"epoch": epoch, "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=epoch * len(train_dataloader))

args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)

print("Training completed")