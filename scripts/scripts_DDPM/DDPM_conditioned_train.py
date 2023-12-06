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
    num_epochs = 10
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 50
    save_model_epochs = 50
    mixed_precision = "fp16"
    output_dir = "DDPM_results/"
    overwrite_output_dir = False
    seed = 0

config = TrainingConfig()

# Assuming the files are in the current working directory
image_slices = [file for file in os.listdir() if file.endswith('.tif')]
image_slices.sort()
data_dir = '32x32_boxes/'

def extract_z_coordinate(self, file_name):
        # Define a regular expression pattern to match the z-coordinate
        pattern = re.compile(r'\d+\.tif$')

        # Use the regular expression to find the match in the file name
        match = pattern.search(file_name)

        # Extract the matched part
        if match:
            return int(match.group(0).replace('.tif', ''))
        else:
            # Handle the case when the z-coordinate is not found
            print(f"Warning: Z-coordinate not found in filename {filename}")
            return None  # You can modify this to handle missing z-coordinate appropriately

class TumorDataSet(Dataset):
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

    def __getitem__(self, idx):
        while True:
            try:
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

                # Check the size of the tensor and log if it's not [1, 32, 32]
                if image_tensor.size() != torch.Size([1, config.image_size, config.image_size]):
                    print(f"Warning: Image {idx} at path {img_path} has unexpected size: {image_tensor.size()}")
                    idx = (idx + 1) % len(self.image_files)
                    continue

                # Extract z-coordinate from the filename or any other source
                z_coord = extract_z_coordinate(self.image_files[idx], img_path)

                return image_tensor, z_coord
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                idx = (idx + 1) % len(self.image_files)  # Move to the next image

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

sample_image, _ = dataset[0]
sample_image = sample_image.unsqueeze(0)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise1 = torch.randn(sample_image.shape)
noise2 = torch.randn(sample_image.shape)
noise = torch.cat([noise1, noise2], dim=1)

timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

noisy_image_float = noisy_image.type(torch.float32)
if len(noisy_image_float.shape) == 3:
    noisy_image_float = noisy_image_float.unsqueeze(0)

noise_pred = model(noisy_image_float, timesteps).sample
loss = F.mse_loss(noise_pred, noise.type(torch.float32))

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
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            # Pass z-coordinate information to the model
            z_coords_tensor = torch.tensor(z_coords, device=clean_images.device, dtype=torch.float32).view(-1, 1, 1, 1)
            z_coords_tensor = z_coords_tensor.expand(-1, 1, config.image_size, config.image_size)

            noisy_z_coords = noise_scheduler.add_noise(clean_images, noise, timesteps) 

            noisy_images = torch.cat([noisy_images, noisy_z_coords], dim=1)

            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
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
