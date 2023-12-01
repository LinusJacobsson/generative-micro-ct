# -*- coding: utf-8 -*-
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
import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt

@dataclass
class TrainingConfig:
    image_size = 512  # the generated image resolution
    train_batch_size = 1
    eval_batch_size = 1  # how many images to sample during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 50
    save_model_epochs = 50
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "DDPM_results/"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


config = TrainingConfig()
# Assuming the files are in the current working directory
image_slices = [file for file in os.listdir() if file.endswith('.tif')]
image_slices.sort()  # Sort the file names if necessary
data_dir = 'whole_tumors/'

class TumorDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = []

        for f in sorted(os.listdir(data_dir)):
            if f.endswith('.tif') and not f.startswith('._'):
                img_path = os.path.join(data_dir, f)
                # Optionally, add a check here to ensure the image can be opened
                try:
                    with Image.open(img_path) as img:
                        # Check if image can be opened and read
                        img.verify()  # or img.load() to catch more types of errors
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

                # Check the size of the tensor and log if it's not [1, 128, 128]
                if image_tensor.size() != torch.Size([1, 512, 512]):
                    print(f"Warning: Image {idx} at path {img_path} has unexpected size: {image_tensor.size()}")
                    idx = (idx + 1) % len(self.image_files)
                    continue

                return image_tensor
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                idx = (idx + 1) % len(self.image_files)  # Move to next image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((512, 512), antialias=True),
    # You might want to adjust or remove the Normalize transform 
    # depending on how you wish to handle the 16-bit data.
    transforms.Normalize([0.5], [0.5]),
])

# Create the dataset
dataset = TumorDataSet(data_dir, transform)
print(len(dataset))

image = dataset[0]
image_np = image.squeeze().numpy()
# Plot the image
#plt.imshow(image_np, cmap='gray')  # 'cmap=gray' is used for proper grayscale display
#plt.title('Resized Image')
#plt.axis('off')  # To turn off the axis
#plt.show()

train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

model = UNet2DModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=1,  # the number of input channels, 3 for RGB images
    out_channels=1,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)


sample_image = dataset[0].unsqueeze(0)
# Assuming 'sample_image' is defined and its shape is [1, 1, 128, 128]

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
# Create noise that matches the single-channel image shape
noise = torch.randn(sample_image.shape)

timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

# Process the image for visualization
# Remove the batch dimension
noisy_image = noisy_image.squeeze(0)
# Scale to [0, 255] and convert to uint8
noisy_image = ((noisy_image + 1.0) * 127.5).clamp(0, 255).type(torch.uint8)

# Convert to PIL Image for grayscale
image = Image.fromarray(noisy_image.squeeze(0).cpu().numpy(), 'L')

# Display or save the image as needed
#image.show()

noisy_image_float = noisy_image.type(torch.float32)
if len(noisy_image_float.shape) == 3:
    noisy_image_float = noisy_image_float.unsqueeze(0)

# Pass the float tensor to the model
noise_pred = model(noisy_image_float, timesteps).sample
loss = F.mse_loss(noise_pred, noise.type(torch.float32))

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("L", size=(cols * w, rows * h))  # 'L' mode for grayscale
    for i, image in enumerate(images):
        grid.paste(image.convert("L"), box=(i % cols * w, i // cols * h))  # Convert each image to grayscale
    return grid


def evaluate(config, epoch, pipeline):
    # Sample some images from random noise
    images = pipeline(batch_size=config.eval_batch_size, generator=torch.manual_seed(config.seed)).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images locally
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0
    total_training_steps = len(train_dataloader) * config.num_epochs
    progress_bar = tqdm(total=total_training_steps, disable=not accelerator.is_local_main_process)
    # Now you train the model
    for epoch in range(config.num_epochs):
        for step, clean_images in enumerate(train_dataloader):

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if  epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                # Save the model locally
                model_save_path = os.path.join(config.output_dir, f"model_epoch_{epoch}.pt")
                torch.save(model.state_dict(), model_save_path)
        # Update the progress bar after each epoch
        progress_bar.update(len(train_dataloader))
        logs = {"epoch": epoch, "loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=epoch * len(train_dataloader))


args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)
sample_images = sorted(glob.glob(f"{config.output_dir}/samples/*.png"))
#Image.open(sample_images[-1])

