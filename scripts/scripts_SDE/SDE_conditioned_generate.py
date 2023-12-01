import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import functools
import pandas as pd
import sys
import os
from PIL import Image
import re
from scipy import integrate
from torchvision.utils import make_grid
from skimage import io
from skimage.metrics import structural_similarity as compare_ssim

import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt


## PARAMETERS

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
error_tolerance = 1e-5 #@param {'type': 'number'}
sample_batch_size = 4 #@param {'type':'integer'}
z_index =0

# same as for training
sigma = 25.0 #@param {'type':'number'}
n_epochs = 10000 #@param {'type':'integer'}
batch_size = 32 #@param {'type':'integer'}
lr = 1e-3 #@param {'type':'number'}

## GENERATING

# Load weights
load_filename = f"batchsize_{batch_size}lr_{lr}_sigma_{sigma}_epochs_{n_epochs}.pth"
score_model = ScoreNet(marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256)
score_model.load_state_dict(torch.load(load_filename, map_location=device))
score_model.eval() # Ensure the model is in evaluation mode (not training mode)

diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
sampler = ode_sampler

## Generate samples using the specified sampler.
samples = sampler(score_model,
                  marginal_prob_std_fn,
                  diffusion_coeff_fn,
                  subset_dataset,
                  z_index,
                  sample_batch_size,
                  device=device)

#Sample visualization.
samples = samples.clamp(0.0, 1.0)
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6, 6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1., cmap='gray')

# Save the sample visualization figure
filename = f"batchsize_{batch_size}lr_{lr}_sigma_{sigma}_epochs_{n_epochs}_index_{z_index}"
sample_fig_path = filename+'_generated.png'
plt.savefig(sample_fig_path, bbox_inches='tight', pad_inches=0.1)
plt.close()  # Close the figure to start a new one

image_list, z_coord = subset_dataset[z_index]
prev_image = image_list[0]
current_image = image_list[1]
next_image = image_list[2]

prev_image_np = prev_image.cpu().numpy()
current_image_np = current_image.cpu().numpy()
next_image_np = next_image.cpu().numpy()

# Plotting
plt.figure(figsize=(8, 4))

# Plotting the first image
plt.subplot(1, 3, 1)
plt.imshow(prev_image_np, cmap='gray')  # Assuming images are grayscale
plt.title('Previous Image')
plt.axis('off')

# Plotting the second image
plt.subplot(1, 3, 2)
plt.imshow(current_image_np, cmap='gray')  # Assuming images are grayscale
plt.title('Current Image')
plt.axis('off')

# Plotting the third image
plt.subplot(1, 3, 3)
plt.imshow(next_image_np, cmap='gray')  # Assuming images are grayscale
plt.title('Next Image')
plt.axis('off')

# Save the second set of figures
image_fig_path = filename+'_original.png'
plt.savefig(image_fig_path, bbox_inches='tight', pad_inches=0.1)
plt.close()  # Close the figure


## SSIM

print(f"SSIM of Previous image and generated: {mean_ssim(prev_image_np, samples)}")
print(f"SSIM of Current image and generated: {mean_ssim(current_image_np, samples)}")
print(f"SSIM of Next image and generated: {mean_ssim(next_image_np, samples)}")


## FUNCTIONS

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, output_padding=1)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 3, 3, stride=1)

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, t):
      embed = self.act(self.embed(t))

      # Encoding path
      h1 = self.conv1(x)
      h1 += self.dense1(embed)
      h1 = self.gnorm1(h1)
      h1 = self.act(h1)

      h2 = self.conv2(h1)
      h2 += self.dense2(embed)
      h2 = self.gnorm2(h2)
      h2 = self.act(h2)

      h3 = self.conv3(h2)
      h3 += self.dense3(embed)
      h3 = self.gnorm3(h3)
      h3 = self.act(h3)

      h4 = self.conv4(h3)
      h4 += self.dense4(embed)
      h4 = self.gnorm4(h4)
      h4 = self.act(h4)

      # Decoding path
      h = self.tconv4(h4)
      h += self.dense5(embed)
      h = self.tgnorm4(h)
      h = self.act(h)

      h = self.tconv3(torch.cat([h, h3], dim=1))
      h += self.dense6(embed)
      h = self.tgnorm3(h)
      h = self.act(h)

      h = self.tconv2(torch.cat([h, h2], dim=1))
      h += self.dense7(embed)
      h = self.tgnorm2(h)
      h = self.act(h)

      h = self.tconv1(torch.cat([h, h1], dim=1))

      h = h / self.marginal_prob_std(t)[:, None, None, None]
      return h


def marginal_prob_std(t, sigma):
  """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The standard deviation.
  """
  t = torch.tensor(t, device=device)
  return torch.sqrt((sigma**(2 * t) - 1.) / 2. / np.log(sigma))


def diffusion_coeff(t, sigma):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.

  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t, device=device)


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  """The loss function for training score-based generative models.

  Args:
    model: A PyTorch model instance that represents a
      time-dependent score-based model.
    x: A mini-batch of training data.
    marginal_prob_std: A function that gives the standard deviation of
      the perturbation kernel.
    eps: A tolerance value for numerical stability.
  """
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps
  z = torch.randn_like(x)

  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None, None, None]
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None, None, None] + z)**2, dim=(1,2,3)))
  return loss


def ode_sampler(score_model,
                marginal_prob_std,
                diffusion_coeff,
                dataset,
                z_index,
                batch_size=64,
                atol=error_tolerance,
                rtol=error_tolerance,
                device='cuda',
                z=None,
                eps=1e-3):
  """Generate samples from score-based models with black-box ODE solvers.

  Args:
    score_model: A PyTorch model that represents the time-dependent score-based model.
    marginal_prob_std: A function that returns the standard deviation
      of the perturbation kernel.
    diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
    batch_size: The number of samplers to generate by calling this function once.
    atol: Tolerance of absolute errors.
    rtol: Tolerance of relative errors.
    device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
    z: The latent code that governs the final sample. If None, we start from p_1;
      otherwise, we start from the given z.
    eps: The smallest time step for numerical stability.
  """

  image_list, z_coord = dataset[z_index]
  prev_image = image_list[0]
  prev_image = prev_image.expand(batch_size, 1, -1, -1)
  prev_image = prev_image.to(device)
  next_image = image_list[2]
  next_image = next_image.expand(batch_size, 1, -1, -1)
  next_image = next_image.to(device)

  t = torch.ones(batch_size, device=device)

  # Create the latent code
  if z is None:
    noise = torch.randn(batch_size, 1, 32, 32, device=device) * marginal_prob_std(t)[:, None, None, None]
    init_x = torch.cat([prev_image, noise, next_image], dim=1)
  else:
    init_x = z

  shape = init_x.shape

  def score_eval_wrapper(sample, time_steps):
    """A wrapper of the score-based model for use by the ODE solver."""
    sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
    time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
    with torch.no_grad():
      score = score_model(sample, time_steps)
    return score.cpu().numpy().reshape((-1,)).astype(np.float64)

  def ode_func(t, x):
    """The ODE function for use by the ODE solver."""
    time_steps = np.ones((shape[0],)) * t
    g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
    return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

  # Run the black-box ODE solver.
  res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')
  print(f"Number of function evaluations: {res.nfev}")
  x = torch.tensor(res.y[:, -1], device=device).reshape(shape)

  return x[:,1:2,:,:]


def calculate_ssim(image_1, image_2):
    # Convert PyTorch tensor to NumPy array if necessary
    if isinstance(image_2, torch.Tensor):
        image_2 = image_2.squeeze().detach().cpu().numpy()

    # Check if dimensions match
    if image_1.shape != image_2.shape:
        raise ValueError("The dimensions of the two images do not match.")

    # Compute SSIM between two images
    ssim = compare_ssim(image_1, image_2)

    return ssim


def mean_ssim(image, samples): 
  ssim = 0

  for sample in samples:
     ssim += calculate_ssim(image, sample)

  ssim = ssim/len(samples)
  return ssim