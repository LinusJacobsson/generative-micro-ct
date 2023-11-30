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
from torch.optim import Adam
import tqdm
from scipy import integrate
from torchvision.utils import make_grid

import matplotlib
matplotlib.use('Agg')  # Set the backend before importing pyplot
import matplotlib.pyplot as plt


## PARAMETERS

device = 'cuda' #@param ['cuda', 'cpu'] {'type':'string'}
sigma = 25.0 #@param {'type':'number'}
n_epochs = 10000 #@param {'type':'integer'}
batch_size = 32 #@param {'type':'integer'}
lr = 1e-3 #@param {'type':'number'}


## LOADING DATA

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '32x32_boxes')

transform = transforms.Compose([
    transforms.Resize((32, 32))
])


## TRAINING

marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)

dataset = TumorDataSet(data_dir, transform)
num_images = len(dataset)

for i in range(num_images):

  image = dataset[i]

  image_list, z_coord = image
  prev_image = image_list[0]
  current_image = image_list[1]
  next_image = image_list[2]
  current_image_np = current_image.squeeze().numpy()
  prev_image_np = prev_image.squeeze().numpy()
  next_image_np = next_image.squeeze().numpy()

dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = tqdm.trange(n_epochs)
losses = []

for epoch in tqdm_epoch:
  avg_loss = 0
  num_items = 0
  for x, z_coord in dataloader:
    x = x.to(device)
    loss = loss_fn(score_model, x, marginal_prob_std_fn)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    avg_loss += loss.item() * x.shape[0]
    num_items += x.shape[0]
  # Print the averaged training loss so far.
  losses.append(avg_loss / num_items)
  tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

# Saving data
filename = f"batchsize_{batch_size}lr_{lr}_sigma_{sigma}_epochs_{n_epochs}"
torch.save(score_model.state_dict(), filename+'.pth') # only save at the end

plt.plot(range(1, n_epochs + 1), losses, label='Average loss')
plt.xlabel('Epoch')
plt.ylabel('Average loss')
plt.title('Training loss over time')
plt.yscale('log')
plt.savefig(filename +'.png')


## FUNCTIONS

class TumorDataSet(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Only include files, exclude directories
        self.image_files = [f for f in sorted(os.listdir(data_dir)) if os.path.isfile(os.path.join(data_dir, f)) and f.endswith('.tif')]
        # Calculate min and max z-coordinates for normalization
        self.min_z = min(self.extract_z_coordinate(file) for file in self.image_files)
        self.max_z = max(self.extract_z_coordinate(file) for file in self.image_files)

    def __len__(self):
        return len(self.image_files)

    def extract_z_coordinate(self, file_name):
        # Define a regular expression pattern to match the z-coordinate
        pattern = re.compile(r'\d+\.tif$')

        # Use the regular expression to find the match in the file name
        match = pattern.search(file_name)

        # Extract the matched part
        if match:
            return int(match.group(0).replace('.tif', ''))
        else:
            return None

    def normalize_z_coordinate(self, z):
        # Normalize z-coordinate to [0, 1]
        #return (z - self.min_z) / (self.max_z - self.min_z)
        return z

    def __getitem__(self, idx):

        prev_index = max(idx-1, 0)
        next_index = min(idx+1, len(self.image_files)-1)

        z_coordinate = self.extract_z_coordinate(self.image_files[idx])
        normalized_z = self.normalize_z_coordinate(z_coordinate)

        # create paths
        current_path = os.path.join(self.data_dir, self.image_files[idx])
        prev_path = os.path.join(self.data_dir, self.image_files[prev_index])
        next_path = os.path.join(self.data_dir, self.image_files[next_index])

        # load images
        current_image = self.load_img(current_path)
        prev_image = self.load_img(prev_path)
        next_image = self.load_img(next_path)


        # Concatenate images along the channel dimension
        input_images = torch.cat([prev_image, current_image, next_image], dim=0)

        return input_images, normalized_z

    def load_img(self, path):
          image = Image.open(path)
          image = np.array(image)

          image_8bit = ((image - np.min(image)) / (np.max(image) - np.min(image)) * 255).astype(np.uint8)
          image_norm = image_8bit / 255.0
          image_tensor = torch.tensor(image_norm, dtype=torch.float32)
          if self.transform:
              image_tensor = self.transform(image_tensor.unsqueeze(0))

          return image_tensor


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