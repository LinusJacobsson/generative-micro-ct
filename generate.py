#SAMPLING TO GENERATE IMAGE
import argparse
import torch
import numpy as np
from torchvision.utils import make_grid
from functions import ScoreNet, marginal_prob_std, ode_sampler, diffusion_coeff
import functools
from scipy import integrate

# Set up argument parser
parser = argparse.ArgumentParser(description='Tumor Image Generating Script')
parser.add_argument('-device', type=str, default='cpu', help='Choice of device: "cpu" or "cuda"')
parser.add_argument('-weight_file', type=str, default='ckpt.pth', help='Path to model weights')
parser.add_argument('-sigma', type=float, default=25.0, help='Sigma for the PDE, same as when training')
parser.add_argument('-sample_batch_size', type=int, default=9, help='Number of images to be generated')
parser.add_argument('-pixel_size', type=int, default=32, help='Pixel size for image resizing')
parser.add_argument('-epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('-learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
# Parse arguments
args = parser.parse_args()

device = args.device
model_path = args.weight_file
sigma = args.sigma
sample_batch_size = args.batch_size


sampler = ode_sampler
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
## Generate samples using the specified sampler.
score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
score_model = score_model.to(device)

## Load the pre-trained checkpoint from disk.
ckpt = torch.load(model_path, map_location=device)
score_model.load_state_dict(ckpt)

samples = sampler(score_model, 
                  marginal_prob_std_fn,
                  diffusion_coeff_fn, 
                  sample_batch_size, 
                  device=device)
## Sample visualization.
samples = samples.clamp(0.0, 1.0)
# %matplotlib inline
import matplotlib.pyplot as plt
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))
filename = f"size_{args.pixel_size}_lr_{args.learning_rate}_sigma_{args.sigma}_epochs_{args.epochs}_generated"
plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.savefig(filename+'.png')
plt.show()