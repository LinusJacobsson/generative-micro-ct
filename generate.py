#SAMPLING TO GENERATE IMAGES

import torch
import numpy as np
from torchvision.utils import make_grid
from functions import ScoreNet, marginal_prob_std, ode_sampler, diffusion_coeff

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std))
#score_model = score_model.to(device)

## Load the pre-trained checkpoint from disk.
device = 'cpu'
ckpt = torch.load('ckpt.pth', map_location=device)
score_model.load_state_dict(ckpt)

sample_batch_size = 16
sampler = ode_sampler

## Generate samples using the specified sampler.
samples = sampler(score_model,
                  marginal_prob_std,
                  diffusion_coeff,
                  sample_batch_size,
                  device=device)

## Sample visualization.
samples = samples.clamp(0.0, 1.0)
# %matplotlib inline
import matplotlib.pyplot as plt
sample_grid = make_grid(samples, nrow=int(np.sqrt(sample_batch_size)))

plt.figure(figsize=(6,6))
plt.axis('off')
plt.imshow(sample_grid.permute(1, 2, 0).cpu(), vmin=0., vmax=1.)
plt.show()