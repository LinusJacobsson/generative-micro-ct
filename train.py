import argparse
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import tqdm
from tqdm import notebook
from functions import ScoreNet, marginal_prob_std, diffusion_coeff, loss_fn, TumorDataSet

# Set up argument parser
parser = argparse.ArgumentParser(description='Tumor Image Training Script')
parser.add_argument('-sigma', type=float, default=25.0, help='Sigma value for the score network')
parser.add_argument('-pixel_size', type=int, default=128, help='Pixel size for image resizing')
parser.add_argument('-epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('-learning_rate', type=float, default=1e-4, help='Learning rate')
parser.add_argument('-batch_size', type=int, default=32, help='Batch size')

# Parse arguments
args = parser.parse_args()

data_dir = '../../cropped_medium_res/'
transform = transforms.Compose([
    transforms.Resize((args.pixel_size, args.pixel_size)),
    transforms.ToTensor(),
])

dataset = TumorDataSet(data_dir, transform)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

sigma = args.sigma
marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)

score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))

n_epochs = args.epochs
batch_size = args.batch_size
lr = args.learning_rate

optimizer = Adam(score_model.parameters(), lr=lr)
tqdm_epoch = tqdm.trange(n_epochs)
for epoch in tqdm_epoch:
    avg_loss = 0.
    num_items = 0
    for x in dataloader:
        loss = loss_fn(score_model, x, marginal_prob_std_fn)
        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
    torch.save(score_model.state_dict(), 'ckpt.pth')
