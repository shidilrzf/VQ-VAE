from __future__ import print_function
import numpy as np
import json
import pickle
import time
import random
import os
import argparse
import torch
import torch.utils.data
from torch import optim


import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image


from torch.optim.lr_scheduler import ReduceLROnPlateau


from tensorboardX import SummaryWriter
from torchsummary import summary
from torchvision import transforms

from data import *
from models import *
from utils import *


parser = argparse.ArgumentParser(description='IBN example')

parser.add_argument('--uid', type=str, default='Vq_vae',
                    help='Staging identifier (default: IBN)')

# Device (GPU)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables cuda (default: False')

parser.add_argument('--seed', type=int, default=1,
                    help='Seed for numpy and pytorch (default: None')

# Model
parser.add_argument('--z-dim', type=int, default=64, metavar='N',
                    help='Latent size (default: 20')

parser.add_argument('--clusters', type=int, default=512, metavar='N',
                    help='Number of clusters (default: 100')

# Optimizer
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='number of training epochs')

parser.add_argument('--lr', type=float, default=3e-4,
                    help='Learning rate (default: 2e-4')

parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input training batch-size')

# data loader parameters
parser.add_argument('--dataset-name', type=str, default='mnist',
                    help='Name of dataset (default: MNIST')

# Log directory
parser.add_argument('--log-dir', type=str, default='runs',
                    help='logging directory (default: runs)')

args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

use_cuda = not args.no_cuda and torch.cuda.is_available()
print(use_cuda)
torch.cuda.empty_cache()

torch.manual_seed(args.seed)
if use_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


# model parameters
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2
z_dim = args.z_dim


# optimization parameters
commitment_cost = 0.25
decay = 0.99


# data perperation

data_dir = 'data'
download_data = True

transform = transforms.Compose([transforms.Resize([32, 32]), transforms.ToTensor()])
inlier_classes = [0]
batch_size = args.batch_size
train_dataloader = get_dataloadr(inlier_classes, True, transform, args.batch_size, use_cuda)
val_dataloader = get_dataloadr(inlier_classes, False, transform, args.batch_size, use_cuda)
val_dataloader_outlier = get_dataloadr(outlier_classes, False , transform, args.batch_size, use_cuda)


# Set tensorboard
log_dir = args.log_dir

# dir, args.uid, timestamp
logger = SummaryWriter(comment='_' + args.uid + '_' + args.dataset_name)


# Define model
model = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
               args.clusters, z_dim,
               commitment_cost, device, decay).to(device)


# optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=False)
scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)


def train_validate(model, loader, optimizer, train, device):
    model.train() if train else model.eval()

    total_loss = 0
    total_recon_loss = 0
    total_vq_loss = 0
    total_perplexity = 0

    for batch_idx, (x, y) in enumerate(loader):
        loss = 0
        x = x.to(device)
        if train:
            optimizer.zero_grad()

        vq_loss, x_recon, perplexity = model(x)

        recon_loss = torch.mean((x_recon - x)**2)
        loss = recon_loss + vq_loss

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss.item()
        total_perplexity += perplexity.item()

        if train:
            loss.backward()
            optimizer.step()
    return total_loss / len(loader.dataset), total_recon_loss / len(loader.dataset), total_vq_loss / len(loader.dataset), total_perplexity / len(loader.dataset)


def execute_graph(model, train_dataloader, val_dataloader, optimizer, scheduler, device):
    # Training loss
    t_loss, t_recon_loss, t_vq_loss, t_prxl = train_validate(model, train_dataloader, optimizer, True, device)

    # Validation loss
    v_loss, v_recon_loss, v_vq_loss, v_prxl = train_validate(model, val_dataloader, optimizer, False, device)
    print('====> Epoch: {}..................'.format(epoch))
    print('====> Train: Average  loss: {:.4f}'.format(t_loss))
    print('====> Train: Average Recon loss: {:.4f}'.format(t_recon_loss))
    print('====> Train: Average VQ loss: {:.4f}'.format(t_vq_loss))
    print('====> Train: Average Perpexility: {:.4f}'.format(t_prxl))

    print('====> Valid: Average Valid loss: {:.4f}'.format(v_loss))
    print('====> Valid: Average Recon loss: {:.4f}'.format(v_recon_loss))
    print('====> Valid: Average VQ loss: {:.4f}'.format(v_vq_loss))
    print('====> Valid: Average Perpexility: {:.4f}'.format(v_prxl))

    print('================================================================>')

    # Training and validation loss
    logger.add_scalar(log_dir + '/validation-loss', v_loss, epoch)
    logger.add_scalar(log_dir + '/training-loss', t_loss, epoch)

#    # image generation examples
#    sample = generation_example(model, args.z_dim, data_loader.train_loader, input_shape, num_class, use_cuda)
#    sample = sample.detach()
#    sample = tvu.make_grid(sample, normalize=False, scale_each=True)
#    logger.add_image('generation example', sample, epoch)
#
#    # image reconstruction examples
    comparison = reconstrct_images(model, val_dataloader, device)
    comparison = comparison.detach()
    comparison = tvu.make_grid(comparison, normalize=False, scale_each=True)
    save_img(comparison, 'results/reconstruction example_{}.png'.format(epoch))
    logger.add_image('reconstruction example', comparison, epoch)
    
    if epoch % 2 == 0:
        directory = 'results'
        if not os.path.exists(directory):
            os.makedirs(directory)
        (x_inlier, _) = next(iter(val_dataloader))
        (x_outlier, _) = next(iter(val_dataloader_outlier))
        
        mse_score_inlier = get_mse_score(model, x_inlier, device)
        mse_score_outlier = get_mse_score(model, x_outlier, device)
        filename = 'results/outlier_vs_inlier_{}'.format(epoch)
        plot_mse_outliers(mse_score_inlier, mse_score_outlier, filename)

    scheduler.step(v_loss)

    return v_loss


num_epochs = args.epochs
best_loss = np.inf

# Main training and validation loop
val_losses = []

for epoch in range(1, num_epochs + 1):
    v_loss = execute_graph(model, train_dataloader, val_dataloader, optimizer, scheduler, device)
    val_losses.append(v_loss)
    if v_loss < best_loss:
        best_loss = v_loss
        print('Writing model checkpoint')
        save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': model.state_dict(),
                        'val_loss': v_loss
                        },
                        'models/' + args.uid + '_' + args.dataset_name + '.pt')
