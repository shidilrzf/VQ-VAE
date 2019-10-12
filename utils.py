import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import torchvision.utils as tvu
from torch.autograd import Variable
import matplotlib.pyplot as plt


def generate_images(generator, centers, num_clusters, alpha, z_dim, device):
    idx_centers = torch.from_numpy(np.random.choice(np.arange(num_clusters), 16))
    eps = torch.FloatTensor(16, z_dim).uniform_(-alpha, alpha).to(device)
    noise = centers[idx_centers] + eps
    num_images = noise.shape[0]
    rows = int(math.sqrt(num_images))

    images = generator(noise).cpu().detach()
    grid_img = tvu.make_grid(images, nrow=rows)
    return grid_img


def reconstrct_images(model, dataloader, device):
    model.eval()

    (x, _) = next(iter(dataloader))
    x = x.to(device)
    x_pre_vq = model._pre_vq_conv(model._encoder(x))
    _, x_quantize, _, _ = model._vq_vae(x_pre_vq)
    x_hat = model._decoder(x_quantize).cpu().detach()

    #grid_img = tvu.make_grid(x_hat, nrow=rows)
    x = x[:10].cpu().view(10 * 32, 32)
    x_hat = x_hat[:10].cpu().view(10 * 32, 32)
    comparison = torch.cat((x, x_hat), 1).view(10 * 32, 2 * 32)
    return comparison


def type_tdouble(use_cuda=False):
    return torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)


def one_hot(labels, n_class, use_cuda=False):
    # Ensure labels are [N x 1]
    if len(list(labels.size())) == 1:
        labels = labels.unsqueeze(1)
    mask = type_tdouble(use_cuda)(labels.size(0), n_class).fill_(0)
    # scatter dimension, position indices, fill_value
    return mask.scatter_(1, labels, 1)


def to_cuda(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.cuda()
    return tensor


def conv_size(H_in, k_size, stride, padd, dil=1):
    H_out = np.floor((H_in + 2 * padd - dil * (k_size - 1) - 1) / stride + 1)
    return np.int(H_out)


def shuffle(X):
    np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)


def numpy2torch(x):
    return torch.from_numpy(x)


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size, :, :]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)


def plot_scatter_outliers(mse_score_inlier, discriminator_score_inlier, mse_score_outlier, discriminator_score_outlier, epoch):

    plt.scatter(mse_score_inlier, discriminator_score_inlier)
    plt.scatter(mse_score_outlier, discriminator_score_outlier)
    plt.xlabel('MSE_distance')
    plt.ylabel('Discriminator_distance')
    #plt.legend()
    plt.grid(True)
    plt.savefig('results/inlier_vs_outlier_{}.png'.format(epoch))
    plt.close()

def get_mse_score(model, x, device):
    N = x.size(0)
    x = x.to(device)
    _, x_hat, _ = model(x)
    x = x.squeeze().cpu().detach().numpy()
    x_hat = x_hat.squeeze().cpu().detach().numpy()
    
    mse_score= []
    for i in range(N):
        distance = np.sum(np.power(x_hat[i].flatten() - x[i].flatten(), 2.0))
        mse_score.append(distance)
    
    return mse_score

def plot_mse_outliers(mse_score_inlier, mse_score_outlier, filename):
    
    plt.hist(mse_score_inlier, 10, density=1, facecolor='g', alpha=0.75)
    plt.hist(mse_score_outlier, 10, density=1, facecolor='r', alpha=0.75)
    plt.xlabel('MSE_distance')
    plt.ylabel('Histogram')
    #plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def save_checkpoint(state, filename):
    torch.save(state, filename)


def save_img(img, filename):
    npimg = img.numpy()

    fig = plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(filename)
    plt.close()
