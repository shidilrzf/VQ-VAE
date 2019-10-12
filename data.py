from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import numpy as np
import random

import torch

import pickle


def get_dataloadr(inlier_classes, is_train, transform, batch_size, use_cuda):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    dataset = MNIST('data', is_train, transform=transform, download=True)

    for cl in inlier_classes:
        idx = dataset.targets == cl
        y = dataset.targets[idx]
        x = dataset.data[idx.numpy().astype(np.bool)]
        if cl == inlier_classes[0]:
            data = x
            label = y
        else:
            data = torch.cat((data, x), 0)
            label = torch.cat((label, y), 0)
    dataset.data = data
    dataset.target = label

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    return dataloader


def get_data(data_dir, folding_id, folds, inlier_classes):
    mnist_train = []
    mnist_valid = []
    for i in range(folds):
        if i != folding_id:
            with open(data_dir + 'data_fold_%d.pkl' % i, 'rb') as pkl:
                fold = pickle.load(pkl)
            if len(mnist_valid) == 0:
                mnist_valid = fold
            else:
                mnist_train += fold
    #keep only train classes
    mnist_train = [x for x in mnist_train if x[0] in inlier_classes]
    mnist_valid = [x for x in mnist_valid if x[0] in inlier_classes]
    random.shuffle(mnist_train)
    random.shuffle(mnist_valid)
    print("Train set size:", len(mnist_train))
    print("Valid set size:", len(mnist_valid))

    mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
    mnist_valid_x, mnist_valid_y = list_of_pairs_to_numpy(mnist_valid)

    return mnist_train_x, mnist_train_y, mnist_valid_x, mnist_valid_y


def list_of_pairs_to_numpy(l):
    return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)
