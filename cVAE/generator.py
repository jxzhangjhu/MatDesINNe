
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 500         # number of data points in each batch
N_EPOCHS = 1000           # times to run the model on complete data
INPUT_DIM = 7     # size of each input
HIDDEN_DIM = 128        # hidden dimension
LATENT_DIM = 3         # latent vector dimension
N_CLASSES = 1          # number of classes in the data
lr = 1e-3               # learning rate

## MoS2 - 02-18-2021 dataset
rng = 0
data_x = pd.read_csv('../Simulated_DataSets/MoS2/data_x.csv', header=None).values
data_y = pd.read_csv('../Simulated_DataSets/MoS2/data_y.csv', header=None).values

xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=0.2, random_state=rng)

x_train = torch.tensor(xtrain, dtype=torch.float)
y_train = torch.tensor(ytrain, dtype=torch.float)
x_test = torch.tensor(xtest, dtype=torch.float)
y_test = torch.tensor(ytest, dtype=torch.float)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

def idx2onehot(idx, n=N_CLASSES):

    assert idx.shape[1] == 1
    assert torch.max(idx).item() < n

    onehot = torch.zeros(idx.size(0), n)
    onehot.scatter_(1, idx.data, 1)

    return onehot


class Encoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim + n_classes, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x is of shape [batch_size, input_dim + n_classes]

        hidden = F.relu(self.linear(x))
        # hidden is of shape [batch_size, hidden_dim]

        # latent parameters
        mean = self.mu(hidden)
        # mean is of shape [batch_size, latent_dim]
        log_var = self.var(hidden)
        # log_var is of shape [batch_size, latent_dim]

        return mean, log_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self, latent_dim, hidden_dim, output_dim, n_classes):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.latent_to_hidden = nn.Linear(latent_dim + n_classes, hidden_dim)
        self.hidden_to_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x is of shape [batch_size, latent_dim + num_classes]
        x = F.relu(self.latent_to_hidden(x))
        # x is of shape [batch_size, hidden_dim]
#         generated_x = F.sigmoid(self.hidden_to_out(x))
        generated_x = self.hidden_to_out(x)
        # x is of shape [batch_size, output_dim]

        return generated_x


class CVAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.
    '''
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden dimension.
            latent_dim: A integer indicating the latent size.
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()

        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, n_classes)

    def forward(self, x, y):

        x = torch.cat((x, y), dim=1)

        # encode
        z_mu, z_var = self.encoder(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        z = torch.cat((x_sample, y), dim=1)

        # decode
        generated_x = self.decoder(z)

        return generated_x, z_mu, z_var


def calculate_loss(x, reconstructed_x, mean, log_var):
    # reconstruction loss
    RCL = F.mse_loss(reconstructed_x, x, size_average=False)
    # kl divergence loss
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return RCL + KLD

model = CVAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, N_CLASSES)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)


def train():
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i, (x, y) in enumerate(train_loader):
        # reshape the data into [batch_size, 784]

#         x = x.view(-1, 28 * 28)
        x = x.to(device)
        # convert y into one-hot encoding
#         y = idx2onehot(y.view(-1, 1))
        y = y.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        reconstructed_x, z_mu, z_var = model(x, y)

        # loss
        loss = calculate_loss(x, reconstructed_x, z_mu, z_var)

        # backward pass
        loss.backward()
        train_loss += loss.item()

        # update the weights
        optimizer.step()

    return train_loss


def test():
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            # reshape the data
#             x = x.view(-1, 28 * 28)
            x = x.to(device)

            # convert y into one-hot encoding
#             y = idx2onehot(y.view(-1, 1))
            y = y.to(device)

            # forward pass
            reconstructed_x, z_mu, z_var = model(x, y)

            # loss
            loss = calculate_loss(x, reconstructed_x, z_mu, z_var)
            test_loss += loss.item()

    return test_loss



# Specify a path
PATH = 'MoS2_cvae.pt'

# Save
# torch.save(model, PATH)

# Load
model = torch.load(PATH)
model.to(device)
model.eval()

y0 = 1.0
num_z = 1000
rev_x = np.zeros((num_z,7))

for i in range(num_z):
    z = torch.randn(1, LATENT_DIM).to(device)
    y = torch.tensor([[y0]]).to(device)
    z = torch.cat((z, y), dim=1)
    reconstructed_x = model.decoder(z)
    rev_x[i,:] = reconstructed_x.cpu().data.numpy()
#     print(reconstructed_x)

fname = 'gen_samps_cvae.csv'
np.savetxt(fname, rev_x, fmt='%.6f', delimiter=',')
