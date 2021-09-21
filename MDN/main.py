import numpy as np
import torch
import torch.optim as optim
import logging
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from models import MixtureDensityNetwork
from torch.autograd import Variable

import pandas as pd
from sklearn.model_selection import train_test_split

device = 'cpu'

rng = 0
batch_size = 500
n_epochs = 1000

## MoS2 - 02-18-2021 dataset
data_x = pd.read_csv('../Simulated_DataSets/MoS2/data_x.csv', header=None).values
data_y = pd.read_csv('../Simulated_DataSets/MoS2/data_y.csv', header=None).values

xtrain, xtest, ytrain, ytest = train_test_split(data_x, data_y, test_size=0.2, random_state=rng)

x_train = torch.tensor(xtrain, dtype=torch.float)
y_train = torch.tensor(ytrain, dtype=torch.float)
x_test = torch.tensor(xtest, dtype=torch.float)
y_test = torch.tensor(ytest, dtype=torch.float)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test),
    batch_size=batch_size, shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=batch_size, shuffle=True, drop_last=True)


model = MixtureDensityNetwork(dim_in=1, dim_out=7, n_components=3)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(n_epochs):
    for j, (x, y) in enumerate(train_loader):
        x, y = Variable(x).to(device), Variable(y).to(device)
        optimizer.zero_grad()
        loss = model.loss(y, x).mean()
        loss.backward()
        optimizer.step()

    if epoch %10==0:
        print(f"epoch: {epoch}, " + f"Loss: {loss.data:.2f}")

# save the model
torch.save(model, 'MoS2_mdn.pkl')
# torch.save(model.state_dict(), 'MoS2_mdn.pkl')
