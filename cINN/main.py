from time import time

import torch
from torch.autograd import Variable

from FrEIA.framework import *
from FrEIA.modules import *

import config as c
import losses
import model

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

cINN_method = 1

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
    batch_size=c.batch_size, shuffle=False, drop_last=True)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=c.batch_size, shuffle=True, drop_last=True)

# training function
def train_epoch():
    """
    The major training function. This would start the training using information given in the flags
    :return: None
    """
    print(model.model)
    print("Starting training now")

    for epoch in range(c.n_epochs):
        # Set to Training Mode
        train_loss = 0
        train_mse_y_loss = 0
        train_mmd_x_loss = 0

        model.model.train()
        loss_factor = min(1., 2. * 0.002 ** (1. - (float(epoch) / c.n_epochs)))

        train_loss_history = []
        for j, (x, y) in enumerate(train_loader):

            batch_losses = []

            x, y = Variable(x).to(c.device), Variable(y).to(c.device)

            ######################
            #  Forward step      #
            ######################
            model.optim.zero_grad()

            if cINN_method:
                z = model.model(x, y)
                zz = torch.sum(z**2, dim=1)
                jac = model.model.log_jacobian(run_forward=False)                # get the log jacobian
                neg_log_likeli = 0.5 * zz - jac
                loss_total= torch.mean(neg_log_likeli)
                loss_total.backward()

            ######################
            #  Gradient Clipping #
            ######################
            for parameter in model.model.parameters():
                parameter.grad.data.clamp_(-c.grad_clamp, c.grad_clamp)

            #########################
            # Descent your gradient #
            #########################
            model.optim.step()  # Move one step the optimizer

            # MLE training
            train_loss += loss_total

        # Calculate the avg loss of training
        train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)

        if epoch % c.eval_test == 0:
            model.model.eval()
            print("Doing Testing evaluation on the model now")

            test_loss = 0
            test_mse_y_loss = 0
            test_mmd_x_loss = 0

            test_loss_history = []
            for j, (x, y) in enumerate(test_loader):

                batch_losses = []

                x, y = Variable(x).to(c.device), Variable(y).to(c.device)

                # ######################
                # #  Forward step      #
                # ######################
                model.optim.zero_grad()
                #
                z = model.model(x, y)
                zz = torch.sum(z**2, dim=1)
                jac = model.model.log_jacobian(run_forward=False)                # get the log jacobian
                neg_log_likeli = 0.5 * zz - jac
                loss_total = torch.mean(neg_log_likeli)

                # MLE training
                # print(loss_total)
                test_loss += loss_total

            # Calculate the avg loss of training
            test_avg_loss = test_loss.cpu().data.numpy() / (j + 1)
            print("This is Epoch %d, training loss %.5f, testing loss %.5f" % (epoch, train_avg_loss, test_avg_loss))

        model.scheduler_step()


if __name__ == "__main__":
    train_epoch()
    torch.save(model.model.state_dict(), 'MoS2_cinn.pkl')
