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

rng = 0
data_x = pd.read_csv('../../Simulated_DataSets/MoS2/data_x.csv', header=None).values
data_y = pd.read_csv('../../Simulated_DataSets/MoS2/data_y.csv', header=None).values

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

# ICLR or NIPS
ICLR_method = 1


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
        # If MMD on x-space is present from the start, the model can get stuck.
        # Instead, ramp it up exponetially.
        loss_factor = min(1., 2. * 0.002 ** (1. - (float(epoch) / c.n_epochs)))

        train_loss_history = []
        for j, (x, y) in enumerate(train_loader):

            batch_losses = []

            x, y = Variable(x).to(c.device), Variable(y).to(c.device)

            ######################
            #  data pad cat     #
            ######################
            # x_pad and new x, ndim = x_pad + x
            x_pad = c.add_pad_noise * torch.randn(c.batch_size, c.ndim_pad_x).to(c.device)
            x = torch.cat((x, x_pad), dim=1)

            # yz_pad，z_pad,y
            yz_pad = c.add_pad_noise * torch.randn(c.batch_size, c.ndim_pad_zy).to(c.device)
            z_pad = torch.randn(c.batch_size, c.ndim_z).to(c.device)
            y = torch.cat((z_pad, yz_pad, y), dim=1)

            ######################
            #  Forward step      #
            ######################
            model.optim.zero_grad()
            out_y = model.model(x)
            out_x = model.model(y, rev=True)

            if ICLR_method:
                if c.train_max_likelihood:
                    batch_losses.append(losses.loss_max_likelihood(out_y, y))

                if c.train_forward_mmd:
                    batch_losses.extend(losses.loss_forward_fit_mmd(out_y, y))

                if c.train_backward_mmd:
                    batch_losses.append(losses.loss_backward_mmd(x, y))

                if c.train_reconstruction:
                    batch_losses.append(losses.loss_reconstruction(out_y.data, y, x))

                loss_total = sum(batch_losses)
                train_loss_history.append([l.item() for l in batch_losses])
                # print('training losses', np.mean(loss_history, axis=0))

            else:
                # Do the MSE loss for reconstruction, Doesn't compare z part (only pad and y itself)
                MSE_loss_y = losses.l2_fit(out_y[:, c.ndim_z:], y[:, c.ndim_z:])

                # Use the maximum likelihood method
                log_det = model.model.log_jacobian(x=x)
                # print("The log determinant is", log_det)
                loss_total = 0.5 * (MSE_loss_y / c.lambda_mse + torch.mean(torch.pow(z_pad, 2))) - torch.mean(log_det)

            # original y and x
            mse_y_loss = losses.l2_fit(out_y[:, -c.ndim_y:], y[:, -c.ndim_y:])
            mmd_x_loss = torch.mean(losses.backward_mmd(x[:, :c.ndim_x], out_x[:, :c.ndim_x]))

            # cat y and x
            # mse_y_loss = losses.l2_fit(out_y[:, c.ndim_z:], y[:, c.ndim_z:])
            # mmd_x_loss = torch.mean(losses.backward_mmd(x, out_x))


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
            train_mse_y_loss += mse_y_loss
            train_mmd_x_loss += mmd_x_loss

        # Calculate the avg loss of training
        train_avg_loss = train_loss.cpu().data.numpy() / (j + 1)
        train_avg_mse_y_loss = train_mse_y_loss.cpu().data.numpy() / (j + 1)
        train_avg_mmd_x_loss = train_mmd_x_loss.cpu().data.numpy() / (j + 1)

        # print(np.shape(train_avg_mmd_x_loss)) # batch_size x batch_size
        # print('training', np.mean(loss_history, axis=0))

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

                ######################
                #  data pad cat     #
                ######################
                # x_pad and new x, ndim = x_pad + x
                x_pad = c.add_pad_noise * torch.randn(c.batch_size, c.ndim_pad_x).to(c.device)
                x = torch.cat((x, x_pad), dim=1)

                # yz_pad，z_pad,y
                yz_pad = c.add_pad_noise * torch.randn(c.batch_size, c.ndim_pad_zy).to(c.device)
                # print('yz_pad',yz_pad)
                z_pad = torch.randn(c.batch_size, c.ndim_z).to(c.device)
                y = torch.cat((z_pad, yz_pad, y), dim=1)

                ######################
                #  Forward step      #
                ######################
                model.optim.zero_grad()
                out_y = model.model(x)
                out_x = model.model(y, rev=True)

                if ICLR_method:
                    if c.train_max_likelihood:
                        batch_losses.append(losses.loss_max_likelihood(out_y, y))

                    if c.train_forward_mmd:
                        batch_losses.extend(losses.loss_forward_fit_mmd(out_y, y))

                    if c.train_backward_mmd:
                        batch_losses.append(losses.loss_backward_mmd(x, y))

                    if c.train_reconstruction:
                        batch_losses.append(losses.loss_reconstruction(out_y.data, y, x))

                    loss_total = sum(batch_losses)
                    test_loss_history.append([l.item() for l in batch_losses])
                    # print('loss_total', loss_total)
                    # print('testing losses',np.mean(loss_history, axis=0))

                else:
                    # Do the MSE loss for reconstruction, Doesn't compare z part (only pad and y itself)
                    MSE_loss_y = losses.l2_fit(out_y[:, c.ndim_z:], y[:, c.ndim_z:])

                    # Use the maximum likelihood method
                    log_det = model.model.log_jacobian(x=x)
                    # print("The log determinant is", log_det)
                    loss_total = 0.5 * (MSE_loss_y / c.lambda_mse + torch.mean(torch.pow(z_pad, 2))) - torch.mean(log_det)

                # original y and x
                mse_y_loss = losses.l2_fit(out_y[:, -c.ndim_y:], y[:, -c.ndim_y:])
                mmd_x_loss = torch.mean(losses.backward_mmd(x[:, :c.ndim_x], out_x[:, :c.ndim_x]))

                # cat y and x
                # mse_y_loss = losses.l2_fit(out_y[:, c.ndim_z:], y[:, c.ndim_z:])
                # mmd_x_loss = torch.mean(losses.backward_mmd(x, out_x))

                # MLE training
                # print(loss_total)
                test_loss += loss_total
                test_mse_y_loss += mse_y_loss
                test_mmd_x_loss += mmd_x_loss

            # print('true', x[0, :c.ndim_x])
            # print('generate', out_x[0, :c.ndim_x])

            # Calculate the avg loss of training
            test_avg_loss = test_loss.cpu().data.numpy() / (j + 1)
            test_avg_mse_y_loss = test_mse_y_loss.cpu().data.numpy() / (j + 1)
            test_avg_mmd_x_loss = test_mmd_x_loss.cpu().data.numpy() / (j + 1)

            print('training losses', np.mean(train_loss_history, axis=0))
            print('testing losses', np.mean(test_loss_history, axis=0))

            print("This is Epoch %d, training loss %.5f, testing loss %.5f" % (epoch, train_avg_loss, test_avg_loss))
            print("This is Epoch %d, training mse y loss %.5f, testing mse y loss %.5f" % (epoch, train_avg_mse_y_loss, test_avg_mse_y_loss))
            print("This is Epoch %d, training mmd x loss %.5f, testing mmd x loss %.5f" % (epoch, train_avg_mmd_x_loss, test_avg_mmd_x_loss))

        model.scheduler_step()


if __name__ == "__main__":
    train_epoch()
    torch.save(model.model.state_dict(), 'MoS2_backward.pkl')
