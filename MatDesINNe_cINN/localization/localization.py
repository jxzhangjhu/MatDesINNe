import torch
import numpy as np
import model
from string import ascii_letters
import matplotlib.pyplot as plt
from torch.autograd import Variable
from numpy import sin, cos
import config as c

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filename = 'MoS2_forward.pkl'
pretrained_net0 = torch.load(filename, map_location=lambda storage, loc: storage)

model_f = model.model
model_f.load_state_dict(pretrained_net0)

## direct posterior samples
gen_samps = np.loadtxt('../generation/gen_samps.csv',delimiter=',')
gen_samps = torch.tensor(gen_samps, dtype=torch.float)
rev_x1 = gen_samps
rev_x1 = Variable(gen_samps, requires_grad=True)
out_y1 = model_f(rev_x1)
print(out_y1[:,-1])

## identify effective samples with a specific target y and delta
y0 = 0.5
delta = 0.1
eff_list = []

for i in range(len(out_y1[:,-1])):

    rev_x1_7 = rev_x1.cpu().data.numpy()
    if out_y1[i,-1]<y0+delta and out_y1[i,-1] > y0-delta and  rev_x1_7[i,-1]<1 and rev_x1_7[i,-1]>-1:
        eff_list.append(i)

rev_x1_eff = rev_x1[eff_list,:]
print('effect sample size',rev_x1_eff.size())

out_y1_eff = model_f(rev_x1_eff)
print('effect sample pred',out_y1_eff[:,-1])

fname0 = 'effective_samples.csv'
np.savetxt(fname0, rev_x1_eff.cpu().data.numpy(), fmt='%.6f', delimiter=',')

fname1 = 'effective_samples_y.csv'
np.savetxt(fname1, out_y1_eff[:,-1].cpu().data.numpy(), fmt='%.6f', delimiter=',')

## localization
n_iter = 2000
lr = 0.01
## graident-based optimization
for i in range(n_iter):

    rev_x1_eff = Variable(rev_x1_eff, requires_grad=True)
    out_y1 = model_f(rev_x1_eff)
    re_loss = torch.mean((out_y1[:,-1] - y0)**2)
    re_loss.backward()

    rev_x1_eff_new = rev_x1_eff - lr*rev_x1_eff.grad
    rev_x1_eff = rev_x1_eff_new

    print(re_loss)


out_y1_eff_loc = model_f(rev_x1_eff)
print('effect sample loc pred',out_y1_eff_loc[:,-1])

fname2 = 'effective_samples_loc.csv'
np.savetxt(fname2, rev_x1_eff.cpu().data.numpy(), fmt='%.6f', delimiter=',')

fname3 = 'effective_samples_y_loc.csv'
np.savetxt(fname3, out_y1_eff_loc[:,-1].cpu().data.numpy(), fmt='%.6f', delimiter=',')
