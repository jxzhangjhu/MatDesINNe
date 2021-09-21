import torch
import numpy as np
import model
from string import ascii_letters
import matplotlib.pyplot as plt
from torch.autograd import Variable
from numpy import sin, cos
import config as c

device = 'cuda' if torch.cuda.is_available() else 'cpu'

filename = 'MoS2_cinn.pkl'
pretrained_net1 = torch.load(filename,map_location=lambda storage, loc: storage)

model_b = model.model
model_b.load_state_dict(pretrained_net1)
# print(model_b)

n_samps = 1000
y0 = 0.5

y_fix = np.zeros((n_samps, 1)) + y0
y_fix = torch.tensor(y_fix, dtype=torch.float)

# y_fix += c.add_y_noise * torch.randn(n_samps, c.ndim_y)
# y_fix = torch.cat([torch.randn(n_samps, c.ndim_z), c.add_z_noise * torch.zeros(n_samps, 0), y_fix], dim=1)
y_fix = y_fix.to(device)

dim_z = 7
z_fix = torch.randn(n_samps, dim_z, device=device)
# z_fix = torch.cat([torch.randn(n_samps, c.ndim_z), y_fix], dim=1)

rev_x0 = model_b(z_fix, y_fix, rev=True).cpu().data.numpy()

# posterior samples
# rev_x0 = model_b(y_fix, rev=True)
# print(rev_x0)

## current backward model to predict
# out_y0 = model_b(rev_x0)
# print(out_y0[:,-1])


## save generative samples
# rev_x = rev_x0.cpu().data.numpy()

fname = 'gen_samps.csv'
np.savetxt(fname, rev_x0, fmt='%.6f', delimiter=',')

print('Generating samples: done!')
