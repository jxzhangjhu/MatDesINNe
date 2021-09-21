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

n_samps = 1000
y0 = 1.0

y_fix = np.zeros((n_samps, 1)) + y0
y_fix = torch.tensor(y_fix, dtype=torch.float)
y_fix = y_fix.to(device)

dim_z = 7
z_fix = torch.randn(n_samps, dim_z, device=device)
rev_x0 = model_b(z_fix, y_fix, rev=True).cpu().data.numpy()

fname = 'gen_samps.csv'
np.savetxt(fname, rev_x0, fmt='%.6f', delimiter=',')
