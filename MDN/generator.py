#!/usr/bin/env python
# coding: utf-8

# In[39]:


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
import numpy as np
device          = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = MixtureDensityNetwork(dim_in=1, dim_out=7, n_components=3)


# Specify a path
PATH = 'MoS2_mdn.pkl'
# Load
model = torch.load(PATH)

## sampling
n_samps = 1000
y0 = 1.0

y_fix = np.zeros((n_samps, 1)) + y0
y_fix = torch.tensor(y_fix, dtype=torch.float)
rev_x = model.sample(y_fix)

fname = 'gen_samps_mdn.csv'
np.savetxt(fname, rev_x, fmt='%.6f', delimiter=',')
