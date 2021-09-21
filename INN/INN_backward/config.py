import torch
######################
#  General settings  #
######################

lambda_mse = 0.1
grad_clamp = 15
eval_test = 10 # crystal small data issue, we have to use smaller one

# Compute device to perform the training on, 'cuda' or 'cpu'
device          = 'cuda' if torch.cuda.is_available() else 'cpu'

#######################
#  Training schedule  #
#######################

# Initial learning rate
lr_init         = 1.0e-3
#Batch size
batch_size      = 500
# Total number of epochs to train for
n_epochs        = 1000
# End the epoch after this many iterations (or when the train loader is exhausted)
pre_low_lr      = 0
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
final_decay     = 0.02
# L2 weight regularization of model parameters
l2_weight_reg   = 1e-5
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

#####################
#  Data dimensions  #
#####################

ndim_x     = 7
ndim_pad_x = 0

ndim_y     = 1
ndim_z     = 6
ndim_pad_zy = 0

train_forward_mmd    = True
train_backward_mmd   = True
train_reconstruction = False
train_max_likelihood = False

lambd_fit_forw       = 1
lambd_mmd_forw       = 1
lambd_reconstruct    = 1
lambd_mmd_back       = 100
lambd_max_likelihood = 1

# Both for fitting, and for the reconstruction, perturb y with Gaussian
# noise of this sigma
add_y_noise     = 5e-3
# For reconstruction, perturb z
add_z_noise     = 2e-3
# In all cases, perturb the zero padding
add_pad_noise   = 1e-3
# MLE loss
zeros_noise_scale = 5e-3

# For noisy forward processes, the sigma on y (assumed equal in all dimensions).
# This is only used if mmd_back_weighted of train_max_likelihoiod are True.
y_uncertainty_sigma = 0.12 * 4

mmd_forw_kernels = [(0.2, 2), (1.5, 2), (3.0, 2)]
mmd_back_kernels = [(0.2, 0.1), (0.2, 0.5), (0.2, 2)]
mmd_back_weighted = False

###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.10
#
N_blocks   = 6
#
exponent_clamping = 2.0
#
hidden_layer_sizes = 256
#
use_permutation = True
#
verbose_construction = False
