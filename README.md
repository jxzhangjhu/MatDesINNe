# iPage+ for MoS2 dataset

## Trained model

Forward model is saved in forward/MoS2_forward.pkl

The config parameters are uses
lambd_fit_forw       = 10

lambd_mmd_forw       = 1

lambd_reconstruct    = 1

lambd_mmd_back       = 1

lambd_max_likelihood = 1

Backward model is saved in forward/MoS2_back.pkl

The config parameters are uses
lambd_fit_forw       = 1

lambd_mmd_forw       = 1

lambd_reconstruct    = 1

lambd_mmd_back       = 10

lambd_max_likelihood = 1


## Generative sampling

Run MoS2_generative_baseline.ipynb for generating candidate samples


## Localization step

Run MoS2_localization_baseline.ipynb for optimizing the generated samples using gradient descent
