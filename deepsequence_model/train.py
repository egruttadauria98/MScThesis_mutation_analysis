import numpy as np
import torch
from torch import nn, optim

from blitz.modules.base_bayesian_module import BayesianModule

def compute_kl_weights(model):
    kl_weights = 0
    for module in model.modules():
        if isinstance(module, (BayesianModule)):
            kl_weights += module.log_variational_posterior - module.log_prior
    return kl_weights


def quadratic_anneal(epoch, num_updates):
    if epoch < num_updates/2:
        kl_scale_n = (2*epoch/num_updates)**2
    else:
        kl_scale_n = 1.0
    
    return kl_scale_n
        

def train(data,
          model,
          finetune=False,
          kl_latent_scale=1.0,
          kl_weights_scale=1.0,
          lr=1e-3,
          batch_size=100,
          num_updates=300000):


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr)

    # MSE for recostruction loss
    mse = nn.MSELoss()

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Send model to device
    model = model.to(device)

    for epoch in range(num_updates):

        optimizer.zero_grad()

        # The output is always the one-hot encoding of the amino acids in the sequence
        y = data.x_train

        if not finetune:
            # Normally, the input is the same as the output
            X = y
        else:
            # Finetuning is carried out using the MSA representation from AlhaFold
            X = data.x_train_finetune

        # Create the batch indices
        # If weights of the sequences are not isotropic, replace p with the probability distribution
        batch_order = np.arange(X.shape[0])
        batch_index = np.random.choice(batch_order, batch_size, p=None).tolist() 

        # Create the batch tensor and send them to the device
        X_batch = torch.Tensor(X[batch_index]).to(device)
        y_batch = torch.Tensor(y[batch_index]).to(device)

        # Forward method for the model
        reconstructed_x, mu, logvar = model(X_batch)

        # Compute MSE (instead of negative log likelihood)
        mse_loss =  mse(y_batch.view(-1, data.seq_len * data.alphabet_size), reconstructed_x)

        # Compute KL latent under normal prior
        kl_latent = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        # Compute KL weights on Bayesian layers
        kl_weights = compute_kl_weights(model)

        # Annealing KL latent
        if kl_latent_scale == 'anneal':
            kl_latent_scale_n = quadratic_anneal(epoch, num_updates)
        else:
            kl_latent_scale_n = kl_latent_scale

        # Annealing KL weights
        if kl_weights_scale == 'anneal':
            kl_weights_scale_n = quadratic_anneal(epoch, num_updates)
        else:
            kl_weights_scale_n = kl_weights_scale

        # Total loss
        loss = mse_loss + kl_latent_scale_n*kl_latent + kl_weights_scale_n*kl_weights

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()
        
        if epoch%100==0:
            print(f"Epoch: {epoch}")
            print(f"\tMSE loss: {mse_loss}")
            print(f"\tKL latent loss: {kl_latent}")
            print(f"\tKL weights loss: {kl_weights}")
            print(f"\tTotal loss:{loss}")

    return model




    