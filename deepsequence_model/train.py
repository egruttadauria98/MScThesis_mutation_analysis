import numpy as np
import torch
from torch import nn, optim

from blitz.modules.base_bayesian_module import BayesianModule

from sklearn.model_selection import train_test_split

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


def compute_loss(model, data, X_data, y_data, epoch, num_updates, kl_latent_scale, kl_weights_scale):
    # MSE for recostruction loss
    mse = nn.MSELoss()

    # Forward method for the model
    reconstructed_x, mu, logvar = model(X_data)

    # Compute MSE (instead of negative log likelihood)
    mse_loss =  mse(y_data.view(-1, data.seq_len * data.alphabet_size), reconstructed_x)

    # Compute KL latent under normal prior
    # TODO: make this a method of the model
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

    return loss, mse_loss, kl_latent, kl_weights
        

def train(data,
          model,
          kl_latent_scale=1.0,
          kl_weights_scale=1.0,
          lr=1e-3,
          batch_size=100,
          num_updates=300000):


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr)

    # Set the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Send model to device
    model = model.to(device)

    X_test = torch.Tensor(data.X_test).to(device)
    y_test = torch.Tensor(data.y_test).to(device)

    track_epoch_vars = dict()
    track_epoch_vars['mse_loss'] = []
    track_epoch_vars['kl_latent'] = []
    track_epoch_vars['kl_weights'] = []
    track_epoch_vars['train_loss'] = []
    track_epoch_vars['validation_loss'] = []

    for epoch in range(num_updates):

        optimizer.zero_grad()

        # Create the batch indices
        # If weights of the sequences are not isotropic, replace p with the probability distribution
        batch_order = np.arange(data.X_train.shape[0])
        batch_index = np.random.choice(batch_order, batch_size, p=None).tolist() 

        # Create the batch tensor and send them to the device
        X_batch = torch.Tensor(data.X_train[batch_index]).to(device)
        y_batch = torch.Tensor(data.y_train[batch_index]).to(device)

        train_loss, mse_loss, kl_latent, kl_weights = compute_loss(model, data, X_batch, y_batch, epoch, num_updates, kl_latent_scale, kl_weights_scale)

        # Backpropagation and optimization
        train_loss.backward()
        optimizer.step()
        
        if epoch%100==0:
            print(f"\n\nEpoch: {epoch}")
            print(f"\tMSE loss: {mse_loss}")
            print(f"\tKL latent loss: {kl_latent}")
            print(f"\tKL weights loss: {kl_weights}")
            print(f"\tTotal loss:{train_loss}")

            # Validation loss
            validation_loss, _, _, _ = compute_loss(model, data, X_test, y_test, epoch, num_updates, kl_latent_scale, kl_weights_scale)

            print(f"\n\tValidation loss: {validation_loss}")

            track_epoch_vars['mse_loss'].append(mse_loss.cpu().detach().numpy())
            track_epoch_vars['kl_latent'].append(kl_latent.cpu().detach().numpy())

            try:
                track_epoch_vars['kl_weights'].append(kl_weights.cpu().detach().numpy())
            except AttributeError:
                track_epoch_vars['kl_weights'].append(kl_weights)

            track_epoch_vars['train_loss'].append(train_loss.cpu().detach().numpy())
            track_epoch_vars['validation_loss'].append(validation_loss.cpu().detach().numpy())

    return model, track_epoch_vars




    