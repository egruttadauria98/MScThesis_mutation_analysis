from torch import nn, optim
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

@variational_estimator
class VAE(nn.Module):

    def __init__(self,
                 data,
                 encoder_architecture=[1500,1500], 
                 decoder_architecture=[100, 2000], 
                 n_latent=40,
                 bayesian=False):

        super(VAE, self).__init__()

        # VAE model parameters 
        self.encoder_architecture = encoder_architecture
        self.decoder_architecture = decoder_architecture
        self.n_latent = n_latent

        # Get data input size for the first layer of the VAE
        self.seq_len = data.seq_len
        self.alphabet_size = data.alphabet_size
        
        # TODO: delete this after debugging
        self.n_eff = data.n_eff

        # If bayesian=True, creates weight distribution for the decoder
        # Default behavious is vanilla VAE
        self.bayesian = bayesian

        # Instantiate the layers of the VAE
        self._create_layers()


    def _create_layers(self):

        # Encoder
        self.fc1 = nn.Linear(self.seq_len * self.alphabet_size, self.encoder_architecture[0])
        self.fc2 = nn.Linear(self.encoder_architecture[0], self.encoder_architecture[1])

        # Bottleneck hiddent representation of mu and logvar
        self.fc3_mu = nn.Linear(self.encoder_architecture[1], self.n_latent)
        self.fc3_logvar = nn.Linear(self.encoder_architecture[1], self.n_latent)

        # Decoder and output layers 
        if self.bayesian:
            self.bfc4 = BayesianLinear(self.n_latent, self.decoder_architecture[0], bias=False)
            self.bfc5 = BayesianLinear(self.decoder_architecture[0], self.decoder_architecture[1])
            self.bfc6 = BayesianLinear(self.decoder_architecture[1], self.seq_len * self.alphabet_size)

        else:
            self.bfc4 = nn.Linear(self.n_latent, self.decoder_architecture[0], bias=False)
            self.bfc5 = nn.Linear(self.decoder_architecture[0], self.decoder_architecture[1])
            self.bfc6 = nn.Linear(self.decoder_architecture[1], self.seq_len * self.alphabet_size)


    def _encode(self, x):
        h_enc_1 = F.relu(self.fc1(x))
        h_enc_2 = F.relu(self.fc2(h_enc_1))
        return self.fc3_mu(h_enc_2), self.fc3_logvar(h_enc_2)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def _decode_variational(self, z):
        bfc_4_out = F.relu(self.bfc4(z))
        bfc_5_out = F.relu(self.bfc5(bfc_4_out))

        # Last layer uses sigmoid
        bfc_6_out = torch.sigmoid(self.bfc6(bfc_5_out))

        # Reshape output back into original tensor
        #output = bfc_6_out.reshape(self.batch_size, self.seq_len, self.alphabet_size)
        return bfc_6_out


    def forward(self, x):

        # TODO: remove after debugging
        #x = torch.Tensor(x)

        # Encode to bottleneck and sample from bottleneck
        mu, logvar = self._encode(x.view(-1, self.seq_len * self.alphabet_size))
        z = self._reparameterize(mu, logvar)

        # Variational decoder
        reconstructed_x = self._decode_variational(z)

        return reconstructed_x, mu, logvar 
        



