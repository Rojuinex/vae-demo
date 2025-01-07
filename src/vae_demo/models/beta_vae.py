import torch
import torch.nn as nn
from vae_demo.models.autoencoder import Autoencoder


class BetaVAE(Autoencoder):
    def __init__(self, z_dim=2, beta=5):
        super(BetaVAE, self).__init__(z_dim=z_dim)
        self.beta = beta
        self.fc1 = nn.Linear(128, z_dim*2)

    def loss(self, x, x_hat, mean, log_var):
        MSE = nn.functional.mse_loss(x_hat, x, reduction="sum")
        KLD = self.beta * (-0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp()))
        loss = MSE+KLD
        return {
            "total_loss": loss,
            "components": {
                "MSE": MSE,
                "KLD": KLD
            }
        }

    def reparameterize(self, mu, log_var):
        std = log_var.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(mu.device)
        z = mu + std * esp
        return z
    
    def bottleneck(self, h):
        x = self.fc1(h)
        mu, log_var = x[:, :self.z_dim], x[:, self.z_dim:]
        z = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def encode(self, x):
        h = self.encoder(x)
        z, mu, log_var = self.bottleneck(h)
        return z, mu, log_var

    def forward(self, x):
        z, mu, log_var = self.encode(x)
        x_hat = self.decode(z)

        return x_hat, mu, log_var
