from typing import Literal
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(
        self,
        z_dim=2,
        criterion: Literal["MSE", "BCE"] = "MSE",
    ):
        super(Autoencoder, self).__init__()
        self.criterion = criterion
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1), # [1, 28, 28] -> [16, 14, 14]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # [16, 14, 14] -> [32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [32, 7, 7] -> [64, 7, 7]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=0), # [64, 7, 7] -> [128, 1, 1]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten(),
        )

        self.fc1 = nn.Linear(128*1*1, z_dim)                              # 128 -> 2
        self.fc2 = nn.Linear(z_dim, 128*1*1)                             # 2 -> 128

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (128, 1, 1)),
            nn.ConvTranspose2d(128, 64, kernel_size=7, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def loss(self, x, x_hat):
        if self.criterion == "BCE":
            loss = nn.functional.binary_cross_entropy(x_hat, x)
        elif self.criterion == "MSE":
            loss = nn.functional.mse_loss(x_hat, x)
        else:
            raise ValueError("Invalid criterion")

        return {
            "total_loss": loss,
            "components": {
                self.criterion: loss,
            }
        }

    def encode(self, x):
        h = self.encoder(x)
        z = self.fc1(h)
        return z
    
    def decode(self, z):
        z = self.fc2(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
