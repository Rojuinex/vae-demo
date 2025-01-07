import math
from typing import Literal
import torch
from vae_demo.models.vae import VAE


class BetaVAE(VAE):
    def __init__(
        self,
        z_dim=2,
        beta=5,
        beta_schedule: Literal[
            "constant",
            "linear-increasing",
            "linear-decreasing",
            "cosine-increasing",
            "cosine-decreasing",
        ] = "constant",
    ):
        super(BetaVAE, self).__init__(z_dim=z_dim)
        self.beta = beta
        self.beta_schedule = beta_schedule

    def _schedule_beta(self, current_step: int, max_steps: int):
        if self.beta_schedule == "constant":
            return
        
        if not hasattr(self, "_beta"):
            self._beta = self.beta

        if self.beta_schedule == "linear-increasing":
            self.beta = self._beta * torch.tensor(current_step / max_steps).float()
        elif self.beta_schedule == "linear-decreasing":
            self.beta = self._beta * torch.tensor(1 - current_step / max_steps).float()
        elif self.beta_schedule == "cosine-increasing":
            self.beta = self._beta * torch.tensor(
                0.5 * (1 + torch.cos(torch.tensor((1 - current_step / max_steps) * math.pi)))
            ).float()
        elif self.beta_schedule == "cosine-decreasing":
            self.beta = self._beta * torch.tensor(
                0.5 * (1 + torch.cos(torch.tensor(current_step / max_steps * math.pi)))
            ).float()
        else:
            raise ValueError("Invalid beta schedule")

    def loss(self, x, x_hat, mean, log_var):
        losses = super().loss(x, x_hat, mean, log_var)
        losses["components"]["KLD"] = self.beta * losses["components"]["KLD"]
        losses["total_loss"] = losses["components"]["MSE"] + losses["components"]["KLD"]
        return losses