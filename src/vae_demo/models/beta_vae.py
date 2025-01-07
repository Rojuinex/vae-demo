import math
from typing import Literal
import torch
from vae_demo.models.vae import VAE


class BetaVAE(VAE):
    def __init__(
        self,
        z_dim=2,
        criterion: Literal["MSE", "BCE"] = "MSE",
        beta=5,
        beta_scheduler: Literal[
            "constant",
            "linear-increasing",
            "cosine-increasing",
        ] = "constant",
    ):
        super(BetaVAE, self).__init__(z_dim=z_dim, criterion=criterion)
        self.beta = beta
        self.beta_scheduler = beta_scheduler

    def _schedule_beta(self, current_step: int, max_steps: int):
        if self.beta_scheduler == "constant":
            return 1.0
        
        if not hasattr(self, "_beta"):
            self._beta = self.beta

        if self.beta_scheduler == "linear-increasing":
            factor = torch.tensor(current_step / max_steps).float()
        elif self.beta_scheduler == "cosine-increasing":
            factor = torch.tensor(
                0.5 * (1 + torch.cos(torch.tensor((1 - current_step / max_steps) * math.pi)))
            ).float()
        else:
            raise ValueError("Invalid beta schedule")
        
        self.beta = self._beta * factor
        return factor

    def loss(self, x, x_hat, mean, log_var):
        losses = super().loss(x, x_hat, mean, log_var)
        losses["components"]["KLD/Unscaled"] = losses["components"]["KLD"]
        losses["components"]["KLD"] = self.beta * losses["components"]["KLD"]
        losses["total_loss"] = losses["components"][self.criterion] + losses["components"]["KLD"]
        return losses