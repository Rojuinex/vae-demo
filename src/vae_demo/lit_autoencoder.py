import torch
from torch import optim
import torchvision.utils
import lightning as L
import numpy as np

class LitAutoencoder(L.LightningModule):
    def __init__(self, model, lr_scheduler: str = "reduce_on_plateau"):
        super().__init__()
        self.model = model
        self._lr_scheduler = lr_scheduler

    def _loss(self, phase, *args):
        loss = self.model.loss(*args)
        # Logging to TensorBoard (if installed) by default
        self.log(f"loss/{phase}", loss['total_loss'])
        if "components" in loss:
            for name, l in loss['components'].items():
                self.log(f"loss/{name}/{phase}", l)
        return loss['total_loss']
    
    def _forward(self, x):
        args = self.model(x)
        if type(args) == tuple:
            return args
        else:
            return (args,)

    def _step(self, phase, batch, batch_idx, log_images = False):
        x, _ = batch
        x_hat, *args = self._forward(x)
        loss = self._loss(phase, x, x_hat, *args)

        if log_images:
            # interleave input and reconstruction
            images = torch.stack([x, x_hat], dim=1).view(-1, 1, 28, 28)
            self.logger.experiment.add_images(f"images/{phase}", images[:16], self.global_step)

        return loss

    def training_step(self, batch, batch_idx):
        log_images = batch_idx % (self.trainer.estimated_stepping_batches // 10) == 0
        return self._step("train", batch, batch_idx, log_images)
    
    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx, batch_idx == 0)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx, batch_idx == 0)

    def on_validation_epoch_end(self):
        # We can only visualize the latent space if the model has a 2D latent space
        if not hasattr(self.model, "z_dim") or self.model.z_dim != 2:
            return
        
        self.model.eval()

        latent_range = 10

        with torch.no_grad():
            imgs = []
            xv, yv = np.meshgrid(np.linspace(-latent_range, latent_range, 10), np.linspace(-latent_range, latent_range, 10))
            for i in range(len(xv)):
                for j in range(len(yv)):
                    x, y = xv[i][j], yv[i][j]
                    z = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0).to(self.device)
                    imgs.append(self.model.decode(z).squeeze(0))

            ig = torchvision.utils.make_grid(imgs, nrow=len(xv))
            self.logger.experiment.add_image(f"images/latent_space/x1", ig, self.global_step)

            imgs = []
            xv, yv = np.meshgrid(np.linspace(-latent_range, latent_range, 25), np.linspace(-latent_range, latent_range, 25))
            for i in range(len(xv)):
                for j in range(len(yv)):
                    x, y = xv[i][j], yv[i][j]
                    z = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0).to(self.device)
                    imgs.append(self.model.decode(z).squeeze(0))

            ig = torchvision.utils.make_grid(imgs, nrow=len(xv))
            self.logger.experiment.add_image(f"images/latent_space/x2", ig, self.global_step)

            imgs = []
            xv, yv = np.meshgrid(np.linspace(-latent_range, latent_range, 50), np.linspace(-latent_range, latent_range, 50))
            for i in range(len(xv)):
                for j in range(len(yv)):
                    x, y = xv[i][j], yv[i][j]
                    z = torch.tensor([x, y], dtype=torch.float32).unsqueeze(0).to(self.device)
                    imgs.append(self.model.decode(z).squeeze(0))

            ig = torchvision.utils.make_grid(imgs, nrow=len(xv))
            self.logger.experiment.add_image(f"images/latent_space/x3", ig, self.global_step)
        self.model.train()


    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=1e-3,
            weight_decay=1e-5,
        )
        if self._lr_scheduler == "one_cycle":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=1e-3,
                    total_steps=self.trainer.estimated_stepping_batches,
                ),
                "interval": "step"
            }
        elif self._lr_scheduler == "reduce_on_plateau":
            lr_scheduler = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=0.5,
                    patience=3,
                    verbose=True,
                ),
                "monitor": "loss/val",
                "interval": "epoch",
            }
        else:
            raise ValueError("Invalid lr_scheduler")
        
        return {
            "optimizer":  optimizer,
            "lr_scheduler": lr_scheduler
        }
