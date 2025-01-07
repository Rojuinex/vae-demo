from typing import Literal

def train(
    batch_size: int,
    max_epochs: int | None,
    max_steps: int,
    lr_scheduler: Literal["one_cycle", "reduce_on_plateau"],
    model_type: Literal["autoencoder", "vae", "beta-vae"],
    z_dim: int,
    beta: int | None,
    beta_schedule: Literal[
        "constant",
        "linear-increasing",
        "linear-decreasing",
        "cosine-increasing",
        "cosine-decreasing",
    ] = "constant",
):
    if max_epochs is not None and max_steps > -1:
        raise ValueError("Only one of `max_epochs` or `max_steps` can be set")

    if beta is not None and model_type != "beta-vae":
        raise ValueError("Beta can only be set for `beta-vae` models")

    # Putting imports here optimizes the startup time of the script
    # so that `--help` responds faster
    import os
    from torchinfo import summary
    import torch
    from  torch.utils.data import DataLoader
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    import lightning as L
    from lightning.pytorch.callbacks import LearningRateMonitor
    from sklearn.model_selection import train_test_split
    from lightning.pytorch.loggers import TensorBoardLogger
    import orjson

    
    from vae_demo import (
        Autoencoder,
        VAE,
        BetaVAE,
        LitAutoencoder
    )

    if model_type == "autoencoder":
        model = Autoencoder(z_dim=z_dim)
    elif model_type == "vae":
        model = VAE(z_dim=z_dim)
    elif model_type == "beta-vae":
        model = BetaVAE(z_dim=z_dim, beta=beta, beta_schedule=beta_schedule)
    else:
        raise ValueError("Invalid model type")
    
    print(model)

    summary(model, (batch_size, 1, 28, 28))

    autoencoder = LitAutoencoder(model, lr_scheduler=lr_scheduler)

    # setup data
    dataset = MNIST(os.getcwd(), download=True, transform=ToTensor())
    train_set, val_set = train_test_split(
        dataset,
        test_size=0.1,
        random_state=42,
        stratify=dataset.targets.numpy(),
    )
    test_set = MNIST(os.getcwd(), train=False, download=True, transform=ToTensor())
    del dataset

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, persistent_workers=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, num_workers=4, persistent_workers=True)


    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True,
        log_weight_decay=True,
    )

    log_name=f"{batch_size}/"
    if max_epochs is not None:
        log_name += f"epochs-{max_epochs}"
    else:
        log_name += f"steps-{max_steps}"
    log_name += f"--z_dim-{z_dim}"
    if beta is not None:
        log_name += f"--beta-{beta}"
    if beta_schedule != "constant":
        log_name += f"-{beta_schedule}"
    log_name += f"--lr-{lr_scheduler}"

    logger = TensorBoardLogger(
        f"runs/{model_type}",
        name=log_name,
    )

    trainer = L.Trainer(
        logger=logger,
        callbacks=[lr_monitor],
        max_epochs=max_epochs,
        max_steps=max_steps,
    )

    trainer.fit(
        model=autoencoder,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4, persistent_workers=True)
    eval_output = trainer.test(
        model=autoencoder,
        dataloaders=test_loader
    )

    print(eval_output)
    
    hparams = {
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "lr_scheduler": lr_scheduler,
        "model_type": model_type,
        "z_dim": z_dim,
        "beta": beta,
        "beta_schedule": beta_schedule,
    }

    logger.log_hyperparams(hparams, eval_output[0])

    model_dir = f"models/{model_type}/{log_name}/version_{logger.version}"
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), f"{model_dir}/model.pth")
    with open(f"{model_dir}/hyperparams.json", "wb") as f:
        f.write(orjson.dumps(hparams))

    with open(f"{model_dir}/eval_output.json", "wb") as f:
        f.write(orjson.dumps(eval_output))