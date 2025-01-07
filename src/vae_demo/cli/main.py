import click
from typing import List

@click.group()
def main():
    pass

@main.command(name="train")
@click.option(
    "--batch-size",
    default=1024,
    help="Batch size for training. Default is 1024.",
)
@click.option(
    "--max-epochs",
    type=int,
    help="Maximum number of epochs to train. Default is 100 if both `max_epochs` and `max_steps` are not set.",
)
@click.option(
    "--max-steps",
    type=int,
    help="Maximum number of steps to train.",
)
@click.option(
    "--criterion",
    default="MSE",
    type=click.Choice(["MSE", "BCE"]),
    help="Loss criterion to use. Default is `MSE`.",
)
@click.option(
    "--lr-scheduler",
    default="reduce_on_plateau",
    type=click.Choice(["one_cycle", "reduce_on_plateau"]),
    help="Learning rate scheduler to use. Default is `reduce_on_plateau`.",
)
@click.option(
    "--model",
    default="autoencoder",
    type=click.Choice(["autoencoder", "vae", "beta-vae"]),
    help="Type of model to train. Default is `autoencoder`.",
)
@click.option(
    "--z-dim",
    default=2,
    help="Dimension of the latent space. Default is 2.",
)
@click.option(
    "--beta",
    type=int,
    help="Beta parameter for beta-VAE models. Default is 5.",
)
@click.option(
    "--beta-schedule",
    default="constant",
    type=click.Choice([
        "constant",
        "linear-increasing",
        "cosine-increasing",
    ]),
    help="Schedule for beta parameter. Default is `constant`.",
)
@click.option(
    '--latent-space-magnitude',
    default=10,
    help='Magnitude of the latent space visualization. Default is 10.',
)
@click.option(
    '--latent-space-resolution',
    'latent_space_resolutions',
    default=[10, 25, 50],
    multiple=True,
    type=int,
    help='Resolutions of the latent space visualization. Default is 10, 25, 50.',
)
def cli_train(
    batch_size: int,
    max_epochs: int | None,
    max_steps: int | None,
    criterion: str,
    lr_scheduler: str,
    model: str,
    z_dim: int,
    beta: int | None,
    beta_schedule: str,
    latent_space_magnitude: int,
    latent_space_resolutions: List[int],
):
    """Trains an autoencoder model.
    """
    from vae_demo.train import train

    if max_epochs is None and max_steps is None:
        max_epochs = 100

    if model == "beta-vae" and beta is None:
        beta = 5

    if max_steps is None:
        max_steps = -1

    train(
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_steps=max_steps,
        criterion=criterion,
        lr_scheduler=lr_scheduler,
        model_type=model,
        z_dim=z_dim,
        beta=beta,
        beta_scheduler=beta_schedule,
        latent_space_magnitude=latent_space_magnitude,
        latent_space_resolutions=latent_space_resolutions,
    )

@main.command(name="experiment")
@click.option(
    '--model',
    'models',
    default=["autoencoder", "vae", "beta-vae"],
    multiple=True,
    type=click.Choice(["autoencoder", "vae", "beta-vae"]),
    help='Model to train. Default is all models.',
)
@click.option(
    "--batch-size",
    "batch_sizes",
    default=[1024],
    multiple=True,
    type=int,
    help="Batch size for training. Default is 1024.",
)
@click.option(
    "--max-epochs",
    "max_epoch_list",
    type=int,
    multiple=True,
    help="Maximum number of epochs to train. Default is 100.",
)
@click.option(
    "--max-steps",
    "max_step_list",
    type=int,
    multiple=True,
    help="Maximum number of steps to train.",
)
@click.option(
    "--criterion",
    "criteria",
    default=["MSE"],
    multiple=True,
    type=click.Choice(["MSE", "BCE"]),
    help="Loss criterion to use. Default is `MSE`.",
)
@click.option(
    "--lr-scheduler",
    "lr_schedulers",
    default=["reduce_on_plateau"],
    multiple=True,
    type=click.Choice(["one_cycle", "reduce_on_plateau"]),
    help="Learning rate scheduler to use. Default is `reduce_on_plateau`.",
)
@click.option(
    "--z-dim",
    "z_dims",
    default=[2],
    multiple=True,
    help="Dimension of the latent space. Default is 2.",
)
@click.option(
    "--beta",
    "betas",
    default=[5],
    multiple=True,
    type=int,
    help="Beta parameter for beta-VAE models. Default is 5.",
)
@click.option(
    "--beta-schedule",
    "beta_schedulers",
    default=["constant"],
    type=click.Choice([
        "constant",
        "linear-increasing",
        "cosine-increasing",
    ]),
    multiple=True,
    help="Schedule for beta parameter. Default is `constant`.",
)
@click.option(
    '--latent-space-magnitude',
    default=10,
    help='Magnitude of the latent space visualization. Default is 10.',
)
@click.option(
    '--latent-space-resolution',
    'latent_space_resolutions',
    default=[10, 25, 50],
    multiple=True,
    type=int,
    help='Resolutions of the latent space visualization. Default is 10, 25, 50.',
)
@click.option(
    '--dry-run',
    is_flag=True,
    help='Print the combinations of hyperparameters without training.',
)
def cli_experiment(
    models: List[str],
    batch_sizes: List[int],
    max_epoch_list: List[int],
    max_step_list: List[int],
    criteria: List[str],
    lr_schedulers: List[str],
    z_dims: List[int],
    betas: List[int],
    beta_schedulers: List[str],
    latent_space_magnitude: int,
    latent_space_resolutions: List[int],
    dry_run: bool,
):
    """Trains multiple models for comparison."""
    from vae_demo.train import train
    import itertools
    import orjson

    if len(max_epoch_list) > 0 and len(max_step_list) > 0:
        raise ValueError("Cannot mix `max_epochs` and `max_steps`")

    if len(max_epoch_list) == 0 and len(max_step_list) == 0:
        max_epoch_list = [100]

    if len(max_epoch_list) == 0:
        max_epoch_list = [None]

    if len(max_step_list) == 0:
        max_step_list = [-1]

    for (
        max_steps,
        max_epochs,
        batch_size,
        lr_scheduler,
        z_dim,
        criterion,
        model
    ) in itertools.product(
        max_step_list,
        max_epoch_list,
        batch_sizes,
        lr_schedulers,
        z_dims,
        criteria,
        models,
    ):
        _betas = [None]
        _beta_schedulers = [None]
        if model == "beta-vae":
            _betas = betas
            _beta_schedulers = beta_schedulers

        for beta, beta_scheduler in itertools.product(_betas, _beta_schedulers):
            if dry_run:
                print(orjson.dumps({
                    "model": model,
                    "criterion": criterion,
                    "z_dim": z_dim,
                    "lr_scheduler": lr_scheduler,
                    "batch_size": batch_size,
                    "max_epochs": max_epochs,
                    "max_steps": max_steps,
                    "beta": beta,
                    "beta_scheduler": beta_scheduler,
                    "latent_space_magnitude": latent_space_magnitude,
                    "latent_space_resolutions": latent_space_resolutions
                }).decode())
            else:
                train(
                    batch_size=batch_size,
                    max_epochs=max_epochs,
                    max_steps=max_steps,
                    criterion=criterion,
                    lr_scheduler=lr_scheduler,
                    model_type=model,
                    z_dim=z_dim,
                    beta=beta,
                    beta_scheduler=beta_scheduler,
                    latent_space_magnitude=latent_space_magnitude,
                    latent_space_resolutions=latent_space_resolutions
                )

if __name__ == "__main__":
    main()