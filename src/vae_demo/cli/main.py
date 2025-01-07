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
def cli_train(
    batch_size: int,
    max_epochs: int | None,
    max_steps: int | None,
    lr_scheduler: str,
    model: str,
    z_dim: int,
    beta: int | None,
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
        lr_scheduler=lr_scheduler,
        model_type=model,
        z_dim=z_dim,
        beta=beta,
    )

@main.command(name="examples")
@click.option(
    "--batch-size",
    default=1024,
    help="Batch size for training. Default is 1024.",
)
@click.option(
    "--max-epochs",
    type=int,
    help="Maximum number of epochs to train. Default is 100.",
)
@click.option(
    "--max-steps",
    type=int,
    help="Maximum number of steps to train.",
)
@click.option(
    "--lr-scheduler",
    default="reduce_on_plateau",
    type=click.Choice(["one_cycle", "reduce_on_plateau"]),
    help="Learning rate scheduler to use. Default is `reduce_on_plateau`.",
)
@click.option(
    "--z-dim",
    default=2,
    help="Dimension of the latent space. Default is 2.",
)
@click.option(
    "--beta",
    default=[5],
    multiple=True,
    type=int,
    help="Beta parameter for beta-VAE models. Default is 5.",
)
def cli_examples(
    batch_size: int,
    max_epochs: int | None,
    max_steps: int | None,
    lr_scheduler: str,
    z_dim: int,
    beta: List[int] | None,
):
    """Trains multiple models for comparison."""
    from vae_demo.train import train

    if max_epochs is None and max_steps is None:
        max_epochs = 100

    if max_steps is None:
        max_steps = -1

    for b in beta:
        train(
            batch_size=batch_size,
            max_epochs=max_epochs,
            max_steps=max_steps,
            lr_scheduler=lr_scheduler,
            model_type="beta-vae",
            z_dim=z_dim,
            beta=b,
        )
    train(
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_steps=max_steps,
        lr_scheduler=lr_scheduler,
        model_type="autoencoder",
        z_dim=z_dim,
        beta=None,
    )
    train(
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_steps=max_steps,
        lr_scheduler=lr_scheduler,
        model_type="vae",
        z_dim=z_dim,
        beta=None,
    )
 

if __name__ == "__main__":
    main()