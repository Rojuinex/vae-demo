# VAE Demo

This repository demonstrates the differences in latent spaces between
a normal convolutional autoencoder and a variational autoencoder when trained
on MNIST.

# Prerequisites

1. [UV](https://docs.astral.sh/uv)

    * MacOS / Linux
        ```sh
        curl -LsSf https://astral.sh/uv/install.sh | sh
        ```
    
    * Windows
        ```powershell
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        ```

2. Python 3.12
    ```sh
    uv python install 3.12
    ```

## Setup

```sh
git clone git@github.com:Rojuinex/vae-demo.git
cd vae-demo
uv sync
```

## Train

### Demo experiments

To train the models using the configuration from the demo run
```sh
uv run vae-demo examples
```

### Monitor training

Launch tensorboard to monitor training, then navigate to http://localhost:6006

```sh
uv run tensorboard --logdir ./runs
```


### Custom configuration

Train a VAE model for the same number of steps as the two examples, but with
a smaller batch size.

```sh
uv run vae-demo train --batch-size=64 --max-steps=5299 --model=vae
```

Train a VAE model for the same number of epochs as the examples, but with
a smaller batch size.

```sh
uv run vae-demo train --batch-size=64 --max-epochs=100 --model=vae
```

See all of configuration options
```sh
$ uv run vae-demo train --help
Usage: vae-demo train [OPTIONS]

  Trains an autoencoder model.

Options:
  --batch-size INTEGER
  --z-dim INTEGER
  --max-epochs INTEGER
  --lr-scheduler [one_cycle|reduce_on_plateau]
  --model [autoencoder|vae]
  --help    
```