from dataclasses import dataclass

import os

import torch


@dataclass
class StableDiffusionConfig:
    # Training info
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 32
    num_epochs = 20

    # Data info
    img_channels = 3
    img_size = 256

    # VAE info
    vae_features_dims = [128, 256, 512]
    vae_latent_dim = 8
    vae_num_groups = 32
    vae_num_heads = 8
    vae_dropout = 0.2

    # UNet info
    T = 2000
    d_model = 1024

    # Neptune log tracking
    neptune_project_name = "bng215/Transformer-edu",
    neptune_project_api_token = os.environ.get('NEPTUNE_API_TOKEN'),
    neptune_run_id = None

    # Model saving
    weights_folder = "weights/"
    model_name = "stable_diffusion_"