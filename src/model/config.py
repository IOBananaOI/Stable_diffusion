from dataclasses import dataclass

import os

import torch


@dataclass(init=True)
class StableDiffusionConfig:
    # Training info
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size : int = 1
    num_epochs = 20

    # Data info
    img_channels = 3
    img_size = 256
    vocab_size = 49408

    # Tokenizer info
    tokenizer_vocab_path = '../../data/vocab.json'
    tokenizer_merges_path = '../../data/merges.txt'

    # VAE info
    vae_features_dims = [128, 256, 512]
    vae_latent_dim = 4
    vae_num_groups = 32
    vae_num_heads = 8
    vae_dropout = 0.2

    # Latent img_size
    latent_img_size = img_size // len(vae_features_dims)

    # CLIP info
    clip_emb_dim = 768
    clip_seq_len = 77
    clip_emb_dim_scale_factor = 4
    clip_attn_num_heads = 12
    clip_num_layers = 12
    clip_dropout = 0.1

    # UNet info
    unet_time_emb_dim = 320
    unet_time_emb_dim_scale_factor = 4
    unet_features_dims = [320, 640, 1280]
    unet_attn_num_heads = 8
    unet_attn_dim = 40


    T = 2000
    d_model = 1024

    # Neptune log tracking
    neptune_project_name = "bng215/Transformer-edu",
    neptune_project_api_token = os.environ.get('NEPTUNE_API_TOKEN'),
    neptune_run_id = None

    # Model saving
    weights_folder = "weights/"
    model_name = "stable_diffusion_"
    weights_name : str = None
