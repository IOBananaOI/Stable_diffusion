from dataclasses import dataclass

import os

import numpy as np
import torch

import torch.nn.functional as F

from pathlib import Path

def get_alpha_bar(T, device, s = 0.08):
    f = lambda t: np.cos((t + s) / (1 + s) * np.pi/2)**2
    
    return torch.tensor([f((t+1) / T)/f(0) for t in range(T)], device=device, dtype=torch.float32)


def get_betas_from_alpha_bar(device, T, max_beta = 0.999, s = 0.08):
    f = lambda t: np.cos((t + s) / (1 + s) * np.pi/2)**2

    betas = []
    for t in range(T):
        t1 = t / T
        t2 = (t + 1) / T
        
        betas.append(min(1 - f(t2)/f(t1), max_beta))

    return torch.tensor(betas, device=device)


@dataclass(init=True)
class StableDiffusionConfig:
    # Training info
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = 4
    num_epochs = 1
    lr = 1e-3
    preload = True
    to_eval = True
    eval_caption = "A cat with a hat"

    # Data info
    img_channels = 3
    img_size = 256
    vocab_size = 49408
    img_path = Path('data/images/train2017/')
    captions_path = Path('data/captions/captions_train2017.json')

    # Tokenizer info
    tokenizer_vocab_path = Path('data/tokenizer_vocab.json')
    tokenizer_merges_path = Path('data/tokenizer_merges.txt')

    # VAE info
    vae_features_dims = [128, 256, 512]
    vae_latent_dim = 4
    vae_num_groups = 32
    vae_num_heads = 8
    vae_dropout = 0.2

    # Latent img_size
    img_latent_size = img_size // 2**len(vae_features_dims)

    # CLIP info
    clip_emb_dim = 768
    clip_seq_len = 30
    clip_emb_dim_scale_factor = 4
    clip_attn_num_heads = 12
    clip_num_layers = 6
    clip_dropout = 0.1


    # UNet info
    unet_time_emb_dim = 320
    unet_time_emb_dim_scale_factor = 2
    unet_features_dims = [320, 640, 1280]
    unet_attn_num_heads = 8
    unet_attn_dim = 40

    # Forward diffusion info
    T = 2000
    alpha_bar = get_alpha_bar(T, device)
    alpha_bar_prev = F.pad(alpha_bar[:-1], (1, 0), value=1.0)
    betas = get_betas_from_alpha_bar(device, T)
    alphas = 1. - betas
    posterior_variance = betas * (1. - alpha_bar_prev) / (1. - alpha_bar)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alpha_bar)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alpha_bar)    

    # Classifier free guidance
    do_cfg = False
    cfg_scale = 7.5

    # Neptune log tracking
    neptune_project_name = "bng215/Transformer-edu"
    neptune_project_api_token = os.environ.get('NEPTUNE_API_TOKEN')
    neptune_run_id = None

    # Model saving
    weights_folder = Path('src/model/weights/')
    model_name = "stable_diffusion_"
    weights_name : str = 'latest'
    saving_strategy = 'last'
