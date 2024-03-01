import torch
from torch import nn

import numpy as np

import sys 
sys.path.append('../')

from .config import StableDiffusionConfig
from .vae import VAE_Encoder, VAE_Decoder
from .unet import UNet
from .clip import CLIPEncoder
from .tokenizer import Tokenizer

from tqdm import tqdm

class StableDiffusion(nn.Module):
    def __init__(self, config : StableDiffusionConfig) -> None:
        super().__init__()
        self.config = config
        self.device = config.device

        self.vae_enc = VAE_Encoder(
            config.img_channels,
            config.vae_features_dims,
            config.vae_num_groups,
            config.vae_num_heads,
            config.vae_dropout,
            config.vae_latent_dim
        ).to(self.device)

        self.vae_dec = VAE_Decoder(
            config.vae_latent_dim,
            config.vae_features_dims,
            config.vae_num_groups,
            config.vae_dropout,
            config.vae_num_heads,
            config.img_channels,
        ).to(self.device)

        self.clip = CLIPEncoder(
            config.vocab_size, 
            config.clip_emb_dim, 
            config.clip_seq_len,
            config.clip_attn_num_heads, 
            config.clip_emb_dim_scale_factor, 
            config.clip_num_layers, 
            config.clip_dropout
        ).to(self.device)

        self.unet = UNet(
            config.vae_latent_dim,
            config.unet_features_dims,
            config.unet_attn_num_heads,
            config.unet_attn_dim,
            config.unet_time_emb_dim,
            config.unet_time_emb_dim_scale_factor
        ).to(self.device)


    def get_time_embedding(self, timestep : int):
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
        x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1).to(self.device)
        return x   


    def forward(
            self, 
            img : torch.Tensor, 
            caption : torch.Tensor, 
            time : torch.Tensor,
            tokenizer : Tokenizer,
            do_cfg=True
            ):

        # VAE ENCODER 
        noise = torch.randn((self.config.batch_size, self.config.vae_latent_dim, self.config.img_latent_size, self.config.img_latent_size))
        
        # Get reduced img
        img = self.vae_enc(img, noise)

        # Context generation with CLIP

        if do_cfg:
            # Tokenize the prompt
            cond_tokens = tokenizer.encode_batch(caption)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=self.device)
            cond_context = self.clip(cond_tokens)

            # Handling uncond_prompt
            uncond_prompt = uncond_prompt or [""] * len(caption)
            uncond_tokens = tokenizer.encode_batch(uncond_prompt)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=self.device)
            uncond_context = self.clip(uncond_tokens)

            context = torch.cat([cond_context, uncond_context])

        else:
            tokens = tokenizer.encode_batch(caption)
            tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)

        context = self.clip(tokens)

        # UNET
        time = self.get_time_embedding(time)

        if do_cfg:
            # (batch_size, vae_latent_dim, latent_img_size, latent_img_size) -> # (2 * batch_size, vae_latent_dim, latent_img_size, latent_img_size)
            img = img.repeat(2, 1, 1, 1)

        unet_output = self.unet(img, time, context)

        if do_cfg:
            output_cond, output_uncond = unet_output.chunk(2)

            unet_output = self.config.cfg_scale * (output_cond - output_uncond) + output_uncond

        # VAE Decoder
        out = self.vae_dec(unet_output)

        return out