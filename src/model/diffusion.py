import torch
from torch import nn

from .config import StableDiffusionConfig
from .vae import VAE_Encoder, VAE_Decoder
from .unet import UNet
from .clip import CLIPEncoder

class StableDiffusion(nn.Module):
    def __init__(self, config : StableDiffusionConfig) -> None:
        super().__init__()
        self.config = config

        self.vae_enc = VAE_Encoder(
            config.img_channels,
            config.vae_features_dims,
            config.vae_num_groups,
            config.vae_num_heads,
            config.vae_dropout,
            config.vae_latent_dim
        )

        self.vae_dec = VAE_Decoder(
            config.vae_latent_dim,
            config.vae_features_dims,
            config.vae_num_groups,
            config.vae_dropout,
            config.vae_num_heads,
            config.img_channels,
        )

        self.clip = CLIPEncoder(
            config.vocab_size, 
            config.clip_emb_dim, 
            config.clip_seq_len,
            config.clip_attn_num_heads, 
            config.clip_emb_dim_scale_factor, 
            config.clip_num_layers, 
            config.clip_dropout
        )

        self.unet = UNet(
            config.vae_latent_dim,
            config.unet_features_dims,
            config.unet_attn_num_heads,
            config.unet_attn_dim,
            config.unet_time_emb_dim,
            config.unet_time_emb_dim_scale_factor
        )

    def forward(self, img : torch.Tensor, caption : torch.Tensor):
        img_latent_size = self.config.img_size // 2**(len(self.config.vae_features_dims))

        # VAE ENCODER 
        noise = torch.randn((self.config.batch_size, self.config.vae_latent_dim, img_latent_size, img_latent_size))
        z = self.vae_enc(img, noise).to(self.config.device)

        # # Context generation with CLIP
        context = self.clip(caption)


        # # VAE Decoder
        # out = self.vae_dec(z)

        return z