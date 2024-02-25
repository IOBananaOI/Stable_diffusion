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
from utils import get_time_embedding

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


    def generate(
            self, 
            prompt : str,
            uncond_prompt : str = None,
            strength=0.8,
            do_cfg=True,
            cfg_scale=7.5,
            sampler_name='ddpm',
            n_inference_steps=50,
            seed=None
    ):
        with torch.no_grad():
            assert 0 <= strength <= 1, "Strength must me between 0 and 1"

            generator = torch.Generator(device=self.device)
            if seed is None:
                generator.seed()
            else:
                generator.manual_seed(seed)

            uncond_prompt = uncond_prompt or [""] * len(prompt)

            tokenizer = Tokenizer()

            if do_cfg:
                # Tokenize the prompt
                cond_tokens = tokenizer.encode_batch(prompt)
                cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=self.device)
                cond_context = self.clip(cond_tokens)

                # Handling uncond_prompt
                uncond_tokens = tokenizer.encode_batch(uncond_prompt)
                uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=self.device)
                uncond_context = self.clip(uncond_tokens)

                context = torch.cat([cond_context, uncond_context])

            else:
                tokens = tokenizer.encode_batch(prompt)
                tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)

                context = self.clip(tokens)

            del tokenizer

            sampler = DDPMSampler()

            noise_shape = (1, self.config.vae_latent_dim, self.config.latent_img_size)

            noise = torch.randn(noise_shape, generator=generator, device=self.device)
            noise *= sampler.initial_scale

            timesteps = sampler.timesteps
            for i, timestep in enumerate(timesteps):
                time_embedding = self.get_time_embedding(timestep)

                # (batch_size, vae_latent_dim, latent_img_size, latent_img_size)
                model_input = noise * sampler.get_input_scale()

                if do_cfg:
                    # (batch_size, vae_latent_dim, latent_img_size, latent_img_size) -> # (2 * batch_size, vae_latent_dim, latent_img_size, latent_img_size)
                    model_input = model_input.repeat(2, 1, 1, 1)

                model_output = self.unet(model_input, context, time_embedding)

                if do_cfg:
                    output_cond, output_uncond = model_input.chunk(2)

                    model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

                
        


    def forward(self, img : torch.Tensor, caption : torch.Tensor, time : torch.Tensor):
        img_latent_size = self.config.img_size // 2**(len(self.config.vae_features_dims))

        # VAE ENCODER 
        noise = torch.randn((self.config.batch_size, self.config.vae_latent_dim, img_latent_size, img_latent_size))
        z = self.vae_enc(img, noise)

        # Context generation with CLIP
        context = self.clip(caption)

        # UNET

        z = self.unet(z, time, context)

        print(z.shape)

        # VAE Decoder
        out = self.vae_dec(z)

        return z