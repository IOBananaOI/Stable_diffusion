import torch
from torch import nn

import torch.nn.functional as F

from .config import StableDiffusionConfig

class PrenormResidualConnection(nn.Module):
    def __init__(self, sublayer: nn.Module, in_features : int, out_features : int, num_groups : int, dropout : int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_features)
        self.sublayer = sublayer
        self.dropout = nn.Dropout(dropout)

        if in_features == out_features:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1)

    def forward(self, x):
        return self.residual_layer(x) + self.dropout(self.sublayer(F.silu(self.norm(x))))


# VAE Block
class VAE_Block(nn.Module):
    def __init__(self, in_features : int, out_features : int, num_groups : int, dropout : int, resize=True) -> None:
        super().__init__()
        
        self.conv_layer_1 = PrenormResidualConnection(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, padding=1, bias=False),
            in_features,
            out_features,
            num_groups,
            dropout
        )

        self.conv_layer_2 = PrenormResidualConnection(
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, padding=1, bias=False),
            out_features,
            out_features,
            num_groups,
            dropout
        )

        self.resize = resize

        if self.resize:

            # Check if the block is in encoder or decoder
            if in_features < out_features:
                # (batch_size, vae_features_dims[i+1], img_size, img_size) -> (batch_size, vae_features_dims[i+1], img_size // 2, img_size // 2)
                self.resize_layer = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=2, bias=False)
            else:
                # (batch_size, vae_features_dims[i+1], img_size, img_size) -> (batch_size, vae_features_dims[i+1], img_size * 2, img_size * 2)
                self.resize_layer = nn.Upsample(scale_factor=2)

    def forward(self, x):
        # (batch_size, vae_features_dims[i], img_size, img_size) -> (batch_size, vae_features_dims[i+1], img_size, img_size) 
        x = self.conv_layer_1(x)

        # (batch_size, vae_features_dims[i+1], img_size, img_size) -> (batch_size, vae_features_dims[i+1], img_size, img_size) 
        x = self.conv_layer_2(x)

        if self.resize:
            x = self.resize_layer(F.pad(x, (0, 1, 0, 1)))

        return x


class VAE_AttentionBlock(nn.Module):
    def __init__(self, attn_dim : int, num_groups : int, num_heads : int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, attn_dim)
        self.attn = nn.MultiheadAttention(attn_dim, num_heads=num_heads)

    def forward(self, x: torch.Tensor):
        resid = x

        n, c, h, w = x.shape
        
        # (batch_size, attn_dim, img_latent_size, img_latent_size) -> (batch_size, attn_dim, img_latent_size**2) 
        x = x.view(n, c, h * w)

        # (batch_size, attn_dim, img_latent_size**2) -> (batch_size, img_latent_size**2, attn_dim) 
        x = x.transpose(-1, -2)

        # (batch_size, img_latent_size**2, attn_dim) -> (batch_size, img_latent_size**2, attn_dim)
        x, _ = self.attn(x, x, x, need_weights=False)

        # (batch_size, img_latent_size**2, attn_dim) -> (batch_size, attn_dim, img_latent_size**2)
        x = x.transpose(-1, -2)

        # (batch_size, attn_dim, img_latent_size**2) -> (batch_size, attn_dim, img_latent_size, img_latent_size)
        x = x.view(n, c, h, w)

        x += resid

        return x


class VAE_Encoder(nn.Module):
    def __init__(self, img_channels : int, vae_features_dims : list, num_groups : int, num_heads : int, dropout : int, vae_latent_dim : int) -> None:
        super().__init__()

        self.projection_layer = nn.Conv2d(in_channels=img_channels, out_channels=vae_features_dims[0], kernel_size=3, padding=1, bias=False)

        # Make first encoder_block with no channels changes
        first_enc_block = VAE_Block(vae_features_dims[0], vae_features_dims[0], num_groups, dropout, resize=False)
        
        # Build all encoder blocks
        self.enc_layers = nn.ModuleList(
            [first_enc_block] +
            [VAE_Block(vae_features_dims[i], vae_features_dims[i+1], num_groups, dropout) for i in range(len(vae_features_dims)-1)]
        )

        attn_dim = vae_features_dims[-1]

        self.attn_layer = nn.Sequential(
            ### IF THE RESULTS AREN'T GOOD, TRY ADD MORE ENCODER BLOCKS
            VAE_Block(attn_dim, attn_dim, num_groups, dropout, False),
            VAE_Block(attn_dim, attn_dim, num_groups, dropout, False),
            VAE_AttentionBlock(attn_dim, num_groups, num_heads),
            VAE_Block(attn_dim, attn_dim, num_groups, dropout, False),
            nn.GroupNorm(num_groups, attn_dim),
            nn.SiLU()
        )

        self.latent_dim_proj_layer = nn.Sequential(
            nn.Conv2d(attn_dim, vae_latent_dim, kernel_size=3, padding=1),
            nn.Conv2d(vae_latent_dim, vae_latent_dim, kernel_size=1)
        )

    def forward(self, x, noise : torch.Tensor):
        # (batch_size, img_channels, img_size, img_size) -> (batch_size, vae_features_dims[0], img_size, img_size)
        x = self.projection_layer(x)
        print(x.shape)

        # (batch_size, vae_features_dims[0], img_size, img_size) -> (batch_size, attn_dim, img_latent_size, img_latent_size) 
        # where img_latent_size = img_size // 2^(len(vae_features_dims)-1)
        for layer in self.enc_layers:
            x = layer(x)

        # (batch_size, attn_dim, img_latent_size, img_latent_size) -> (batch_size, attn_dim, img_latent_size, img_latent_size)
        x = self.attn_layer(x)

        # (batch_size, attn_dim, img_latent_size, img_latent_size) -> (batch_size, vae_latent_dim, img_latent_size, img_latent_size)
        x = self.latent_dim_proj_layer(x)
        
        mean, log_var = torch.chunk(x, 2, dim=1)

        log_var = torch.clamp(log_var, -30, 20)

        variance = log_var.exp()

        std = variance.sqrt()
        print(std.shape, mean.shape)
        x = mean + std * noise

        # Scale the output as in original repository
        x *= 0.18215

        return x
    

class VAE_Decoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return x
    

class VAE(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        pass

