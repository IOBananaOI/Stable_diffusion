import sys
sys.path.append('../')

import torch
from torch import nn

import torch.nn.functional as F

from utils import SelfAttention

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
    """
    VAE_Encoder/Decoder Block. Consists of 2 conv layers with prenorm and residual connection.
    If resize == 'upscale' -> doubles the img_size
    If resize == 'downscale' -> reduces the img_size by 2 times
    If resize == None -> just applys conv layers. 
    """
    def __init__(self, in_features : int, out_features : int, num_groups : int, dropout : int, resize : str = None) -> None:
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

        assert resize in ["upscale", "downscale", None], "resize parameter must be 'upscale', 'downscale' or None"

        if resize == "upscale":
            # (batch_size, vae_features_dims[i+1], img_size, img_size) -> (batch_size, vae_features_dims[i+1], img_size * 2, img_size * 2)
            self.resize_layer = nn.Upsample(scale_factor=2)
        
        elif resize == "downscale":
            # (batch_size, vae_features_dims[i+1], img_size, img_size) -> (batch_size, vae_features_dims[i+1], img_size // 2, img_size // 2)
            self.resize_layer = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=2, bias=False)


    def forward(self, x):
        # (batch_size, vae_features_dims[i], img_size, img_size) -> (batch_size, vae_features_dims[i+1], img_size, img_size) 
        x = self.conv_layer_1(x)

        # (batch_size, vae_features_dims[i+1], img_size, img_size) -> (batch_size, vae_features_dims[i+1], img_size, img_size) 
        x = self.conv_layer_2(x)

        if self.resize:
            if self.resize == 'downscale':
                x = F.pad(x, (0, 1, 0, 1))
            x = self.resize_layer(x)

        return x


class VAE_AttentionBlock(nn.Module):
    def __init__(self, attn_dim : int, num_groups : int, num_heads : int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, attn_dim)
        self.attn = SelfAttention(1, attn_dim)

    def forward(self, x: torch.Tensor):
        resid = x

        n, c, h, w = x.shape
        
        # (batch_size, attn_dim, latent_img_size, latent_img_size) -> (batch_size, attn_dim, latent_img_size**2) 
        x = x.view(n, c, h * w)

        # (batch_size, attn_dim, latent_img_size**2) -> (batch_size, latent_img_size**2, attn_dim) 
        x = x.transpose(-1, -2)

        # (batch_size, latent_img_size**2, attn_dim) -> (batch_size, latent_img_size**2, attn_dim)
        x = self.attn(x)

        # (batch_size, latent_img_size**2, attn_dim) -> (batch_size, attn_dim, latent_img_size**2)
        x = x.transpose(-1, -2)

        # (batch_size, attn_dim, latent_img_size**2) -> (batch_size, attn_dim, latent_img_size, latent_img_size)
        x = x.view(n, c, h, w)

        x += resid

        return x


class VAE_Encoder(nn.Module):
    def __init__(self, img_channels : int, vae_features_dims : list, num_groups : int, num_heads : int, dropout : int, vae_latent_dim : int) -> None:
        super().__init__()

        self.projection_layer = nn.Conv2d(in_channels=img_channels, out_channels=vae_features_dims[0], kernel_size=3, padding=1, bias=False)

        # Make first encoder_block with no channels changes
        first_enc_block = VAE_Block(vae_features_dims[0], vae_features_dims[0], num_groups, dropout, resize='downscale')
        
        # Build all encoder blocks
        self.enc_layers = nn.ModuleList(
            [first_enc_block] +
            [VAE_Block(vae_features_dims[i], vae_features_dims[i+1], num_groups, dropout, resize='downscale') for i in range(len(vae_features_dims)-1)]
        )

        attn_dim = vae_features_dims[-1]

        self.attn_layer = nn.Sequential(
            ### IF THE RESULTS AREN'T GOOD, TRY ADD MORE ENCODER BLOCKS
            VAE_Block(attn_dim, attn_dim, num_groups, dropout),
            VAE_Block(attn_dim, attn_dim, num_groups, dropout),
            VAE_AttentionBlock(attn_dim, num_groups, num_heads),
            VAE_Block(attn_dim, attn_dim, num_groups, dropout),
            nn.GroupNorm(num_groups, attn_dim),
            nn.SiLU()
        )

        self.latent_dim_proj_layer = nn.Sequential(
            nn.Conv2d(attn_dim, vae_latent_dim * 2, kernel_size=3, padding=1),
            nn.Conv2d(vae_latent_dim * 2, vae_latent_dim * 2, kernel_size=1)
        )

    def forward(self, x : torch.Tensor, noise : torch.Tensor):
        # (batch_size, img_channels, img_size, img_size) -> (batch_size, vae_features_dims[0], img_size, img_size)
        x = self.projection_layer(x)

        # (batch_size, vae_features_dims[0], img_size, img_size) -> (batch_size, attn_dim, latent_img_size, latent_img_size) 
        for layer in self.enc_layers:
            x = layer(x)

        # (batch_size, attn_dim, latent_img_size, latent_img_size) -> (batch_size, attn_dim, latent_img_size, latent_img_size)
        x = self.attn_layer(x)

        # (batch_size, attn_dim, latent_img_size, latent_img_size) -> (batch_size, vae_latent_dim * 2, latent_img_size, latent_img_size)
        x = self.latent_dim_proj_layer(x)

        # (batch_size, vae_latent_dim * 2, latent_img_size, latent_img_size) -> 2 tensors of shape (batch_size, vae_latent_dim, latent_img_size, latent_img_size)
        mean, log_var = torch.chunk(x, 2, dim=1)

        log_var = torch.clamp(log_var, -30, 20)

        variance = log_var.exp()

        std = variance.sqrt()

        mean = mean.to(noise.device)
        std = std.to(noise.device)

        # (batch_size, vae_latent_dim, latent_img_size, latent_img_size)
        x = mean + std * noise

        # Scale the output as in original repository
        x *= 0.18215

        return x
    

class VAE_Decoder(nn.Module):
    def __init__(self, vae_latent_dim : int, vae_features_dims : list, num_groups : int, dropout : int, num_heads : int, img_channels : int) -> None:
        super().__init__()

        attn_dim = vae_features_dims[-1]

        self.first_proj_layer = nn.Sequential(
            nn.Conv2d(vae_latent_dim, vae_latent_dim, kernel_size=1, bias=False),
            nn.Conv2d(vae_latent_dim, attn_dim, kernel_size=3, padding=1, bias=False)
        )

        self.attn_layer = nn.Sequential(
            VAE_Block(attn_dim, attn_dim, num_groups, dropout),
            VAE_AttentionBlock(attn_dim, num_groups, num_heads),
            VAE_Block(attn_dim, attn_dim, num_groups, dropout),
            VAE_Block(attn_dim, attn_dim, num_groups, dropout),
            VAE_Block(attn_dim, attn_dim, num_groups, dropout),            
        )

        # Make first decoder_block with no channels changes
        first_dec_block = nn.Sequential(
            VAE_Block(attn_dim, attn_dim, num_groups, dropout, resize='upscale'),
            nn.Conv2d(attn_dim, attn_dim, kernel_size=3, padding=1)
        )

        # Make for decoder reversed features dims list
        vae_dec_features_dims = vae_features_dims[::-1]

        self.dec_layers = nn.ModuleList(
            [first_dec_block] + 
            [
                nn.Sequential
                (
                    VAE_Block(vae_dec_features_dims[i], vae_dec_features_dims[i+1], num_groups, dropout),
                    VAE_Block(vae_dec_features_dims[i+1], vae_dec_features_dims[i+1], num_groups, dropout),
                    VAE_Block(vae_dec_features_dims[i+1], vae_dec_features_dims[i+1], num_groups, dropout, resize='upscale'),
                    nn.Conv2d(vae_dec_features_dims[i+1], vae_dec_features_dims[i+1], kernel_size=3, padding=1)
                ) 
            for i in range(len(vae_dec_features_dims)-1)]
        )

        self.second_proj_layer = nn.Sequential(
            nn.GroupNorm(num_groups, vae_dec_features_dims[-1]),
            nn.SiLU(),
            nn.Conv2d(vae_dec_features_dims[-1], img_channels, kernel_size=3, padding=1)
        )


    def forward(self, x):
        x *= 0.18215
        
        # (batch_size, vae_latent_dim, latent_img_size, latent_img_size) -> (batch_size, attn_dim, latent_img_size, latent_img_size)
        x = self.first_proj_layer(x)

        # (batch_size, attn_dim, latent_img_size, latent_img_size) -> (batch_size, attn_dim, latent_img_size, latent_img_size)
        x = self.attn_layer(x)

        # (batch_size, attn_dim, latent_img_size, latent_img_size) -> (batch_size, vae_features_dims[0], img_size, img_size)
        for layer in self.dec_layers:
            x = layer(x)
        
        # (batch_size, vae_features_dims[0], img_size, img_size) -> (batch_size, img_channels, img_size, img_size)
        x = self.second_proj_layer(x)

        return x
    


