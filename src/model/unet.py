import sys
sys.path.append('../')

import torch

import torch.nn.functional as F
from torch import nn

from utils import SelfAttention, CrossAttention

class UNet_TimeEmbedding(nn.Module):
    def __init__(self, emb_dim : int, scale_factor : int) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(emb_dim, scale_factor * emb_dim)
        self.linear_2 = nn.Linear(scale_factor * emb_dim, scale_factor * emb_dim)

    def forward(self, x):
        # (1, emb_dim) -> (1, scale_factor * emb_dim)
        x = self.linear_1(x)

        x = F.silu(x)

        x = self.linear_2(x)

        return x


class UNet_ResidualBlock(nn.Module):
    def __init__(self, in_features : int, out_features : int, time_dim : int) -> None:
        super().__init__()

        self.norm_1 = nn.GroupNorm(32, in_features)
        self.conv_features = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.time_linear_layer = nn.Linear(time_dim, out_features)

        self.norm_2 = nn.GroupNorm(32, out_features)
        self.conv_merged = nn.Conv2d(out_features, out_features, kernel_size=3, padding=1)


        if in_features == out_features:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_features, out_features, kernel_size=1)

    def forward(self, x : torch.Tensor, time : torch.Tensor):
        # x: (batch_size, in_features, img_size, img_size)
        # time: (1, unet_features_dims[-1])

        resid = x

        x = self.norm_1(x)

        x = F.silu(x)

        x = self.conv_features(x)

        time = F.silu(time)

        time = self.time_linear_layer(time)

        merged = x + time.unsqueeze(-1).unsqueeze(-1)

        merged = self.norm_2(merged)

        merged = F.silu(merged)

        merged = self.conv_merged(merged)

        return merged + self.residual_layer(resid)
    

class UNet_AttentionBlock(nn.Module):
    def __init__(self, num_heads : int, emb_dim : int, context_dim : int = 768) -> None:
        super().__init__()
        num_channels = num_heads * emb_dim

        self.norm = nn.GroupNorm(32, num_channels)
        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=1)

        self.ln_1 = nn.LayerNorm(num_channels)
        self.ln_2 = nn.LayerNorm(num_channels)
        self.ln_3 = nn.LayerNorm(num_channels)

        self.self_attn = SelfAttention(num_heads, num_channels)
        self.cross_attn = CrossAttention(num_heads, num_channels, context_dim)

        self.linear_1 = nn.Linear(num_channels, num_channels * 8)
        self.linear_2 = nn.Linear(4 * num_channels, num_channels)

        self.conv_output = nn.Conv2d(num_channels, num_channels, kernel_size=1)


    def forward(self, x, context):
        resid_1 = x

        x = self.norm(x)

        x = self.conv(x)

        n, c, h, w = x.shape

        x = x.view(n, c, h * w)

        x = x.transpose(-1, -2)

        resid_2 = x

        # Apply Self Attention
        x = self.ln_1(x)
        x = self.self_attn(x)
        x += resid_2

        # Apply Cross Attention
        resid_2 = x
        x = self.ln_2(x)
        x = self.cross_attn(x, context)

        x += resid_2

        resid_2 = x

        x = self.ln_3(x)

        x, gate = self.linear_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)

        x = self.linear_2(x)

        x += resid_2

        x = x.transpose(-1, -2)

        x = x.view((n, c, h, w))

        x = self.conv_output(x) + resid_1

        return x


class SwitchSequential(nn.Sequential):

    def forward(self, x : torch.Tensor, context : torch.Tensor, time : torch.Tensor):
        for layer in self:
            if isinstance(layer, UNet_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNet_ResidualBlock):
                x = layer(x, time)
            else: 
                x = layer(x)
        
        return x


class Upsample(nn.Module):
    def __init__(self, num_features : int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # (batch_size, num_features, img_size, img_size) -> (batch_size, num_features, img_size * 2, img_size * 2)
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, vae_latent_dim : int, unet_features_dims : list, 
                 num_heads : int, attn_dim : int, time_emb_dim : int, 
                 time_emb_dim_scale_factor : int) -> None:
        super().__init__()
        
        self.time_embedding = UNet_TimeEmbedding(time_emb_dim, time_emb_dim_scale_factor)

        time_emb_dim *= time_emb_dim_scale_factor

        self.encoder = nn.ModuleList([
            ### BLOCK 1
            # (batch_size, vae_latent_dim, img_size // 8, img_size // 8) -> (batch_size, unet_features_dims[0], img_size // 8, img_size // 8)

            SwitchSequential(nn.Conv2d(vae_latent_dim, unet_features_dims[0], kernel_size=3, padding=1)),
            
            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[0], unet_features_dims[0], time_emb_dim), 
                UNet_AttentionBlock(num_heads, attn_dim)
            ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[0], unet_features_dims[0], time_emb_dim), 
                UNet_AttentionBlock(num_heads, attn_dim)
            ),

            ### BLOCK 2
            # (batch_size, unet_features_dims[0], img_size // 8, img_size // 8) -> (batch_size, unet_features_dims[1], img_size // 16, img_size // 16)

            SwitchSequential(nn.Conv2d(unet_features_dims[0], unet_features_dims[0], kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[0], unet_features_dims[1], time_emb_dim), 
                UNet_AttentionBlock(num_heads, attn_dim * 2)
            ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[1], unet_features_dims[1], time_emb_dim), 
                UNet_AttentionBlock(num_heads, attn_dim * 2)
            ),

            ### BLOCK 3
            # (batch_size, unet_features_dims[1], img_size // 16, img_size // 16) -> (batch_size, unet_features_dims[2], img_size // 32, img_size // 32)

            SwitchSequential(nn.Conv2d(unet_features_dims[1], unet_features_dims[1], kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[1], unet_features_dims[2], time_emb_dim), 
                UNet_AttentionBlock(num_heads, attn_dim * 4)
            ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[2], unet_features_dims[2], time_emb_dim), 
                UNet_AttentionBlock(num_heads, attn_dim * 4)
            ),

            ### BLOCK 4
            # (batch_size, unet_features_dims[2], img_size // 32, img_size // 32) -> (batch_size, unet_features_dims[2], img_size // 64, img_size // 64)

            SwitchSequential(nn.Conv2d(unet_features_dims[2], unet_features_dims[2], kernel_size=3, stride=2, padding=1)),
            
            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[2], unet_features_dims[2], time_emb_dim)
            ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[2], unet_features_dims[2], time_emb_dim)
            )
            
        ])

        self.bottleneck = SwitchSequential(
            UNet_ResidualBlock(unet_features_dims[2], unet_features_dims[2], time_emb_dim),
            UNet_AttentionBlock(num_heads, attn_dim * 4),
            UNet_ResidualBlock(unet_features_dims[2], unet_features_dims[2], time_emb_dim)
        )

        self.decoder = nn.ModuleList([
            ### BLOCK 1

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[2] * 2, unet_features_dims[2], time_emb_dim)
                ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[2] * 2, unet_features_dims[2], time_emb_dim)
                ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[2] * 2, unet_features_dims[2], time_emb_dim), 
                Upsample(unet_features_dims[2])
                ),

            ### BLOCK 2

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[2] * 2, unet_features_dims[2], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim * 4)
                ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[2] * 2, unet_features_dims[2], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim * 4)
                ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[1] * 3, unet_features_dims[2], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim * 4),
                Upsample(unet_features_dims[2])
                ),

            ### BLOCK 3

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[1] * 3, unet_features_dims[1], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim * 2)
                ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[1] * 2, unet_features_dims[1], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim * 2)
                ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[0] * 3, unet_features_dims[1], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim * 2),
                Upsample(unet_features_dims[1])
                ),

            ### BLOCK 4

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[0] * 3, unet_features_dims[0], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim)
                ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[1], unet_features_dims[0], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim * 2)
                ),

            SwitchSequential(
                UNet_ResidualBlock(unet_features_dims[1], unet_features_dims[0], time_emb_dim),
                UNet_AttentionBlock(num_heads, attn_dim)
                ),
        ])


        self.output_layer = nn.Sequential(
            nn.GroupNorm(32, unet_features_dims[0]),
            nn.SiLU(),
            nn.Conv2d(unet_features_dims[0], vae_latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, x : torch.Tensor, time : torch.Tensor, context : torch.Tensor):
        time = self.time_embedding(time)

        # Encoder part
        for layer in self.encoder:
            x = layer(x, context, time)

        print(x.shape)

        # Bottleneck
        x = self.bottleneck(x, context, time)
        print(x.shape)

        # Decoder part
        for layer in self.decoder:
            x = layer(x, context, time)

        
        # Output layer
        out = self.output_layer(x)

        return out


