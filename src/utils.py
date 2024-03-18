import sys
sys.path.insert(0, 'model/')

from typing import List

from model.config import StableDiffusionConfig
from model.tokenizer import Tokenizer

from pathlib import Path

import math

import torch
import torch.nn.functional as F

from torch import nn
from torchvision.transforms import Compose, Lambda, ToPILImage, ToTensor

import numpy as np


def get_weights_model_path(config: StableDiffusionConfig, epoch : int):
    model_filename = f"{config.model_name}{epoch}.pt"
    return str(Path('.') / config.weights_folder / model_filename)


def latest_weights_path(config: StableDiffusionConfig):
    """
    Find the latest weights in weights_folder.
    """
    weights_files = list(Path(config.weights_folder).glob(f"{config.model_name}*"))

    if len(weights_files) == 0:
        return None
    
    weights_files.sort()
    return str(weights_files[-1])


def get_index(vals, t, tensor_shape, device):
    t_batch_size = t.shape[0]

    out = vals.gather(-1, t)
    
    return out.reshape(t_batch_size, *((1,) * (len(tensor_shape) - 1))).to(device)


class SelfAttention(nn.Module):
    def __init__(self, num_heads : int, emb_dim : int, in_proj_bias=True, out_proj_bias=True) -> None:
        super().__init__()

        self.in_projection_layer = nn.Linear(emb_dim, emb_dim * 3, bias=in_proj_bias)
        self.out_projection_layer = nn.Linear(emb_dim, emb_dim, bias=out_proj_bias)
        self.num_heads = num_heads

        self.d_head = emb_dim // num_heads

    
    def forward(self, x : torch.Tensor, causal_mask=False):
        # x: (batch_size, seq_len, emb_dim)

        inp_shape = x.shape
        batch_size, seq_len, emb_dim = inp_shape

        attn_shape = (batch_size, seq_len, self.num_heads, self.d_head)

        # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim * 3) -> 3 tensors of (batch_size, seq_len, emb_dim) 
        q, k, v = self.in_projection_layer(x).chunk(3, dim=-1)

        # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, h, emb_dim // h) -> (batch_size, h, seq_len, emb_dim // h) 
        q = q.view(attn_shape).transpose(1, 2)
        k = k.view(attn_shape).transpose(1, 2)
        v = v.view(attn_shape).transpose(1, 2)

        weights = q @ k.transpose(-1, -2)

        if causal_mask:
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill(mask, -torch.inf)
        
        weights /= math.sqrt(self.d_head)

        weights = F.softmax(weights, dim=-1)

        # (batch_size, h, seq_len, seq_len) @ (batch_size, h, seq_len, emb_dim // h) -> (batch_size, h, seq_len, emb_dim / h)
        output = weights @ v

        # (batch_size, h, seq_len, emb_dim / h) -> (batch_size, seq_len, h, emb_dim / h)
        output = output.transpose(1, 2)

        output = output.reshape(inp_shape)

        output = self.out_projection_layer(output)

        return output
    

class CrossAttention(nn.Module):
    def __init__(self, num_heads : int, emb_dim : int, contenoised_img_dim : int, in_proj_bias=False, out_proj_bias=False) -> None:
        super().__init__()

        self.Q = nn.Linear(emb_dim, emb_dim, bias=in_proj_bias)
        self.K = nn.Linear(contenoised_img_dim, emb_dim, bias=in_proj_bias)
        self.V = nn.Linear(contenoised_img_dim, emb_dim, bias=in_proj_bias)
        self.out_layer = nn.Linear(emb_dim, emb_dim, bias=out_proj_bias)
        self.num_heads = num_heads
        self.d_head = emb_dim // num_heads

    def forward(self, x : torch.Tensor, contenoised_img : torch.Tensor):
        input_shape = x.shape
        batch_size, seq_len, emb_dim = input_shape

        attn_shape = (batch_size, -1, self.num_heads, self.d_head)
        k_shape = (contenoised_img.shape[0], -1, self.num_heads, self.d_head)

        q = self.Q(x)
        k = self.K(contenoised_img)
        v = self.V(contenoised_img)

        q = q.view(attn_shape).transpose(1, 2)
        k = k.view(k_shape).transpose(1, 2)
        v = v.view(k_shape).transpose(1, 2)

        weights = q @ k.transpose(-1, -2)

        weights /= math.sqrt(self.d_head)

        weights = F.softmax(weights, dim=-1)

        output = weights @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_layer(output)

        return output


def forward_diffusion(config : StableDiffusionConfig, x0 : torch.Tensor, t):

    sqrt_alphas_cumprod_t = get_index(config.sqrt_alphas_cumprod, t, x0.shape, config.device)
    sqrt_one_minus_alphas_cumprod_t = get_index(config.sqrt_one_minus_alphas_cumprod, t, x0.shape, config.device)
    
    eps = torch.randn_like(x0.float()).to(config.device)

    return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * eps, eps


def remove_noise(
        config : StableDiffusionConfig, 
        model, 
        t : torch.tensor, 
        noised_img : torch.tensor,
        caption : List[str],
        tokenizer : Tokenizer
    ):
    
    betas_t = get_index(config.betas, t, noised_img.shape, config.device)

    sqrt_one_minus_alphas_cumprod_t = get_index(config.sqrt_one_minus_alphas_cumprod, t, noised_img.shape, config.device)
    
    sqrt_recip_alphas_t = get_index(config.sqrt_recip_alphas, t, noised_img.shape, config.device)

    tokens = torch.tensor(tokenizer.encode(caption)).unsqueeze(0)

    model_output = model(noised_img, tokens, t, config.do_cfg)

    model_mean = sqrt_recip_alphas_t * (
        noised_img - betas_t * model_output / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index(config.posterior_variance, t, noised_img.shape, config.device)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(noised_img)
        return model_mean + torch.sqrt(posterior_variance_t) * noise
    

def convert_tensor_to_image(image):  
    reverse_transforms = Compose([
        Lambda(lambda t: (t + 1) / 2),
        Lambda(lambda t: t.permute(1, 2, 0)),
        Lambda(lambda t: t * 255.),
        Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
        ToPILImage(),
    ])

    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


@torch.no_grad()
def get_sample(config : StableDiffusionConfig, model, caption, n_imgs=1):
    sample = ['']

    x_tm1 = torch.randn((1, config.img_channels, config.img_size, config.img_size), device=config.device, dtype=torch.float32)

    for i in range(config.T-1, -1, -1):
        x_tm1 = remove_noise(config, model, torch.tensor([i]).to(config.device), x_tm1, caption, Tokenizer(config)).to(torch.float32)
        x_tm1 = torch.clamp(x_tm1, -1.0, 1.0)

        if n_imgs == 'all':
            sample.append(convert_tensor_to_image(x_tm1))
        else:
            sample[0] = x_tm1

    if n_imgs == 1:
        sample[0] = convert_tensor_to_image(sample[0])

    return sample
