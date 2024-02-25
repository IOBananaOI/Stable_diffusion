import sys
sys.path.insert(0, 'model/')

from model.config import StableDiffusionConfig

from pathlib import Path

import math

import torch
from torch import nn
import torch.nn.functional as F

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
    def __init__(self, num_heads : int, emb_dim : int, context_dim : int, in_proj_bias=False, out_proj_bias=False) -> None:
        super().__init__()

        self.Q = nn.Linear(emb_dim, emb_dim, bias=in_proj_bias)
        self.K = nn.Linear(context_dim, emb_dim, bias=in_proj_bias)
        self.V = nn.Linear(context_dim, emb_dim, bias=in_proj_bias)
        self.out_layer = nn.Linear(emb_dim, emb_dim, bias=out_proj_bias)
        self.num_heads = num_heads
        self.d_head = emb_dim // num_heads

    def forward(self, x : torch.Tensor, context : torch.Tensor):
        input_shape = x.shape

        batch_size, seq_len, emb_dim = input_shape

        attn_shape = (batch_size, -1, self.num_heads, self.d_head)

        q = self.Q(x)
        k = self.K(context)
        v = self.V(context)

        q = q.view(attn_shape).transpose(1, 2)
        k = k.view(attn_shape).transpose(1, 2)
        v = v.view(attn_shape).transpose(1, 2)

        weights = q @ k.transpose(-1, -2)

        weights /= math.sqrt(self.d_head)

        weights = F.softmax(weights, dim=-1)

        output = weights @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_layer(output)

        return output


def get_alphas_cumprod(beta_start=0.00085, beta_end=0.0120, n_training_steps=1000):
    betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, n_training_steps, dtype=np.float32) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    return alphas_cumprod


class KLMSSampler():
    def __init__(self, n_inference_steps=50, n_training_steps=1000, lms_order=4):
        timesteps = np.linspace(n_training_steps - 1, 0, n_inference_steps)

        alphas_cumprod = get_alphas_cumprod(n_training_steps=n_training_steps)
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        log_sigmas = np.log(sigmas)
        log_sigmas = np.interp(timesteps, range(n_training_steps), log_sigmas)
        sigmas = np.exp(log_sigmas)
        sigmas = np.append(sigmas, 0)
        
        self.sigmas = sigmas
        self.initial_scale = sigmas.max()
        self.timesteps = timesteps
        self.n_inference_steps = n_inference_steps
        self.n_training_steps = n_training_steps
        self.lms_order = lms_order
        self.step_count = 0
        self.outputs = []

    def get_input_scale(self, step_count=None):
        if step_count is None:
            step_count = self.step_count
        sigma = self.sigmas[step_count]
        return 1 / (sigma ** 2 + 1) ** 0.5

    def set_strength(self, strength=1):
        start_step = self.n_inference_steps - int(self.n_inference_steps * strength)
        self.timesteps = np.linspace(self.n_training_steps - 1, 0, self.n_inference_steps)
        self.timesteps = self.timesteps[start_step:]
        self.initial_scale = self.sigmas[start_step]
        self.step_count = start_step

    def step(self, latents, output):
        t = self.step_count
        self.step_count += 1

        self.outputs = [output] + self.outputs[:self.lms_order - 1]
        order = len(self.outputs)
        for i, output in enumerate(self.outputs):
            
            x = np.linspace(self.sigmas[t], self.sigmas[t + 1], 81)
            y = np.ones(81)
            for j in range(order):
                if i == j:
                    continue
                y *= x - self.sigmas[t - j]
                y /= self.sigmas[t - i] - self.sigmas[t - j]
            lms_coeff = np.trapz(y=y, x=x)
            latents += lms_coeff * output
        return latents



        


