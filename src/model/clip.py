import sys
sys.path.append('../')

import torch
from torch import nn

from utils import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size : int, emb_dim : int, seq_len : int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.positional_embedding = nn.Parameter(torch.zeros(seq_len, emb_dim))

    def forward(self, tokens):
        # (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        x = self.token_embedding(tokens)

        # (batch_size, seq_len, emb_dim) -> (batch_size, seq_len, emb_dim)
        print(x.shape, self.positional_embedding.shape)

        x += self.positional_embedding

        return x


class CLIPLayer(nn.Module):
    def __init__(self, emb_dim : int, num_heads : int, emb_dim_scale_factor : int, dropout : float) -> None:
        super().__init__()

        self.norm_1 = nn.LayerNorm(emb_dim)
        self.attn_layer = SelfAttention(num_heads, emb_dim)
        self.norm_2 = nn.LayerNorm(emb_dim)
        self.linear_1 = nn.Linear(emb_dim, emb_dim * emb_dim_scale_factor)
        self.linear_2 = nn.Linear(emb_dim * emb_dim_scale_factor, emb_dim)

    def forward(self, x):
        resid = x

        x = self.norm_1(x)

        x = self.attn_layer(x, causal_mask=True)

        x += resid

        resid = x

        x = self.norm_2(x)

        x = self.linear_1(x)

        x = x * torch.sigmoid(1.702 * x) # QuickGELU

        x = self.linear_2(x)

        x += resid

        return x


class CLIPEncoder(nn.Module):
    def __init__(self, vocab_size : int, emb_dim : int, seq_len : int, 
                 num_heads : int, emb_dim_scale_factor : int, 
                 num_layers : int, dropout : float) -> None:
        super().__init__()

        self.embedding_layer = CLIPEmbedding(vocab_size, emb_dim, seq_len)

        self.layers = nn.ModuleList([
            CLIPLayer(emb_dim, num_heads, emb_dim_scale_factor, dropout) for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(emb_dim)

    
    def forward(self, tokens : torch.Tensor):
        tokens = tokens.type(torch.long).to('cuda')

        # (batch_size, seq_len) -> (batch_size, seq_len, emb_dim)
        emb = self.embedding_layer(tokens)

        for layer in self.layers:
            emb = layer(emb)

        # (batch_size, seq_len, emb_dim)
        out = self.norm(emb)

        return out