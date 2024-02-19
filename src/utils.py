import sys
sys.path.insert(0, 'model/')

from model.config import StableDiffusionConfig

from pathlib import Path

from torch import nn


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
        self.out_projection_layer = nn.Linear
