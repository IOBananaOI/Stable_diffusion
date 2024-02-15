import sys
sys.path.insert(0, 'model/')

from model.config import Stable_diffusion_config

from pathlib import Path

def get_weights_model_path(config: Stable_diffusion_config, epoch : int):
    model_filename = f"{config.model_name}{epoch}.pt"
    return str(Path('.') / config.weights_folder / model_filename)


def latest_weights_path(config: Stable_diffusion_config):
    """
    Find the latest weights in weights_folder.
    """
    weights_files = list(Path(config.weights_folder).glob(f"{config.model_name}*"))

    if len(weights_files) == 0:
        return None
    
    weights_files.sort()
    return str(weights_files[-1])