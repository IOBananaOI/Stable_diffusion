import sys
sys.path.append('src/')

import os
import argparse
from tqdm import tqdm

import torch
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from src.model.config import StableDiffusionConfig
from src.model.diffusion import StableDiffusion
from src.utils import forward_diffusion
from src.model.tokenizer import Tokenizer


def get_args_parser():

    parser = argparse.ArgumentParser(description="Model training", add_help=False)

    # Weights preloading
    parser.add_argument('--weights_name', default='latest', type=str)

    # Training arguments
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)

    # Neptune settings
    parser.add_argument('--neptune_run_id', type=str)
    

    return parser


def build_config(args):
    """
    Create new config object with respect to given arguments in command line.
    """
    config = StableDiffusionConfig()

    for k, v in args.__dict__.items():
        if v:
            setattr(config, k, v)

    return config


def load_weights(
        config : StableDiffusionConfig, 
        model : StableDiffusion,
        optimizer : Optimizer
    ):

    epoch = 0

    if config.weights_name is not None:
        weights_folder = 'src/model/' + config.weights_folder
        weights_list = os.listdir(weights_folder)

        if config.weights_name == 'latest' and len(weights_list) > 0:
            weights_list.sort()
            weights_name = weights_list[-1]

        elif config.weights_name in weights_list:
            weights_name = config.weights_name

        else:
            raise KeyError("No weights with the given name found.")
        
        state_dict = torch.load(weights_folder + weights_name)

        model.load_state_dict(state_dict['model_state_dict'])
        optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        epoch = state_dict['epoch']
    else:
        print("Model weights were not given.")

    return model, optimizer, epoch


def save_model(config : StableDiffusionConfig, model : StableDiffusion, optimizer : Optimizer, epoch : int):
    state_dict = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'epoch' : epoch
    }

    torch.save(state_dict, config.weights_folder + config.model_name + str(epoch) + 'pth')


def train_model(
        config : StableDiffusionConfig,
        model : StableDiffusion,
        optimizer : Optimizer,
        dataloader : DataLoader,
        num_epochs : int,
        init_epoch : int
    ):
    
    for epoch in range(init_epoch, num_epochs):

        with tqdm(dataloader, unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                # Get img tensors from batch
                img_tensor = batch[0].to(config.device, dtype=torch.float32)

                # Make tokenization for image captions
                tokenizer = Tokenizer(config)
                tokens = tokenizer.encode_batch(batch[1])

                # Create t for each image in batch for the following forward diffusion operation
                t = torch.randint(0, config.T, (config.batch_size,)).long().to(config.device)

                # Make forward diffusion and get added noise and noised_images
                noised_img, noise = forward_diffusion(img_tensor, t)

    pass
    

if __name__ == '__main__':

    # Get arguments from command line    
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    # Create config object with the given in command line arguments
    config = build_config(args)

    # Build new model
    model = StableDiffusion(config)

    # Create optimizer instance
    optimizer = Adam(model.parameters(), lr=config.lr)

    # Preloading previous weights for the following training
    model, optimizer, init_epoch = load_weights(config, model, optimizer)
    
