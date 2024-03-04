import sys
sys.path.append('src/')

import os
import argparse
from tqdm import tqdm

import neptune

import torch
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader

from src.model.config import StableDiffusionConfig
from src.model.diffusion import StableDiffusion
from src.utils import forward_diffusion, get_sample
from src.model.tokenizer import Tokenizer
from src.model.dataset import get_dataloader


def get_args_parser():

    parser = argparse.ArgumentParser(description="Model training", add_help=False)

    # Weights preloading
    parser.add_argument('--weights_name', type=str)

    # Training arguments
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--preload', type=bool)
    parser.add_argument('--to_eval', type=bool)

    # Classifier free guidance
    parser.add_argument('--do_cfg', type=bool)
    parser.add_argument('--cfg_scale', type=float)

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


def save_weights(config : StableDiffusionConfig, model : StableDiffusion, optimizer : Optimizer, lr : float, epoch : int):
    state_dict = {
        'model_state_dict' : model.state_dict(),
        'optimizer_state_dict' : optimizer.state_dict(),
        'lr' : lr,
        'epoch' : epoch
    }

    torch.save(state_dict, config.weights_folder + config.model_name + str(epoch) + 'pth')


def train_model(
        config : StableDiffusionConfig,
        model : StableDiffusion,
        optimizer : Optimizer,
        criterion,
        dataloader : DataLoader
    ):
    # Weights preloading
    if config.preload:
        model, optimizer, start_epoch = load_weights(config, model, optimizer)
    else:
        start_epoch = 0

    run = neptune.init_run(
    project=config.neptune_project_name,
    with_id=config.neptune_run_id,
    api_token=config.neptune_project_api_token,
    name="stable_diffusion"
    )

    run['parameters'] = {
        "init_lr" : config.lr, 
        "optimizer" : "Adam",
        "batch_size" : config.batch_size
    }
    
    # Iterate over epochs from initial one
    for epoch in range(start_epoch, config.num_epochs):
        with tqdm(dataloader, unit='batch') as tepoch:
            for batch in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")

                # Get img tensors from batch
                img_tensor = batch[1].to(config.device, dtype=torch.float32)

                # Get captions
                tokens = torch.stack(batch[0], dim=1)


                # Create t for each image in batch for the following forward diffusion operation
                t = torch.randint(0, config.T, (config.batch_size,)).long().to(config.device)

                # Make forward diffusion and get added noise and noised_images
                noised_img, noise = forward_diffusion(config, img_tensor, t)

                noised_img.to(config.device)

                # Model output
                predicted_noise = model(noised_img, tokens, t, config.do_cfg)

                # Get loss
                loss = criterion(noise, predicted_noise)

                # Neptune tracking
                run["loss"].append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                break

        # Evaluation
        if config.to_eval: 
            model.eval()
            generated_img = get_sample(config, model, config.eval_caption, n_imgs=1)[0]

            run["inference"].upload(generated_img)

        # Model saving
        if config.saving_strategy == 'all':
            save_weights(config, model, optimizer, optimizer['lr'], epoch)


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
    model, optimizer, start_epoch = load_weights(config, model, optimizer)
    
    # Define criterion
    criterion = torch.nn.SmoothL1Loss()

    # Get dataloader
    dataloader = get_dataloader(config)

    # Train the model
    train_model(config, model, optimizer, criterion, dataloader)