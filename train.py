import sys
sys.path.append('src/')

import os
import argparse

import torch

from src.model.config import StableDiffusionConfig
from src.model.diffusion import StableDiffusion


def get_args_parser():

    parser = argparse.ArgumentParser(description="Model training", add_help=False)

    # Weights preloading
    parser.add_argument('--weights_name', default='latest', type=str)

    return parser


def build_or_load_model(config : StableDiffusionConfig):
    model = StableDiffusion(config)

    if config.weights_name is not None:
        weights_folder = 'src/model/' + config.weights_folder
        weights_list = os.listdir(weights_folder)

        if config.weights_name == 'latest':
            weights_list.sort()
            weights_name = weights_list[-1]

        elif config.weights_name in weights_list:
            weights_name = config.weights_name

        else:
            raise KeyError("No weights with the given name found.")
        
        state_dict = torch.load(weights_folder + weights_name)

        model.load_state_dict(state_dict['model_state_dict'])
    else:
        print("Model weights were not given. Build new model")

    return model


def save_model(model : StableDiffusion):
    pass

def train_model(config : StableDiffusionConfig):
    pass
    

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    config = StableDiffusionConfig(weights_name=args.weights_name)

    print(config.weights_name)

    

    print(args)

    build_or_load_model(config)
    
