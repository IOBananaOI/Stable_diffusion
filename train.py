import sys
sys.path.append('src/')

import os
import argparse

from src import utils
from src.model.config import StableDiffusionConfig
from src.model.diffusion import StableDiffusion

def get_args_parser():

    parser = argparse.ArgumentParser(description="Model training")

    # Weights preloading
    parser.add_argument('--weights_name', default='latest', type=str)

    return parser


def build_or_load_model(config : StableDiffusionConfig):
    model = StableDiffusion()

    if config.weights_name is not None:
        if config.weights_name == 'latest':
            weights_list = os.listdir(config.weights_folder)
            print(weights_list)
    else:
        print("Model weights were not given. Build new model")

    return model


def save_model(model : StableDiffusion):
    pass

def train_model():

    config = StableDiffusionConfig()

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[get_args_parser()])

    args = parser.parse_args()

    print(args)
    
