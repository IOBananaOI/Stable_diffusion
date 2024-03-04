import json

import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from torchvision.transforms import ToTensor, Compose, Resize

from PIL import Image

from .config import StableDiffusionConfig

class SD_Dataset(Dataset):
    def __init__(self, config : StableDiffusionConfig) -> None:
        super().__init__()

        captions_json = json.load(open(config.captions_path.resolve()))

        self.img_path = config.img_path
        self.images = captions_json['images']
        self.captions = np.array(captions_json['annotations'])

        self.transformation = Compose([Resize((256, 256)), ToTensor()])


    def __len__(self):
        return len(self.images)


    def __getitem__(self, ind):
        # Get caption
        img_id = self.images[ind]['id']
        condition_vec = np.vectorize(lambda x: x['id'] == img_id)
        founded_captions = self.captions[condition_vec(self.captions)]

        if founded_captions:
            caption = founded_captions[0]['caption']
            captioned = True
        else:
            caption =  [""]
            captioned = False

        # Get image
        file_name = self.images[ind]['file_name']
        img = Image.open(self.img_path.resolve() / file_name)

        # Apply transformation
        img = self.transformation(img)

        return caption, img
        

def get_dataloader(config : StableDiffusionConfig):
    dataset = SD_Dataset(config)

    dataloader = DataLoader(dataset, config.batch_size, shuffle=True)

    return dataloader
