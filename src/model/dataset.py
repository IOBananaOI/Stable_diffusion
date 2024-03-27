import json

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .tokenizer import Tokenizer

from torchvision.transforms import ToTensor, Compose, Resize

from PIL import Image

from .config import StableDiffusionConfig

class SD_Dataset(Dataset):
    def __init__(self, config : StableDiffusionConfig) -> None:
        super().__init__()

        captions_json = json.load(open(config.captions_path.resolve()))

        self.img_path = config.img_path
        self.images = captions_json['images'][:4]
        self.captions = np.array(captions_json['annotations'])
        self.tokenizer = Tokenizer(config)

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
        else:
            caption =  ""

        # Get image
        file_name = self.images[ind]['file_name']
        img = Image.open(self.img_path.resolve() / file_name).convert("RGB")

        # Apply transformation
        img = self.transformation(img)

        tokens = self.tokenizer.encode(caption)

        return tokens, img
        

def get_dataloader(config : StableDiffusionConfig):
    dataset = SD_Dataset(config)

    dataloader = DataLoader(dataset, config.batch_size, shuffle=True, drop_last=True)

    return dataloader
