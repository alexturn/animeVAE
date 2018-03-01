import torch

from torch.utils.data import Dataset
from skimage.io import imread
from skimage.transform import resize

import os
import pandas as pd


class AnimeFaceDataset(Dataset):
    """Anime Face dataset."""

    def __init__(self, ann_list, img_dir, size=(64,64)):
        """
        Args:
            csv_file (string): Path to the file with annotations.
            root_dir (string): Directory with all the images.
        """
        self.anime_names = []
        self.img_dir = img_dir
        with open(ann_list, 'r') as input_file:
            for line in input_file:
                self.anime_names.append(os.path.join(self.img_dir,
                                        line[:-1]))
        self.size = size

    def __len__(self):
        return len(self.anime_names)

    def __getitem__(self, idx):
        img_name = self.anime_names[idx]
        image = imread(img_name)
        image = resize(image, self.size, mode='constant')
        image = image.transpose(2,0,1)

        return image


class CelebDataset(Dataset):
    """CelebA Dataset"""

    def __init__(self, path=None):
        """
        Args:
            path (string): path to images folder
        """
        if path == None: raise "No path to images sprcified"
        self.imgs = []
        for img_name in os.listdir(path):
            self.imgs.append(os.path.join(path, img_name))

        try: imread(self.imgs[-1])
        except: raise "Wrong image dir"

        self.x_start, self.x_shift = 30, 138
        self.y_start, self.y_shift = 40, 138
        self.size = (64, 64)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = imread(self.imgs[idx])
        img = img[self.y_start:self.y_start+self.y_shift,
                  self.x_start:self.x_start+self.x_shift,:]
        img = resize(img, self.size, mode='constant')
        img = img.transpose(2,0,1)

        return img

