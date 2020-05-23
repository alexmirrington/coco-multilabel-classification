"""Module containing utilities for loading and preprocessing datasets."""
import os.path
import re
from io import StringIO

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImageCaptionDataset(Dataset):
    """Class for dataset interaction, designed for use with PyTorch."""

    TIERS = ('train', 'test')

    def __init__(self, path, tier):
        """Create a new dataset instance that loads images and captions from \
        the provided path for the given tier."""
        super().__init__()
        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, f'{tier}.csv'))
        assert tier in self.TIERS

        self.path = path
        self.tier = tier
        self.data = None
        self._init_dataset()

    def _init_dataset(self):
        # Load captions and add escape chars to aid parsing.
        with open(os.path.join(self.path, f'{self.tier}.csv')) as file:
            lines = [re.sub(
                r'([^,])"(\s*[^\n])',
                r'\1`"\2',
                line
            ) for line in file]
            self.data = pd.read_csv(
                StringIO(''.join(lines)),
                escapechar='`'
            )

    def _transform(self, img):
        """Transform a PIL image into a tensor using a series of image \
        manipulations."""
        pipeline = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.ToTensor(),
        ])
        tensor = pipeline(img)
        mean = tensor.mean(dim=(1, 2))
        std = tensor.std(dim=(1, 2))
        norm = transforms.Normalize(mean, std, inplace=True)
        return norm(tensor)

    def __getitem__(self, key):
        """Get an item from the dataset for a given index.

        Note that the index may not strictly correspond with the image id for \
        the returned record; e.g. if `tier = 'test'`, then index 0 refers to \
        image id 30000.
        """
        image_file = self.data[self.data.columns[0]][key]
        image = Image.open(os.path.join(self.path, 'data', image_file))
        image = self._transform(image)

        caption = self.data[self.data.columns[-1]][key]

        if self.tier != 'test':
            labels = self.data[self.data.columns[1]][key]
            return image, caption, labels

        return image, caption

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)
