"""Module containing utilities for loading and preprocessing datasets."""
import os.path
import re
from collections.abc import Iterable
from io import StringIO

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from utilities import binarise_labels

# from torchtext.vocab import GloVe


def variable_tensor_size_collator(batch):
    """Collate a batch ready for dataloaders to allow for image \
    tensors of variable size."""
    assert isinstance(batch, Iterable)
    transpose = list(zip(*batch))
    for idx, field in enumerate(transpose):
        try:
            transpose[idx] = torch.stack(field)
        except Exception:
            continue
    return transpose


class ImageCaptionDataset(Dataset):
    """Class for dataset interaction, designed for use with PyTorch."""

    TIERS = ('train', 'test')
    CLASSES = tuple(range(20))

    def __init__(
            self,
            path,
            tier,
            embeddings=None,
            preprocessor=None,
            transform=True):
        """Create a new dataset instance that loads images and captions from \
        the provided path for the given tier."""
        super().__init__()
        assert os.path.isdir(path)
        assert os.path.exists(os.path.join(path, f'{tier}.csv'))
        assert tier in self.TIERS

        self.path = path
        self.tier = tier
        self.data = None
        self.embeddings = embeddings
        self.preprocessor = preprocessor
        self.is_transform = transform
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

        # Preprocess captions
        caption_col = self.data.columns[-1]
        if self.preprocessor is not None:
            self.data[caption_col] = self.data[caption_col].apply(
                self.preprocessor
            )

        # Preprocess labels
        if self.tier != self.TIERS[-1]:
            lbl_col = self.data.columns[1]
            self.data[lbl_col] = self.data[lbl_col].apply(
                lambda lbls: [int(lbl) for lbl in lbls.split()]
            )
            binarised, _ = binarise_labels(
                list(self.data[lbl_col]),
                classes=self.CLASSES
            )
            binarised = [torch.Tensor(vec) for vec in binarised]
            self.data[lbl_col] = binarised

    def __getitem__(self, key):
        """Get an item from the dataset for a given index.

        Note that the index may not strictly correspond with the image id for \
        the returned record; e.g. if `tier = 'test'`, then index 0 refers to \
        image id 30000.
        """
        # Get images and apply transforms
        image_file = self.data[self.data.columns[0]][key]
        image = Image.open(os.path.join(self.path, 'data', image_file))
        if self.is_transform:
            image = self.transform(image)

        if not torch.is_tensor(image):
            image = transforms.ToTensor()(image)

        # Get caption embeddings
        caption = self.data[self.data.columns[-1]][key]
        if self.embeddings is not None:
            caption = [self.embeddings[word] for word in caption]

        if self.tier != self.TIERS[-1]:
            labels = self.data[self.data.columns[1]][key]
            return image_file, image, caption, labels

        return image_file, image, caption

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def transform(self, image):
        """Transform image so that it can be properly used as input tensors."""
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        input_tensor = preprocess(image)
        return input_tensor
