"""Module containing tests for dataset loading and preprocessing."""

import os
import unittest
from algorithm.dataset import ImageCaptionDataset
from torch.utils.data import Dataset, DataLoader
from torch import Tensor


class TestImageCaptionDataset(unittest.TestCase):
    """Class containing tests related to the `ImageCaptionDataset` class."""

    @classmethod
    def setUpClass(cls):
        """Perform setup operations for data loading."""
        cls.data_dir = os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'input'
        ))

    def test_tier_train(self):
        """Assert that `ImageCaptionDataset` successfully loads training set \
        data using a `DataLoader`."""
        tier = 'train'
        data = ImageCaptionDataset(self.data_dir, tier)
        self.assertIsInstance(data, Dataset)
        loader = DataLoader(data)
        sample = next(iter(loader))
        # Ensure labels were returned
        self.assertEqual(len(sample), 3)
        # Check image output of data loader
        images, _, _ = sample
        self.assertIsInstance(images, Tensor)

    def test_tier_test(self):
        """Assert that `ImageCaptionDataset` successfully loads training set \
        data using a `DataLoader`."""
        tier = 'test'
        data = ImageCaptionDataset(self.data_dir, tier)
        self.assertIsInstance(data, Dataset)
        loader = DataLoader(data)
        sample = next(iter(loader))
        # Ensure there are no labels
        self.assertEqual(len(sample), 2)
        # Check image output of data loader
        images, _ = sample
        self.assertIsInstance(images, Tensor)


if __name__ == '__main__':
    unittest.main()
