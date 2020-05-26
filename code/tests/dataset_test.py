"""Module containing tests for dataset loading and preprocessing."""

import os
import unittest

from algorithm.dataset import ImageCaptionDataset
from torch import Tensor
from torch.utils.data import Dataset


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

    def validate_sample(self, sample, has_labels):
        """Validate a sample from an `ImageCaptionDataset` instance."""
        num_fields = 3
        if has_labels:
            num_fields += 1

        self.assertEqual(len(sample), num_fields)

        if num_fields == 4:
            image_id, images, captions, labels = sample
            self.assertIsInstance(labels, Tensor)
        else:
            image_id, images, captions = sample

        self.assertIsInstance(image_id, str)
        self.assertIsInstance(images, Tensor)

    def test_tier_train(self):
        """Assert that `ImageCaptionDataset` successfully loads training set \
        data."""
        tier = 'train'
        data = ImageCaptionDataset(self.data_dir, tier)
        self.assertIsInstance(data, Dataset)
        # Get first sample from dataset
        sample = data[0]
        self.validate_sample(sample, has_labels=True)

    def test_tier_test(self):
        """Assert that `ImageCaptionDataset` successfully loads test set \
        data."""
        tier = 'test'
        data = ImageCaptionDataset(self.data_dir, tier)
        self.assertIsInstance(data, Dataset)
        # Get first sample from dataset
        sample = data[0]
        self.validate_sample(sample, has_labels=False)

    def test_tier_train_length(self):
        """Assert that the `ImageCaptionDataset` __len__ method corresponds \
        to the actual length of the training dataset."""
        tier = 'train'
        data = ImageCaptionDataset(self.data_dir, tier)
        self.assertIsInstance(data, Dataset)

        # Get length of dataset and check boundary conditions
        length = len(data)

        with self.assertRaises(KeyError):
            _ = data[-1]

        first = data[0]
        self.validate_sample(first, has_labels=True)

        last = data[length - 1]
        self.validate_sample(last, has_labels=True)

        with self.assertRaises(KeyError):
            _ = data[length]

    def test_tier_test_length(self):
        """Assert that the `ImageCaptionDataset` __len__ method corresponds \
        to the actual length of the test dataset."""
        tier = 'test'
        data = ImageCaptionDataset(self.data_dir, tier)
        self.assertIsInstance(data, Dataset)

        # Get length of dataset and check boundary conditions
        length = len(data)

        with self.assertRaises(KeyError):
            _ = data[-1]

        first = data[0]
        self.validate_sample(first, has_labels=False)

        last = data[length - 1]
        self.validate_sample(last, has_labels=False)

        with self.assertRaises(KeyError):
            _ = data[length]


if __name__ == '__main__':
    unittest.main()
