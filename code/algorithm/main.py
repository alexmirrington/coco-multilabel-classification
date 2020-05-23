"""Main entrypoint for model training, evaluation and testing."""

import argparse
import os.path
import sys

from dataset import ImageCaptionDataset
from torch.utils.data import Subset


def main(config):
    """Run the object detection model according to the specified config."""
    train_val_data = ImageCaptionDataset(config.data_dir, 'train')
    test_data = ImageCaptionDataset(config.data_dir, 'test')

    split = int(0.8*len(train_val_data))
    train_data = Subset(train_val_data, range(0, split))
    val_data = Subset(train_val_data, range(split, len(train_val_data)))

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))


def parse_args(args):
    """Parse the arguments from the command line, and return a config object that \
    stores the values of each config parameter."""
    parser = argparse.ArgumentParser()
    model_params_group = parser.add_argument_group('Model parameters')
    config_group = parser.add_argument_group('Configuration')

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print additional debugging information.'
    )
    config_group.add_argument(
        '--data-dir',
        default=os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'input'
        )),
        help='The directory to load the dataset from.'
    )
    model_params_group.add_argument(
        '--batch-size',
        default=64,
        help='The batch size to use when training, validating and testing.'
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
