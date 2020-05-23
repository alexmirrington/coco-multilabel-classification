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

    general_group = parser.add_argument_group('General')
    model_params_group = parser.add_argument_group('Model parameters')
    config_group = parser.add_argument_group('Configuration')

    general_group.add_argument(
        '--debug',
        action='store_true',
        help='Print additional debugging information.'
    )
    general_group.add_argument(
        '--test',
        action='store_true',
        help='Evaluate the model on the test set and save predictions \
            to the output directory as specified by --output-dir.'
    )
    general_group.add_argument(
        '--load',
        type=str,
        required=False,
        help='Load model weights from the specified file.'
    )
    general_group.add_argument(
        '--save',
        type=str,
        required=False,
        help='Save the model weights to the specified file.'
    )
    config_group.add_argument(
        '--data-dir',
        default=os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'input'
        )),
        type=str,
        required=False,
        help='The directory to load the dataset from.'
    )
    config_group.add_argument(
        '--output-dir',
        default=os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'output'
        )),
        type=str,
        required=False,
        help='The directory to save test set predictions to.'
    )
    model_params_group.add_argument(
        '--batch-size',
        default=64,
        type=int,
        required=False,
        help='The batch size to use when training, validating and testing.'
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
