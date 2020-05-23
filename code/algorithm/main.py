"""Main entrypoint for model training, evaluation and testing."""

import argparse
import os
import os.path
import sys
from collections.abc import Iterable

import numpy as np
import torch
from dataset import ImageCaptionDataset
from modules.faster_rcnn import FasterRCNN
from termcolor import colored
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm


def main(config):
    """Run the object detection model according to the specified config."""
    print(colored('Preprocessing...', color='cyan'))

    train_val_data = ImageCaptionDataset(config.data_dir, 'train')
    split = int(0.9*len(train_val_data))
    train_data = Subset(train_val_data, range(0, split))
    val_data = Subset(train_val_data, range(split, len(train_val_data)))
    print(f'Train: {len(train_data)}')
    print(f'Val: {len(val_data)}')

    test_data = ImageCaptionDataset(config.data_dir, 'test')
    print(f'Test: {len(test_data)}')

    model = FasterRCNN()
    if config.test:
        test(model, config, test_data)


def variable_tensor_size_collator(batch):
    """Collate a batch ready for dataloaders to allow for image \
    tensors of variable size."""
    assert isinstance(batch, Iterable)
    return list(zip(*batch))


def test(model, config, data):
    """Test a model and output results to the ouput directory."""
    print(colored('Testing...', color='cyan'))
    # Move model to device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f'Using device "{device}"')
    model = model.to(device)
    model.eval()

    # Create dataloader
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    loader = DataLoader(
        data,
        collate_fn=variable_tensor_size_collator,
        batch_size=1,
        shuffle=False,
        **loader_kwargs
    )

    path = os.path.join(config.output_dir, 'predicted_labels.txt')
    if os.path.isfile(path):
        confirmation = input(colored(
            f'Are you sure you want to replace predictions at: {path} (y/n) ',
            color='red'
        ))
        if confirmation != 'y':
            print('Aborted.')
            return

    # Run model predictions and save results
    with open(path, 'w') as f:
        f.write('ImageID,Labels\n')

    for img_ids, imgs, captions in tqdm(loader):
        imgs = [img.to(device) for img in imgs]
        results = model(imgs, captions)
        labels = [np.unique(
            result['labels'].cpu().numpy()
        ) for result in results]
        labels = [[str(lbl) for lbl in lbls if lbl < 20] for lbls in labels]
        with open(path, 'a') as f:
            for im_id, out in zip(img_ids, labels):
                string = f'{im_id},{" ".join(out)}\n'
                f.write(string)


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
    config_group.add_argument(
        '--log-dir',
        default=os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'logs'
        )),
        type=str,
        required=False,
        help='The directory to save logs to.'
    )
    model_params_group.add_argument(
        '--batch-size',
        default=4,
        type=int,
        required=False,
        help='The batch size to use when training, validating and testing.'
    )
    model_params_group.add_argument(
        '--epochs',
        default=1,
        type=int,
        required=False,
        help='The number of epochs to train the model for.'
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
