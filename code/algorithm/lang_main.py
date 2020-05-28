"""Placeholder file until models other than RCNN are in main."""
import sys

import torch
from dataset import ImageCaptionDataset
from main import load, parse_args, test, train
from metrics import MetricCollection, macro_f1, micro_f1, weighted_f1
from modules.rcnn_lstm import RCNN_LSTM_Bilinear
from preprocessing import preprocess_caption
from termcolor import colored
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import Subset
from torchtext.vocab import GloVe


def main(config):
    """Run the object detection model according to the specified config."""
    print(colored('Environment:', attrs=['bold', ]))
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f'torch: {torch.__version__}')
    if cuda:
        print(f'{device}: {torch.cuda.get_device_name(device)}')
    else:
        print('Using CPU')
    print(colored('Preprocessing...', color='cyan', attrs=['bold', ]))

    embeddings = GloVe(name='6B', dim=100)

    train_val_data = ImageCaptionDataset(config.data_dir,
                                         'train',
                                         embeddings=embeddings,
                                         preprocessor=preprocess_caption)
    split = int(0.9*len(train_val_data))
    train_data = Subset(train_val_data, range(0, split))
    val_data = Subset(train_val_data, range(split, len(train_val_data)))

    print(f'Train: {len(train_data)}')
    print(f'Val: {len(val_data)}')

    test_data = ImageCaptionDataset(config.data_dir,
                                    'test',
                                    embeddings=embeddings,
                                    preprocessor=preprocess_caption)
    print(f'Test: {len(test_data)}')

    # Set up metrics to collect
    metrics = MetricCollection(metrics={
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    })

    if config.load:
        # Load model from file
        try:
            model, optimiser, loss_func, epoch = load(config)
        except ValueError as e:
            print(colored(e.args[0], color='red'))
            return
        train(model, optimiser, loss_func, train_data, val_data,
              config, metrics, epoch)
    else:
        # Example configuration, TODO migrate to model factory
        model = RCNN_LSTM_Bilinear(ImageCaptionDataset.CLASSES)

        loss_func = BCEWithLogitsLoss()
        optimiser = Adam(model.parameters())
        train(model, optimiser, loss_func, train_data, val_data,
              config, metrics)
    if config.test:
        test(model, config, test_data)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
