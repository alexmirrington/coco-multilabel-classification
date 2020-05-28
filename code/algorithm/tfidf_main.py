"""Placeholder file until models other than RCNN are in main."""
import sys

import torch
from dataset import ImageCaptionDataset
from main import load, parse_args, test, train
from metrics import MetricCollection, macro_f1, micro_f1, weighted_f1
from modules.rcnn_lstm import RCNN_LSTM_Bilinear
from modules.tfidf import TFIDF
from preprocessing import preprocess_caption
from sklearn.feature_extraction.text import TfidfVectorizer
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


    train_val_data = ImageCaptionDataset(config.data_dir,
                                         'train')

    split = int(0.9*len(train_val_data))
    train_data = Subset(train_val_data, range(0, split))
    val_data = Subset(train_val_data, range(split, len(train_val_data)))
    print(f'Train: {len(train_data)}')
    print(f'Val: {len(val_data)}')

    test_data = ImageCaptionDataset(config.data_dir,
                                    'test')
    print(f'Test: {len(test_data)}')

    # Set up metrics to collect
    metrics = MetricCollection(metrics={
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1
    })


    # Get captions in training data as a list of captions
    caption_list = list(train_val_data.data[train_val_data.data.columns[-1]])
    # Learn tfidf vectoriser from the train data
    tfidf_vectorizer = TfidfVectorizer(max_features=500,
                                       use_idf=True,
                                       tokenizer=preprocess_caption)
    tfidf_vectorizer.fit(caption_list)

    # if config.load:
    #     # Load model from file
    #     try:
    #         model, optimiser, loss_func, epoch = load(config)
    #     except ValueError as e:
    #         print(colored(e.args[0], color='red'))
    #         return
    #     train(model, optimiser, loss_func, train_data, val_data,
    #           config, metrics, epoch)
    # else:
        # Example configuration, TODO migrate to model factory
    model = TFIDF(ImageCaptionDataset.CLASSES, tfidf_vectorizer)

    loss_func = BCEWithLogitsLoss()
    optimiser = Adam(model.parameters())
    train(model, optimiser, loss_func, train_data, val_data,
          config, metrics)
    if config.test:
        test(model, config, test_data)


# REDEFINE LOAD FOR THIS MODEL!

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
