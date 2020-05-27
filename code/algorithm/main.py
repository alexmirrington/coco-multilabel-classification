"""Main entrypoint for model training, evaluation and testing."""

import argparse
import os
import os.path
import sys
import time

import torch
from dataset import ImageCaptionDataset, variable_tensor_size_collator
from metrics import MetricCollection, macro_f1, micro_f1, weighted_f1
from modules.faster_rcnn import FasterRCNN
from termcolor import colored
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from utilities import log_metrics_file, log_metrics_stdout


def main(config):
    """Run the object detection model according to the specified config."""
    print(colored('Environment:', attrs=['bold', ]))
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f'torch: {torch.__version__}')
    print(f'{device}: {torch.cuda.get_device_name(device)}')
    print(colored('Preprocessing...', color='cyan', attrs=['bold', ]))

    train_val_data = ImageCaptionDataset(config.data_dir, 'train')
    split = int(0.9*len(train_val_data))
    train_data = Subset(train_val_data, range(0, split))
    val_data = Subset(train_val_data, range(split, len(train_val_data)))
    print(f'Train: {len(train_data)}')
    print(f'Val: {len(val_data)}')

    test_data = ImageCaptionDataset(config.data_dir, 'test')
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
        model = FasterRCNN(ImageCaptionDataset.CLASSES)
        loss_func = BCEWithLogitsLoss()
        optimiser = Adam(model.parameters())
        train(model, optimiser, loss_func, train_data, val_data,
              config, metrics)
    if config.test:
        test(model, config, test_data)


def load(config):
    """Load a model from file."""
    print(colored('Loading model...', color='cyan', attrs=['bold', ]))

    path = config.load
    if os.path.exists(os.path.join(config.checkpoint_dir, config.load)):
        path = os.path.join(config.checkpoint_dir, config.load)

    try:
        checkpoint = torch.load(path)
    except IOError as e:
        raise ValueError(f'Could not load model from file: {path}', e)

    model_class = checkpoint['model_class']
    optimiser_class = checkpoint['optimiser_class']

    model = model_class(ImageCaptionDataset.CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Move model to device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model = model.to(device)

    optimiser = optimiser_class(model.parameters())
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    return model, optimiser, loss, epoch


def save(model, optimiser, loss, config, epoch):
    """Save a model to file."""
    print(colored('Saving model...', color='cyan', attrs=['bold', ]))
    path = os.path.join(config.checkpoint_dir, f'{config.name}.pt')
    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    torch.save({
        'epoch': epoch,
        'loss': loss,
        'model_class': model.__class__,
        'model_state_dict': model.state_dict(),
        'optimiser_class': optimiser.__class__,
        'optimiser_state_dict': optimiser.state_dict(),
    }, path)


def train(model, optimiser, loss_func, train_data, val_data, config, metrics,
          epoch=None):
    """Train a model using the given optimiser and loss function."""
    print(colored('Training...', color='cyan', attrs=['bold', ]))

    # Move model to device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model = model.to(device)
    model.train()

    # Create dataloader
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    loader = DataLoader(
        train_data,
        collate_fn=variable_tensor_size_collator,
        batch_size=config.batch_size,
        shuffle=True,
        **loader_kwargs
    )

    # Determine starting epoch
    if epoch is not None:
        start = epoch + 1
    else:
        start = 0

    # Start epoch loop
    for epoch in range(start, config.epochs):
        epoch_loss = 0
        for batch_idx, (img_ids, imgs, captions, labels) in enumerate(loader):
            # Train batch
            imgs = [img.to(device) for img in imgs]
            labels = labels.to(device)
            optimiser.zero_grad()
            preds, output = model(imgs, captions, device)
            loss = loss_func(output, labels)
            loss.backward()
            optimiser.step()

            # Calculate and display metrics
            epoch_loss += loss.item()
            log_metrics_stdout(
                config,
                {'epoch': f'{epoch+1}/{config.epochs}',
                    'batch': f'{batch_idx+1}/{len(loader)}',
                    'loss': epoch_loss/(batch_idx + 1)},
                colors=('grey', 'grey', 'blue'),
                newline=False
            )
        evaluate(model, config, val_data, metrics)

        # Save model
        if not config.nosave:
            save(model, optimiser, loss_func, config, epoch)


def evaluate(model, config, data, metrics):
    """Evaluate a model on the specified metrics."""
    # Move model to device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
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

    # Collect predictions and evaluate metrics
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img_ids, imgs, captions, labels in tqdm(loader):
            imgs = [img.to(device) for img in imgs]
            preds, output = model(imgs, captions, device)
            preds = preds.detach().cpu().numpy()
            labels = labels.numpy()
            all_labels += list(labels)
            all_preds += list(preds)
        metrics.evaluate(all_preds, all_labels)

    # Log metrics
    log_metrics_stdout(config, metrics, colors='green')
    log_metrics_file(config, metrics)


def test(model, config, data):
    """Test a model and output results to the ouput directory."""
    print(colored('Testing...', color='cyan', attrs=['bold', ]))
    # Move model to device
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
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
            color='yellow'
        ))
        if confirmation != 'y':
            print('Aborted.')
            return

    # Run model predictions and save results
    with open(path, 'w') as f:
        f.write('ImageID,Labels\n')

    with torch.no_grad():
        for img_ids, imgs, captions in tqdm(loader):
            imgs = [img.to(device) for img in imgs]
            preds, output = model(imgs, captions, device)
            with open(path, 'a') as f:
                for im_id, pred in zip(img_ids, preds):
                    # Decode preds
                    lbl_idxs = [idx for idx, val in enumerate(pred) if val]
                    out = [str(i) for i in lbl_idxs]
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
        help='Load model state from the specified .pt file, relative to \
            --checkpoint-dir.'
    )
    general_group.add_argument(
        '--nosave',
        action='store_true',
        required=False,
        help='Don\'t save model state.'
    )
    general_group.add_argument(
        '--name',
        type=str,
        default=time.strftime('%Y%m%d-%H%M%S'),
        required=False,
        help='A unique identifier for the current run, used for saving \
            logs and model state checkpoints.'
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
    config_group.add_argument(
        '--checkpoint-dir',
        default=os.path.relpath(os.path.join(
            os.path.dirname(__file__),
            os.path.pardir,
            'checkpoints'
        )),
        type=str,
        required=False,
        help='The directory to save model checkpoints to.'
    )
    model_params_group.add_argument(
        '--batch-size',
        default=4,
        type=int,
        required=False,
        help='The batch size to use when training.'
    )
    model_params_group.add_argument(
        '--epochs',
        default=9,
        type=int,
        required=False,
        help='The number of epochs to train the model for.'
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
