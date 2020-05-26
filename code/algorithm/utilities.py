"""Main module for utility functions e.g. encoding, logging utilities."""
import os

from metrics import MetricCollection
from sklearn.preprocessing import MultiLabelBinarizer
from termcolor import colored


def log_metrics_file(config, metrics):
    """Log all cached metrics in the `metrics` collection to a log file."""
    filename = os.path.join(config.log_dir, f'{config.name}.txt')
    output = ''
    metrics_dict = metrics
    if isinstance(metrics, MetricCollection):
        metrics_dict = metrics.cache
    for idx, (key, value) in enumerate(metrics_dict.items()):
        output += f'{key}: {value:4f}'
        if idx != len(metrics_dict) - 1:
            output += ', '
        else:
            output += '\n'
    if not os.path.exists(config.log_dir):
        os.mkdir(config.log_dir)
    with open(filename, 'a') as f:
        f.write(output)


def log_metrics_stdout(config, metrics, colors=None, newline=True):
    """Log all cached metrics in the `metrics` collection to stdout."""
    output = ''
    metrics_dict = metrics
    if isinstance(metrics, MetricCollection):
        metrics_dict = metrics.cache

    if isinstance(colors, str):
        colors = [colors for idx in range(len(metrics_dict))]

    for idx, (key, value) in enumerate(metrics_dict.items()):
        color = None
        if colors is not None and idx < len(colors):
            color = colors[idx]

        if type(value) == float:
            value = f'{value:4f}'
        valstr = colored(value, color=color, attrs=['bold', ]) \
            if color else value
        output += f'{key}: {valstr} '
    output = output.rstrip()
    if not newline:
        output = f'{output}\r'
    print(output, end='\n' if newline else '')


def binarise_labels(data, classes=None):
    """Convert a set of labels to binary multilabel format.

    `classes` should be set to the classes of the training set
    when evaluating a model in case not all classes are present
    in the predictions.

    Examples
    --------
    Example 1

    `data = [(1, 2), (3,)]`, `classes = None`

    `output = array([[0, 1, 1], [1, 0, 0]])`

    Example 2

    `data = [(1, 2), (3,)]`, `classes = [0, 1, 2, 3]`

    `output = array([[0, 0, 1, 1], [0, 1, 0, 0]])`
    """
    mlb = MultiLabelBinarizer(classes=classes)
    return mlb.fit_transform(data), mlb.classes_
