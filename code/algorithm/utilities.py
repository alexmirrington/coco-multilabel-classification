"""Main module for utility functions e.g. label encodings."""
from sklearn.preprocessing import MultiLabelBinarizer


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
