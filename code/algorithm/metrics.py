"""Main module for metrics calculation."""

from sklearn.metrics import f1_score


class MetricCollection:
    """Wrapper class for storing a collection of metrics \
    to evaluate."""

    def __init__(self, metrics={}):
        """Create a new collection of metrics from a dictionary \
        of `(metric_name, metric_func)` pairs where each metric \
        function takes a list of `y_true` and `y_pred` labels as \
        input."""
        self.metrics = metrics
        self.__cache = None

    @property
    def cache(self):
        """Get the cached values of the the last call to `evaluate`."""
        return self.__cache

    def add_metrics(self, **metrics):
        """Update the current metric set with a collection of metrics \
        from a dictionary of `(metric_name, metric_func)` pairs where \
        each metric function takes a list of `y_true` and `y_pred` \
        labels as input."""
        for key, func in metrics:
            self.metrics[key] = func

    def evaluate(self, y_true, y_pred):
        """Evaluate all metrics in the collection for the given labels \
        and return the results."""
        results = {}
        for key, func in self.metrics.items():
            results[key] = func(y_true, y_pred)
        self.__cache = results
        return results


# region Metric convenience functions
def micro_f1(y_true, y_pred):
    """Compute the micro f1 of two label sets."""
    return f1_score(y_true, y_pred, average='micro')


def macro_f1(y_true, y_pred):
    """Compute the macro f1 of two label sets."""
    return f1_score(y_true, y_pred, average='macro')


def weighted_f1(y_true, y_pred):
    """Compute the weighted f1 of two label sets."""
    return f1_score(y_true, y_pred, average='weighted')
# endregion
