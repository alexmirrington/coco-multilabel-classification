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
        return results


# region Metric convenience functions
def micro_f1(y_true, y_pred):
    """Compute the micro f1 of two label sets."""
    return f1_score(y_true, y_pred, average='micro')

# endregion
