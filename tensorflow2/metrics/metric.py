"""
    Several base metrics.
"""

__all__ = ['EvalMetric', 'CompositeEvalMetric', 'check_label_shapes']

from collections import OrderedDict


def check_label_shapes(labels, preds, shape=False):
    """
    Helper function for checking shape of label and prediction.

    Parameters
    ----------
    labels : list of tensor
        The labels of the data.
    preds : list of tensor
        Predicted values.
    shape : boolean
        If True, check the shape of labels and preds, otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = len(labels), len(preds)
    else:
        label_shape, pred_shape = labels.shape, preds.shape

    if label_shape != pred_shape:
        raise ValueError("Shape of labels {} does not match shape of predictions {}".format(label_shape, pred_shape))


class EvalMetric(object):
    """
    Base class for all evaluation metrics.

    Parameters
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 name,
                 output_names=None,
                 label_names=None,
                 **kwargs):
        super(EvalMetric, self).__init__()
        self.name = str(name)
        self.output_names = output_names
        self.label_names = label_names
        self._has_global_stats = kwargs.pop("has_global_stats", False)
        self._kwargs = kwargs
        self.reset()

    def __str__(self):
        return "EvalMetric: {}".format(dict(self.get_name_value()))

    def get_config(self):
        """
        Save configurations of metric. Can be recreated from configs with metric.create(**config).
        """
        config = self._kwargs.copy()
        config.update({
            "metric": self.__class__.__name__,
            "name": self.name,
            "output_names": self.output_names,
            "label_names": self.label_names})
        return config

    def update_dict(self, label, pred):
        """
        Update the internal evaluation with named label and pred.

        Parameters
        ----------
        labels : OrderedDict of str -> tensor
            name to array mapping for labels.
        preds : OrderedDict of str -> tensor
            name to array mapping of predicted outputs.
        """
        if self.output_names is not None:
            pred = [pred[name] for name in self.output_names]
        else:
            pred = list(pred.values())

        if self.label_names is not None:
            label = [label[name] for name in self.label_names]
        else:
            label = list(label.values())

        self.update(label, pred)

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : tensor
            The labels of the data.
        preds : tensor
            Predicted values.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.num_inst = 0
        self.sum_metric = 0.0
        self.global_num_inst = 0
        self.global_sum_metric = 0.0

    def reset_local(self):
        """
        Resets the local portion of the internal evaluation results to initial state.
        """
        self.num_inst = 0
        self.sum_metric = 0.0

    def get(self):
        """
        Gets the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self.num_inst == 0:
            return self.name, float("nan")
        else:
            return self.name, self.sum_metric / self.num_inst

    def get_global(self):
        """
        Gets the current global evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        if self._has_global_stats:
            if self.global_num_inst == 0:
                return self.name, float("nan")
            else:
                return self.name, self.global_sum_metric / self.global_num_inst
        else:
            return self.get()

    def get_name_value(self):
        """
        Returns zipped name and value pairs.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        name, value = self.get()
        if not isinstance(name, list):
            name = [name]
        if not isinstance(value, list):
            value = [value]
        return list(zip(name, value))

    def get_global_name_value(self):
        """
        Returns zipped name and value pairs for global results.

        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self._has_global_stats:
            name, value = self.get_global()
            if not isinstance(name, list):
                name = [name]
            if not isinstance(value, list):
                value = [value]
            return list(zip(name, value))
        else:
            return self.get_name_value()


class CompositeEvalMetric(EvalMetric):
    """
    Manages multiple evaluation metrics.

    Parameters
    ----------
    name : str, default 'composite'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """

    def __init__(self,
                 name="composite",
                 output_names=None,
                 label_names=None):
        super(CompositeEvalMetric, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.metrics = []

    def add(self, metric):
        """
        Adds a child metric.

        Parameters
        ----------
        metric
            A metric instance.
        """
        self.metrics.append(metric)

    def update_dict(self, labels, preds):
        if self.label_names is not None:
            labels = OrderedDict([i for i in labels.items()
                                  if i[0] in self.label_names])
        if self.output_names is not None:
            preds = OrderedDict([i for i in preds.items()
                                 if i[0] in self.output_names])

        for metric in self.metrics:
            metric.update_dict(labels, preds)

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : tensor
            The labels of the data.

        preds : tensor
            Predicted values.
        """
        for metric in self.metrics:
            metric.update(labels, preds)

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        try:
            for metric in self.metrics:
                metric.reset()
        except AttributeError:
            pass

    def reset_local(self):
        """
        Resets the local portion of the internal evaluation results to initial state.
        """
        try:
            for metric in self.metrics:
                metric.reset_local()
        except AttributeError:
            pass

    def get(self):
        """
        Returns the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get()
            name = [name]
            value = [value]
            names.extend(name)
            values.extend(value)
        return names, values

    def get_global(self):
        """
        Returns the current evaluation result.

        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        names = []
        values = []
        for metric in self.metrics:
            name, value = metric.get_global()
            name = [name]
            value = [value]
            names.extend(name)
            values.extend(value)
        return names, values

    def get_config(self):
        config = super(CompositeEvalMetric, self).get_config()
        config.update({"metrics": [i.get_config() for i in self.metrics]})
        return config
