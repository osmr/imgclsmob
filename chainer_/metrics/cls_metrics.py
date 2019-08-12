"""
Evaluation Metrics for Image Classification.
"""

import numpy as np
from chainer.backends import cuda
from .metric import EvalMetric

__all__ = ['Top1Error', 'TopKError']


class Accuracy(EvalMetric):
    """
    Computes accuracy classification score.

    Parameters
    ----------
    axis : int, default 1
        The axis that represents classes
    name : str, default 'accuracy'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="accuracy",
                 output_names=None,
                 label_names=None):
        super(Accuracy, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = axis

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : xp.array
            The labels of the data with class indices as values, one per sample.
        preds : xp.array
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        if len(preds.shape) == 1:
            num_samples = 1
            num_correct = (preds.argmax() == labels)
        else:
            assert (len(labels) == len(preds))
            num_samples = preds.shape[0]
            label = labels.astype(np.int32).flat
            pred_label = preds.argmax(axis=self.axis).astype(np.int32).flat
            num_correct = (pred_label == label).sum()

        self.sum_metric += num_correct
        self.global_sum_metric += num_correct
        self.num_inst += num_samples
        self.global_num_inst += num_samples


class TopKAccuracy(EvalMetric):
    """
    Computes top k predictions accuracy.

    Parameters
    ----------
    top_k : int, default 1
        Whether targets are in top k predictions.
    name : str, default 'top_k_accuracy'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 top_k=1,
                 name="top_k_accuracy",
                 output_names=None,
                 label_names=None):
        super(TopKAccuracy, self).__init__(
            name,
            top_k=top_k,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.top_k = top_k
        assert (self.top_k > 1), "Please use Accuracy if top_k is no more than 1"
        self.name += "_{:d}".format(self.top_k)

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : xp.array
            The labels of the data.
        preds : xp.array
            Predicted values.
        """
        xp = cuda.get_array_module(preds)
        if len(preds.shape) == 1:
            num_samples = 1
            argsorted_pred = xp.argsort(preds)[-self.top_k:]
            num_correct = int(xp.any(argsorted_pred.T == labels, axis=0))
        else:
            assert (len(labels) == len(preds))
            num_samples = preds.shape[0]
            argsorted_pred = xp.argsort(preds)[:, -self.top_k:]
            num_correct = xp.any(argsorted_pred.T == labels, axis=0).sum()
        assert (num_correct <= num_samples)
        self.sum_metric += num_correct
        self.global_sum_metric += num_correct
        self.num_inst += num_samples
        self.global_num_inst += num_samples


class Top1Error(Accuracy):
    """
    Computes top-1 error (inverted accuracy classification score).

    Parameters
    ----------
    axis : int, default 1
        The axis that represents classes.
    name : str, default 'top_1_error'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 axis=1,
                 name="top_1_error",
                 output_names=None,
                 label_names=None):
        super(Top1Error, self).__init__(
            axis=axis,
            name=name,
            output_names=output_names,
            label_names=label_names)

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
            return self.name, 1.0 - self.sum_metric / self.num_inst


class TopKError(TopKAccuracy):
    """
    Computes top-k error (inverted top k predictions accuracy).

    Parameters
    ----------
    top_k : int
        Whether targets are out of top k predictions, default 1
    name : str, default 'top_k_error'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 top_k=1,
                 name="top_k_error",
                 output_names=None,
                 label_names=None):
        name_ = name
        super(TopKError, self).__init__(
            top_k=top_k,
            name=name,
            output_names=output_names,
            label_names=label_names)
        self.name = name_.replace("_k_", "_{}_".format(top_k))

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
            return self.name, 1.0 - self.sum_metric / self.num_inst
