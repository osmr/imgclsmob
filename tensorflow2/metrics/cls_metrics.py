"""
Evaluation Metrics for Image Classification.
"""

import tensorflow as tf
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
        self.base_acc = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : tensor
            The labels of the data with class indices as values, one per sample.
        preds : tensor
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        self.base_acc.update_state(labels, preds)

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
        return self.name, float(self.base_acc.result().numpy())


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
        self.base_acc = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="topk_acc")

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
        self.base_acc.update_state(labels, preds)

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
        return self.name, float(self.base_acc.result().numpy())


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
        return self.name, 1.0 - float(self.base_acc.result().numpy())


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
        return self.name, 1.0 - float(self.base_acc.result().numpy())
