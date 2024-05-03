"""
    Evaluation Metrics for Image Classification.
"""

import mxnet as mx

__all__ = ['Top1Error', 'TopKError']


class Top1Error(mx.metric.Accuracy):
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


class TopKError(mx.metric.TopKAccuracy):
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
