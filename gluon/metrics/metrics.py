"""
    Evaluation metrics for common tasks.
"""

import mxnet as mx
if mx.__version__ < "2.0.0":
    from mxnet.metric import EvalMetric
else:
    from mxnet.gluon.metric import EvalMetric

__all__ = ['LossValue']


class LossValue(EvalMetric):
    """
    Computes simple loss value fake metric.

    Parameters:
    ----------
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 name="loss",
                 output_names=None,
                 label_names=None):
        super(LossValue, self).__init__(
            name,
            output_names=output_names,
            label_names=label_names)

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters:
        ----------
        labels : None
            Unused argument.
        preds : list of `NDArray`
            Loss values.
        """
        loss = sum([ll.mean().asscalar() for ll in preds]) / len(preds)
        self.sum_metric += loss
        self.global_sum_metric += loss
        self.num_inst += 1
        self.global_num_inst += 1
