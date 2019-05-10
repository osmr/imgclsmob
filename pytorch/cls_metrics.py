"""
Evaluation Metrics for Image Classification.
"""

import numpy as np
import torch
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
        labels : torch.Tensor
            The labels of the data with class indices as values, one per sample.
        preds : torch.Tensor
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        assert (len(labels) == len(preds))
        with torch.no_grad():
            if preds.shape != labels.shape:
                pred_label = torch.argmax(preds, dim=self.axis)
            else:
                pred_label = preds
            pred_label = pred_label.cpu().numpy().astype(np.int32)
            label = labels.cpu().numpy().astype(np.int32)

            label = label.flat
            pred_label = pred_label.flat

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)


class TopKAccuracy(EvalMetric):
    """
    Computes top k predictions accuracy.

    Parameters
    ----------
    top_k : int, default 1
        Whether targets are in top k predictions.
    name : str, default 'top_k_accuracy'
        Name of this metric instance for display.
    torch_like : bool, default True
        Whether to use pytorch-like algorithm.
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
                 torch_like=True,
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
        self.torch_like = torch_like

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : torch.Tensor
            The labels of the data.
        preds : torch.Tensor
            Predicted values.
        """
        assert (len(labels) == len(preds))
        with torch.no_grad():
            if self.torch_like:
                _, pred = preds.topk(k=self.top_k, dim=1, largest=True, sorted=True)
                pred = pred.t()
                correct = pred.eq(labels.view(1, -1).expand_as(pred))
                num_correct = correct.view(-1).float().sum(dim=0, keepdim=True).item()
                num_samples = labels.size(0)
                assert (num_correct <= num_samples)
                self.sum_metric += num_correct
                self.global_sum_metric += num_correct
                self.num_inst += num_samples
                self.global_num_inst += num_samples
            else:
                assert(len(preds.shape) <= 2), "Predictions should be no more than 2 dims"
                pred_label = preds.cpu().numpy().astype(np.int32)
                pred_label = np.argpartition(pred_label, -self.top_k)
                label = labels.cpu().numpy().astype(np.int32)
                assert (len(label) == len(pred_label))
                num_samples = pred_label.shape[0]
                num_dims = len(pred_label.shape)
                if num_dims == 1:
                    num_correct = (pred_label.flat == label.flat).sum()
                    self.sum_metric += num_correct
                    self.global_sum_metric += num_correct
                elif num_dims == 2:
                    num_classes = pred_label.shape[1]
                    top_k = min(num_classes, self.top_k)
                    for j in range(top_k):
                        num_correct = (pred_label[:, num_classes - 1 - j].flat == label.flat).sum()
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
    torch_like : bool, default True
        Whether to use pytorch-like algorithm.
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
                 torch_like=True,
                 output_names=None,
                 label_names=None):
        name_ = name
        super(TopKError, self).__init__(
            top_k=top_k,
            name=name,
            torch_like=torch_like,
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
