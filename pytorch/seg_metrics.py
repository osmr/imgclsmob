# import numpy as np
from .metric import EvalMetric, check_label_shapes
# from .seg_metrics_np import seg_pixel_accuracy_np, seg_mean_iou_imasks_np


class PixelAccuracyMetric(EvalMetric):
    """
    Computes the pixel-wise accuracy.

    Parameters
    ----------
    axis : int, default 1
        The axis that represents classes.
    name : str, default 'pix_acc'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    vague_idx : int, default -1
        Index of masked pixels.
    use_vague : bool, default False
        Whether to use pixel masking.
    macro_average : bool, default True
        Whether to use micro or macro averaging.
    """
    def __init__(self,
                 axis=1,
                 name="pix_acc",
                 output_names=None,
                 label_names=None,
                 vague_idx=-1,
                 use_vague=False,
                 macro_average=True):
        self.macro_average = macro_average
        super(PixelAccuracyMetric, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        self.axis = axis
        self.vague_idx = vague_idx
        self.use_vague = use_vague

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
        check_label_shapes(labels, preds)
        for label, pred in zip(labels, preds):
            pred = pred.max(dim=self.axis)[1]
            label = label.max(dim=self.axis)[1]
            self.sum_metric += pred.eq(label).sum()
            self.num_inst += pred.numel()

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        if self.macro_average:
            # super(PixelAccuracyMetric, self).reset()
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = 0
            self.sum_metric = 0
            self.global_num_inst = 0
            self.global_sum_metric = 0

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
        if self.macro_average:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, float(self.sum_metric) / self.num_inst)


class MeanIoUMetric(EvalMetric):
    """
    Computes the mean intersection over union.

    Parameters
    ----------
    axis : int, default=1
        The axis that represents classes
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
                 axis=1,
                 name='mean_iou',
                 output_names=None,
                 label_names=None):
        super(MeanIoUMetric, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        self.axis = axis

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data.

        preds : list of `NDArray`
            Predicted values.
        """
        check_label_shapes(labels, preds)

        for label, pred in zip(labels, preds):
            # label_hmask = label.cpu().numpy().astype(np.uint8)
            # pred_imask = pred.max(dim=self.axis)[1].cpu().numpy().astype(np.uint8)

            batch_size = label.shape[0]
            for k in range(batch_size):
                pass
                # acc = seg_mean_iou_imasks_np(label_hmask=label_hmask[k, :, :, :], pred_imask=pred_imask[k, :, :])
                # self.sum_metric += acc
                # self.num_inst += 1
