"""
Evaluation Metrics for Semantic Segmentation.
"""

__all__ = ['PixelAccuracyMetric', 'MeanIoUMetric']

import numpy as np
import mxnet as mx
from .seg_metrics_np import seg_pixel_accuracy_np, seg_mean_iou_imasks_np
from .seg_metrics_nd import seg_pixel_accuracy_nd


class PixelAccuracyMetric(mx.metric.EvalMetric):
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
    on_cpu : bool, default True
        Calculate on CPU.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    vague_idx : int, default -1
        Index of masked pixels.
    use_vague : bool, default False
        Whether to use pixel masking.
    macro_average : bool, default True
        Whether to use micro or macro averaging.
    aux : bool, default False
        Whether to support auxiliary predictions.
    """
    def __init__(self,
                 axis=1,
                 name="pix_acc",
                 output_names=None,
                 label_names=None,
                 on_cpu=True,
                 sparse_label=True,
                 vague_idx=-1,
                 use_vague=False,
                 macro_average=True,
                 aux=False):
        if name == "pix_acc":
            name = "{}-pix_acc".format("macro" if macro_average else "micro")
        self.macro_average = macro_average
        super(PixelAccuracyMetric, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        self.axis = axis
        self.on_cpu = on_cpu
        self.sparse_label = sparse_label
        self.vague_idx = vague_idx
        self.use_vague = use_vague
        self.aux = aux

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
        if self.aux:
            preds = [p[0] for p in preds]
        assert (len(labels) == len(preds))
        if self.on_cpu:
            for label, pred in zip(labels, preds):
                if self.sparse_label:
                    label_imask = label.asnumpy().astype(np.int32)
                else:
                    label_imask = mx.nd.argmax(label, axis=self.axis).asnumpy().astype(np.int32)
                pred_imask = mx.nd.argmax(pred, axis=self.axis).asnumpy().astype(np.int32)
                acc = seg_pixel_accuracy_np(
                    label_imask=label_imask,
                    pred_imask=pred_imask,
                    vague_idx=self.vague_idx,
                    use_vague=self.use_vague,
                    macro_average=self.macro_average)
                if self.macro_average:
                    self.sum_metric += acc
                    self.num_inst += 1
                else:
                    self.sum_metric += acc[0]
                    self.num_inst += acc[1]
        else:
            for label, pred in zip(labels, preds):
                if self.sparse_label:
                    label_imask = mx.nd.cast(label, dtype=np.int32)
                else:
                    label_imask = mx.nd.cast(mx.nd.argmax(label, axis=self.axis), dtype=np.int32)
                pred_imask = mx.nd.cast(mx.nd.argmax(pred, axis=self.axis), dtype=np.int32)
                acc = seg_pixel_accuracy_nd(
                    label_imask=label_imask,
                    pred_imask=pred_imask,
                    vague_idx=self.vague_idx,
                    use_vague=self.use_vague,
                    macro_average=self.macro_average)
                if self.macro_average:
                    self.sum_metric += acc
                    self.num_inst += 1
                else:
                    self.sum_metric += acc[0]
                    self.num_inst += acc[1]

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        if self.macro_average:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = 0
            self.sum_metric = 0

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
                return self.name, float("nan")
            else:
                return self.name, self.sum_metric / self.num_inst
        else:
            if self.num_inst == 0:
                return self.name, float("nan")
            else:
                return self.name, float(self.sum_metric) / self.num_inst


class MeanIoUMetric(mx.metric.EvalMetric):
    """
    Computes the mean intersection over union.

    Parameters
    ----------
    axis : int, default 1
        The axis that represents classes
    name : str, default 'mean_iou'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    on_cpu : bool, default True
        Calculate on CPU.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    num_classes : int
        Number of classes
    vague_idx : int, default -1
        Index of masked pixels.
    use_vague : bool, default False
        Whether to use pixel masking.
    bg_idx : int, default -1
        Index of background class.
    ignore_bg : bool, default False
        Whether to ignore background class.
    macro_average : bool, default True
        Whether to use micro or macro averaging.
    """
    def __init__(self,
                 axis=1,
                 name="mean_iou",
                 output_names=None,
                 label_names=None,
                 on_cpu=True,
                 sparse_label=True,
                 num_classes=None,
                 vague_idx=-1,
                 use_vague=False,
                 bg_idx=-1,
                 ignore_bg=False,
                 macro_average=True):
        if name == "pix_acc":
            name = "{}-pix_acc".format("macro" if macro_average else "micro")
        self.macro_average = macro_average
        self.num_classes = num_classes
        self.ignore_bg = ignore_bg
        super(MeanIoUMetric, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        assert ((not ignore_bg) or (bg_idx in (0, num_classes - 1)))
        self.axis = axis
        self.on_cpu = on_cpu
        self.sparse_label = sparse_label
        self.vague_idx = vague_idx
        self.use_vague = use_vague
        self.bg_idx = bg_idx

        assert (on_cpu and sparse_label)

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
        assert (len(labels) == len(preds))
        if self.on_cpu:
            for label, pred in zip(labels, preds):
                if self.sparse_label:
                    label_imask = label.asnumpy().astype(np.int32)
                # else:
                #     label_hmask = label.asnumpy().astype(np.int32)
                pred_imask = mx.nd.argmax(pred, axis=self.axis).asnumpy().astype(np.int32)
                batch_size = label.shape[0]
                for k in range(batch_size):
                    if self.sparse_label:
                        acc = seg_mean_iou_imasks_np(
                            label_imask=label_imask[k, :, :],
                            pred_imask=pred_imask[k, :, :],
                            num_classes=self.num_classes,
                            vague_idx=self.vague_idx,
                            use_vague=self.use_vague,
                            bg_idx=self.bg_idx,
                            ignore_bg=self.ignore_bg,
                            macro_average=self.macro_average)
                    # else:
                    #     acc = seg_mean_iou_np(
                    #         label_hmask=label_hmask[k, :, :, :],
                    #         pred_imask=pred_imask[k, :, :])
                    if self.macro_average:
                        self.sum_metric += acc
                        self.num_inst += 1
                    else:
                        self.area_inter += acc[0]
                        self.area_union += acc[1]
        # else:
        #     for label, pred in zip(labels, preds):
        #         if self.sparse_label:
        #             label_imask = label
        #             n = self.num_classes
        #             label_hmask = mx.nd.one_hot(label_imask, depth=n).transpose((0, 3, 1, 2))
        #         else:
        #             label_hmask = label
        #             n = label_hmask.shape[1]
        #         pred_imask = mx.nd.argmax(pred, axis=self.axis)
        #         pred_hmask = mx.nd.one_hot(pred_imask, depth=n).transpose((0, 3, 1, 2))
        #         acc = seg_mean_iou2_nd(
        #             label_hmask=label_hmask,
        #             pred_hmask=pred_hmask)
        #         self.sum_metric += acc
        #         self.num_inst += 1

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        if self.macro_average:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            class_count = self.num_classes - 1 if self.ignore_bg else self.num_classes
            self.area_inter = np.zeros((class_count,), np.uint64)
            self.area_union = np.zeros((class_count,), np.uint64)

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
                return self.name, float("nan")
            else:
                return self.name, self.sum_metric / self.num_inst
        else:
            class_count = (self.area_union > 0).sum()
            if class_count == 0:
                return self.name, float("nan")
            eps = np.finfo(np.float32).eps
            area_union_eps = self.area_union + eps
            mean_iou = (self.area_inter / area_union_eps).sum() / class_count
            return self.name, mean_iou
