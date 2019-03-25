"""
Evaluation Metrics for Semantic Segmentation
"""

import threading
import numpy as np
import mxnet as mx
from mxnet.metric import EvalMetric

__all__ = ['PixIoUSegMetric', 'PixelAccuracyMetric', 'MeanIoUMetric']


def segm_pixel_accuracy(label_imask,
                        pred_imask,
                        mask_idx=-1,
                        use_mask=False):
    """
    The segmentation pixel accuracy.

    Parameters
    ----------
    label_imask : np.array
        Ground truth index mask (maybe batch of).
    pred_imask : np.array
        Predicted index mask (maybe batch of).
    mask_idx : int, default -1
        Index of masked pixels.
    use_mask : bool, default False
        Whether to use pixel masking.

    Returns
    -------
    float
        PA metric value.
    """
    assert (label_imask.shape == pred_imask.shape)
    if use_mask:
        sum_u_ij = np.sum(label_imask.flat != mask_idx)
        if sum_u_ij == 0:
            return 0.0
        sum_u_ii = np.sum(np.logical_and(pred_imask.flat == label_imask.flat, label_imask.flat != mask_idx))
    else:
        sum_u_ii = np.sum(pred_imask.flat == label_imask.flat)
        sum_u_ij = pred_imask.size
    return float(sum_u_ii) / sum_u_ij


def segm_pixel_accuracy_nd(label_imask,
                           pred_imask,
                           mask_idx=-1,
                           use_mask=False):
    """
    The segmentation pixel accuracy (for MXNet nd-arrays).

    Parameters
    ----------
    label_imask : mx.nd.array
        Ground truth index mask (maybe batch of).
    pred_imask : mx.nd.array
        Predicted index mask (maybe batch of).
    mask_idx : int, default -1
        Index of masked pixels.
    use_mask : bool, default False
        Whether to use pixel masking.

    Returns
    -------
    float
        PA metric value.
    """
    assert (label_imask.shape == pred_imask.shape)
    if use_mask:
        mask = (label_imask != mask_idx)
        sum_u_ij = mask.sum().asscalar()
        if sum_u_ij == 0:
            return 0.0
        sum_u_ii = ((label_imask != pred_imask) * mask).sum().asscalar()
    else:
        sum_u_ii = mx.nd.equal(label_imask, pred_imask).sum().asscalar()
        sum_u_ij = pred_imask.size
    return float(sum_u_ii) / sum_u_ij


class PixelAccuracyMetric(mx.metric.EvalMetric):
    """
    Computes the pixel-wise accuracy.

    Parameters
    ----------
    axis : int, default 1
        The axis that represents classes.
    name : str
        Name of this metric instance for display.
    output_names : list of str, or None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    on_cpu : bool, default True
        Calculate on CPU.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    num_classes : int, default None
        Number of classes.
    mask_idx : int, default -1
        Index of masked pixels.
    use_mask : bool, default False
        Whether to use pixel masking.
    """
    def __init__(self,
                 axis=1,
                 name="pix_acc",
                 output_names=None,
                 label_names=None,
                 on_cpu=True,
                 sparse_label=True,
                 num_classes=None,
                 mask_idx=-1,
                 use_mask=False):
        super(PixelAccuracyMetric, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        self.axis = axis
        self.on_cpu = on_cpu
        self.sparse_label = sparse_label
        self.num_classes = num_classes
        self.mask_idx = mask_idx
        self.use_mask = use_mask

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
                else:
                    label_imask = mx.nd.argmax(label, axis=self.axis).asnumpy().astype(np.int32)
                pred_imask = mx.nd.argmax(pred, axis=self.axis).asnumpy().astype(np.int32)
                acc = segm_pixel_accuracy(
                    label_imask=label_imask,
                    pred_imask=pred_imask,
                    mask_idx=self.mask_idx,
                    use_mask=self.use_mask)
                self.sum_metric += acc
                self.num_inst += 1
        else:
            for label, pred in zip(labels, preds):
                if self.sparse_label:
                    label_imask = mx.nd.cast(label, dtype=np.int32)
                else:
                    label_imask = mx.nd.cast(mx.nd.argmax(label, axis=self.axis), dtype=np.int32)
                pred_imask = mx.nd.cast(mx.nd.argmax(pred, axis=self.axis), dtype=np.int32)
                acc = segm_pixel_accuracy_nd(
                    label_imask=label_imask,
                    pred_imask=pred_imask,
                    mask_idx=self.mask_idx,
                    use_mask=self.use_mask)
                self.sum_metric += acc
                self.num_inst += 1


def segm_mean_iou_imasks(label_imask,
                         pred_imask,
                         num_classes,
                         ignore_bg=False):
    """
    The segmentation mean intersection over union.

    Parameters
    ----------
    label_imask : nd.array
        Ground truth index mask (batch of)
    pred_imask : nd.array
        Predicted index mask (batch of)
    num_classes : int
        Number of classes
    ignore_bg : bool
        Ignoring class 0, if true

    Returns
    -------
    acc : float
        MIoU metric value
    """
    assert (len(label_imask.shape) == 2)
    assert (len(pred_imask.shape) == 2)
    assert (pred_imask.shape == label_imask.shape)

    eps = np.finfo(np.float32).eps

    mini = 1
    maxi = num_classes
    nbins = num_classes
    if ignore_bg:
        maxi -= 1
        nbins -= 1
        pred_imask = pred_imask * (label_imask > 0).astype(pred_imask.dtype)
    else:
        label_imask += 1
        pred_imask += 1
        if (label_imask < 0).any():
            pred_imask = pred_imask * (label_imask >= 0).astype(pred_imask.dtype)

    intersection = pred_imask * (pred_imask == label_imask)

    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(pred_imask, bins=nbins, range=(mini, maxi))
    area_label, _ = np.histogram(label_imask, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_label - area_inter + eps

    mean_iou = (area_inter / area_union).mean()

    return mean_iou


def segm_mean_iou(label_hmask,
                  pred_imask):
    """
    The segmentation mean intersection over union.

    Parameters
    ----------
    label_hmask : np.array
        Ground truth one-hot mask
    pred_imask : np.array
        Predicted index mask

    Returns
    -------
    acc : float
        MIoU metric value
    """
    assert (len(label_hmask.shape) == 3)
    assert (len(pred_imask.shape) == 2)
    assert (pred_imask.shape == label_hmask.shape[1:])
    n = label_hmask.shape[0]
    i_sum = 0
    acc_iou = 0.0
    for i in range(n):
        class_i_pred_mask = (pred_imask == i)
        class_i_label_mask = label_hmask[i, :, :]

        u_i = np.sum(class_i_label_mask)
        u_ji_sj = np.sum(class_i_pred_mask)
        if (u_i + u_ji_sj) == 0:
            continue

        u_ii = np.sum(np.logical_and(class_i_pred_mask, class_i_label_mask))

        acc_iou += float(u_ii) / (u_i + u_ji_sj - u_ii)
        i_sum += 1

    if i_sum > 0:
        mean_iou = acc_iou / i_sum
    else:
        mean_iou = 1.0

    return mean_iou


def segm_mean_iou2(label_hmask,
                   pred_hmask):
    """
    The segmentation mean intersection over union.

    Parameters
    ----------
    label_hmask : nd.array
        Ground truth one-hot mask (batch of)
    pred_hmask : nd.array
        Predicted one-hot mask (batch of)

    Returns
    -------
    acc : float
        MIoU metric value
    """
    assert (len(label_hmask.shape) == 4)
    assert (len(pred_hmask.shape) == 4)
    assert (pred_hmask.shape == label_hmask.shape)

    eps = np.finfo(np.float32).eps
    batch_axis = 0  # The axis that represents mini-batch
    class_axis = 1  # The axis that represents classes

    inter_hmask = label_hmask * pred_hmask
    u_i = label_hmask.sum(axis=[batch_axis, class_axis], exclude=True)
    u_ji_sj = pred_hmask.sum(axis=[batch_axis, class_axis], exclude=True)
    u_ii = inter_hmask.sum(axis=[batch_axis, class_axis], exclude=True)
    class_count = (u_i + u_ji_sj > 0.0).sum(axis=class_axis) + eps
    class_acc = u_ii / (u_i + u_ji_sj - u_ii + eps)
    acc_iou = class_acc.sum(axis=class_axis) + eps
    mean_iou = (acc_iou / class_count).mean().asscalar()

    return mean_iou


class MeanIoUMetric(mx.metric.EvalMetric):
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
    on_cpu : bool
        Calculate on CPU.
    sparse_label : bool, default False
        Whether label is an integer array instead of probability distribution.
    num_classes : int
        Number of classes
    ignore_bg : bool
        Ignoring class 0, if true
    """
    def __init__(self,
                 axis=1,
                 name='mean_iou',
                 output_names=None,
                 label_names=None,
                 on_cpu=True,
                 sparse_label=False,
                 num_classes=None,
                 ignore_bg=False):
        super(MeanIoUMetric, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names)
        self.axis = axis
        self.on_cpu = on_cpu
        self.sparse_label = sparse_label
        self.num_classes = num_classes
        self.ignore_bg = ignore_bg

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
        # time_begin = time.time()
        assert (len(labels) == len(preds))
        if self.on_cpu:
            for label, pred in zip(labels, preds):
                if self.sparse_label:
                    label_imask = label.asnumpy().astype(np.int32)
                else:
                    label_hmask = label.asnumpy().astype(np.int32)
                pred_imask = mx.nd.argmax(pred, axis=self.axis).asnumpy().astype(np.int32)
                batch_size = label.shape[0]
                for k in range(batch_size):
                    if self.sparse_label:
                        acc = segm_mean_iou_imasks(
                            label_imask=label_imask[k, :, :],
                            pred_imask=pred_imask[k, :, :],
                            num_classes=self.num_classes,
                            ignore_bg=self.ignore_bg)
                    else:
                        acc = segm_mean_iou(
                            label_hmask=label_hmask[k, :, :, :],
                            pred_imask=pred_imask[k, :, :])
                    self.sum_metric += acc
                    self.num_inst += 1
        else:
            for label, pred in zip(labels, preds):
                if self.sparse_label:
                    label_imask = label
                    n = self.num_classes
                    label_hmask = mx.nd.one_hot(label_imask, depth=n).transpose((0, 3, 1, 2))
                else:
                    label_hmask = label
                    n = label_hmask.shape[1]
                pred_imask = mx.nd.argmax(pred, axis=self.axis)
                pred_hmask = mx.nd.one_hot(pred_imask, depth=n).transpose((0, 3, 1, 2))
                acc = segm_mean_iou2(
                    label_hmask=label_hmask,
                    pred_hmask=pred_hmask)
                self.sum_metric += acc
                self.num_inst += 1


class PixelAccuracy(EvalMetric):
    """
    Computes pixel accuracy segmentation score.
    """
    def __init__(self,
                 num_classes,
                 axis=1,
                 name="pixel_accuracy",
                 output_names=None,
                 label_names=None):
        super(PixelAccuracy, self).__init__(
            name=name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.num_classes = num_classes

    def update(self,
               labels,
               preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : list of `NDArray`
            The labels of the data with class indices as values, one per sample.
        preds : list of `NDArray`
            Prediction values for samples. Each prediction value can either be the class index,
            or a vector of likelihoods for all classes.
        """
        # labels, preds = check_label_shapes(labels, preds, True)

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                pred_label = mx.nd.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.asnumpy().astype(np.int32)
            label = label.asnumpy().astype(np.int32)
            # flatten before checking shapes to avoid shape miss match
            label = label.flat
            pred_label = pred_label.flat

            # check_label_shapes(label, pred_label)

            num_correct = (pred_label == label).sum()
            self.sum_metric += num_correct
            self.global_sum_metric += num_correct
            self.num_inst += len(pred_label)
            self.global_num_inst += len(pred_label)


class PixIoUSegMetric(EvalMetric):
    """
    Computes pixAcc and mIoU metric scores
    """
    def __init__(self, classes):
        super(PixIoUSegMetric, self).__init__('pixAcc & mIoU')
        self.classes = classes
        self.lock = threading.Lock()
        self.reset()

    def update(self,
               labels,
               preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : 'NDArray' or list of `NDArray`
            The labels of the data.
        preds : 'NDArray' or list of `NDArray`
            Predicted values.
        """
        def evaluate_worker(self,
                            label,
                            pred):
            correct, labeled = batch_pixel_accuracy(pred, label)
            inter, union = batch_intersection_union(pred, label, self.nclass)
            with self.lock:
                self.total_correct += correct
                self.total_label += labeled
                self.total_inter += inter
                self.total_union += union

        if isinstance(preds, mx.nd.NDArray):
            evaluate_worker(self, labels, preds)
        elif isinstance(preds, (list, tuple)):
            threads = [threading.Thread(target=evaluate_worker, args=(self, label, pred)) for (label, pred) in
                       zip(labels, preds)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

    def get(self):
        """
        Gets the current evaluation result.

        Returns
        -------
        metrics : tuple of float
            pixAcc and mIoU
        """
        pix_acc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        iou = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        miou = iou.mean()
        return pix_acc, miou

    def reset(self):
        """
        Resets the internal evaluation result to initial state.
        """
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0


def batch_pixel_accuracy(output,
                         target):
    """
    PixAcc
    """
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    predict = np.argmax(output.asnumpy().astype('int64'), 1) + 1

    target = target.asnumpy().astype('int64') + 1

    pixel_labeled = np.sum(target > 0)
    pixel_correct = np.sum((predict == target) * (target > 0))

    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output,
                             target,
                             num_classes):
    """
    mIoU
    """
    # inputs are NDarray, output 4D, target 3D
    # the category -1 is ignored class, typically for background / boundary
    mini = 1
    maxi = num_classes
    nbins = num_classes
    predict = np.argmax(output.asnumpy().astype('int64'), 1) + 1
    target = target.asnumpy().astype('int64') + 1

    predict = predict * (target > 0).astype(predict.dtype)
    intersection = predict * (predict == target)
    # areas of intersection and union
    area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
    area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
    area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
    area_union = area_pred + area_lab - area_inter
    assert (area_inter <= area_union).all(), "Intersection area should be smaller than Union area"
    return area_inter, area_union


def calc_pixel_accuracy(img_prediction,
                        img_label):
    """
    Calculate pixel-wise accuracy for the prediction and label of a single image
    """

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    pixel_labeled = np.sum(img_label > 0)
    pixel_correct = np.sum((img_prediction == img_label) * (img_label > 0))
    pixel_accuracy = 1.0 * pixel_correct / pixel_labeled
    return pixel_accuracy, pixel_correct, pixel_labeled


def calc_intersection_and_union(img_prediction,
                                img_label,
                                num_classes):
    """
    Calculate intersection and union areas for each class for the prediction and label of a single image.
    """

    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    img_prediction = img_prediction * (img_label > 0)

    # Compute area intersection:
    intersection = img_prediction * (img_prediction == img_label)
    (area_intersection, _) = np.histogram(intersection, bins=num_classes, range=(1, num_classes))

    # Compute area union:
    (area_pred, _) = np.histogram(img_prediction, bins=num_classes, range=(1, num_classes))
    (area_lab, _) = np.histogram(img_label, bins=num_classes, range=(1, num_classes))
    area_union = area_pred + area_lab - area_intersection

    return area_intersection, area_union
