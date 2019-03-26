"""
Routines for segmentation metrics on mx.ndarray.
"""

import numpy as np
import mxnet as mx

__all__ = ['seg_pixel_accuracy_nd', 'segm_mean_accuracy', 'segm_mean_iou', 'seg_mean_iou2_nd', 'segm_fw_iou',
           'segm_fw_iou2']


def seg_pixel_accuracy_nd(label_imask,
                          pred_imask,
                          vague_idx=-1,
                          use_vague=False,
                          macro_average=True,
                          empty_result=0.0):
    """
    The segmentation pixel accuracy (for MXNet nd-arrays).

    Parameters
    ----------
    label_imask : mx.nd.array
        Ground truth index mask (maybe batch of).
    pred_imask : mx.nd.array
        Predicted index mask (maybe batch of).
    vague_idx : int, default -1
        Index of masked pixels.
    use_vague : bool, default False
        Whether to use pixel masking.
    macro_average : bool, default True
        Whether to use micro or macro averaging.
    empty_result : float, default 0.0
        Result value for an image without any classes.

    Returns
    -------
    float or tuple of two floats
        PA metric value.
    """
    assert (label_imask.shape == pred_imask.shape)
    if use_vague:
        mask = (label_imask != vague_idx)
        sum_u_ij = mask.sum().asscalar()
        if sum_u_ij == 0:
            if macro_average:
                return empty_result
            else:
                return 0, 0
        sum_u_ii = ((label_imask == pred_imask) * mask).sum().asscalar()
    else:
        sum_u_ii = (label_imask == pred_imask).sum().asscalar()
        sum_u_ij = pred_imask.size
    if macro_average:
        return float(sum_u_ii) / sum_u_ij
    else:
        return sum_u_ii, sum_u_ij


def segm_mean_accuracy(label_hmask,
                       pred_imask):
    """
    The segmentation mean accuracy.

    Parameters
    ----------
    label_hmask : nd.array
        Ground truth one-hot mask.
    pred_imask : nd.array
        Predicted index mask.

    Returns
    -------
    float
        MA metric value.
    """
    assert (len(label_hmask.shape) == 3)
    assert (len(pred_imask.shape) == 2)
    assert (pred_imask.shape == label_hmask.shape[1:])
    n = label_hmask.shape[0]
    i_sum = 0
    acc_sum = 0.0
    for i in range(n):
        class_i_pred_mask = (pred_imask == i)
        class_i_label_mask = label_hmask[i, :, :]

        u_i = class_i_label_mask.sum().asscalar()
        if u_i == 0:
            continue

        u_ii = (class_i_pred_mask * class_i_label_mask).sum().asscalar()

        class_acc = float(u_ii) / u_i
        acc_sum += class_acc
        i_sum += 1

    if i_sum > 0:
        mean_acc = acc_sum / i_sum
    else:
        mean_acc = 1.0

    return mean_acc


def segm_mean_iou(label_hmask,
                  pred_imask):
    """
    The segmentation mean intersection over union.

    Parameters
    ----------
    label_hmask : nd.array
        Ground truth one-hot mask.
    pred_imask : nd.array
        Predicted index mask.

    Returns
    -------
    float
        MIoU metric value.
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

        u_i = class_i_label_mask.sum().asscalar()
        u_ji_sj = class_i_pred_mask.sum().asscalar()
        if (u_i + u_ji_sj) == 0:
            continue

        u_ii = (class_i_pred_mask * class_i_label_mask).sum().asscalar()

        acc_iou += float(u_ii) / (u_i + u_ji_sj - u_ii)
        i_sum += 1

    if i_sum > 0:
        mean_iou = acc_iou / i_sum
    else:
        mean_iou = 1.0

    return mean_iou


def seg_mean_iou2_nd(label_hmask,
                     pred_hmask):
    """
    The segmentation mean intersection over union.

    Parameters
    ----------
    label_hmask : nd.array
        Ground truth one-hot mask (batch of).
    pred_hmask : nd.array
        Predicted one-hot mask (batch of).

    Returns
    -------
    float
        MIoU metric value.
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


def segm_fw_iou(label_hmask,
                pred_imask):
    """
    The segmentation frequency weighted intersection over union.

    Parameters
    ----------
    label_hmask : nd.array
        Ground truth one-hot mask.
    pred_imask : nd.array
        Predicted index mask.

    Returns
    -------
    float
        FrIoU metric value.
    """
    assert (len(label_hmask.shape) == 3)
    assert (len(pred_imask.shape) == 2)
    assert (pred_imask.shape == label_hmask.shape[1:])
    n = label_hmask.shape[0]
    acc_iou = 0.0
    for i in range(n):
        class_i_pred_mask = (pred_imask == i)
        class_i_label_mask = label_hmask[i, :, :]

        u_i = class_i_label_mask.sum().asscalar()
        u_ji_sj = class_i_pred_mask.sum().asscalar()
        if (u_i + u_ji_sj) == 0:
            continue

        u_ii = (class_i_pred_mask * class_i_label_mask).sum().asscalar()

        acc_iou += float(u_i) * float(u_ii) / (u_i + u_ji_sj - u_ii)

    fw_factor = pred_imask.size

    return acc_iou / fw_factor


def segm_fw_iou2(label_hmask,
                 pred_imask):
    """
    The segmentation frequency weighted intersection over union.

    Parameters
    ----------
    label_hmask : nd.array
        Ground truth one-hot mask.
    pred_imask : nd.array
        Predicted index mask.

    Returns
    -------
    float
        FrIoU metric value.
    """
    assert (len(label_hmask.shape) == 3)
    assert (len(pred_imask.shape) == 2)
    assert (pred_imask.shape == label_hmask.shape[1:])
    n = label_hmask.shape[0]
    acc_iou = mx.nd.array([0.0], ctx=label_hmask.context)
    for i in range(n):
        class_i_pred_mask = (pred_imask == i)
        class_i_label_mask = label_hmask[i, :, :]

        u_i = class_i_label_mask.sum()
        u_ji_sj = class_i_pred_mask.sum()
        if (u_i + u_ji_sj).asscalar() == 0:
            continue

        u_ii = (class_i_pred_mask * class_i_label_mask).sum()

        acc_iou += mx.nd.cast(u_i, dtype=np.float32) *\
                   mx.nd.cast(u_ii, dtype=np.float32) / mx.nd.cast(u_i + u_ji_sj - u_ii, dtype=np.float32)

    fw_factor = pred_imask.size

    return acc_iou.asscalar() / fw_factor
