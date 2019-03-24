"""
Evaluation Metrics for Semantic Segmentation
"""

import threading
import numpy as np
import mxnet as mx
from mxnet.metric import EvalMetric

__all__ = ['PixIoUSegMetric']


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
