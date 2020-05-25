"""
Evaluation Metrics for Object Detection.
"""

import os
import math
import warnings
import numpy as np
import mxnet as mx
from collections import defaultdict

__all__ = ['CocoDetMApMetric', 'VOC07MApMetric', 'WiderfaceDetMetric']


class CocoDetMApMetric(mx.metric.EvalMetric):
    """
    Detection metric for COCO bbox task.

    Parameters
    ----------
    img_height : int
        Processed image height.
    coco_annotations_file_path : str
        COCO anotation file path.
    contiguous_id_to_json : list of int
        Processed IDs.
    validation_ids : bool, default False
        Whether to use temporary file for estimation.
    use_file : bool, default False
        Whether to use temporary file for estimation.
    score_thresh : float, default 0.05
        Detection results with confident scores smaller than `score_thresh` will be discarded before saving to results.
    data_shape : tuple of int, default is None
        If `data_shape` is provided as (height, width), we will rescale bounding boxes when saving the predictions.
        This is helpful when SSD/YOLO box predictions cannot be rescaled conveniently. Note that the data_shape must be
        fixed for all validation images.
    post_affine : a callable function with input signature (orig_w, orig_h, out_w, out_h)
        If not None, the bounding boxes will be affine transformed rather than simply scaled.
    name : str, default 'mAP'
        Name of this metric instance for display.
    """
    def __init__(self,
                 img_height,
                 coco_annotations_file_path,
                 contiguous_id_to_json,
                 validation_ids=None,
                 use_file=False,
                 score_thresh=0.05,
                 data_shape=None,
                 post_affine=None,
                 name="mAP"):
        super(CocoDetMApMetric, self).__init__(name=name)
        self.img_height = img_height
        self.coco_annotations_file_path = coco_annotations_file_path
        self.contiguous_id_to_json = contiguous_id_to_json
        self.validation_ids = validation_ids
        self.use_file = use_file
        self.score_thresh = score_thresh

        self.current_idx = 0
        self.coco_result = []

        if isinstance(data_shape, (tuple, list)):
            assert len(data_shape) == 2, "Data shape must be (height, width)"
        elif not data_shape:
            data_shape = None
        else:
            raise ValueError("data_shape must be None or tuple of int as (height, width)")
        self._data_shape = data_shape

        if post_affine is not None:
            assert self._data_shape is not None, "Using post affine transform requires data_shape"
            self._post_affine = post_affine
        else:
            self._post_affine = None

        from pycocotools.coco import COCO
        self.gt = COCO(self.coco_annotations_file_path)
        self._img_ids = sorted(self.gt.getImgIds())

    def reset(self):
        self.current_idx = 0
        self.coco_result = []

    def get(self):
        """
        Get evaluation metrics.
        """
        if self.current_idx != len(self._img_ids):
            warnings.warn("Recorded {} out of {} validation images, incomplete results".format(
                self.current_idx, len(self._img_ids)))

        from pycocotools.coco import COCO
        gt = COCO(self.coco_annotations_file_path)

        import tempfile
        import json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
            json.dump(self.coco_result, f)
            f.flush()
            pred = gt.loadRes(f.name)

        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(gt, pred, "bbox")
        if self.validation_ids is not None:
            coco_eval.params.imgIds = self.validation_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return self.name, tuple(coco_eval.stats[:3])

    def update2(self,
                pred_bboxes,
                pred_labels,
                pred_scores):
        """
        Update internal buffer with latest predictions. Note that the statistics are not available until you call
        self.get() to return the metrics.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or np.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or np.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or np.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        """
        def as_numpy(a):
            """
            Convert a (list of) mx.NDArray into np.ndarray
            """
            if isinstance(a, (list, tuple)):
                out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
                return np.concatenate(out, axis=0)
            elif isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        for pred_bbox, pred_label, pred_score in zip(*[as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores]]):
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :].astype(np.float)
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred].astype(np.float)

            imgid = self._img_ids[self.current_idx]
            self.current_idx += 1
            affine_mat = None
            if self._data_shape is not None:
                entry = self.gt.loadImgs(imgid)[0]
                orig_height = entry["height"]
                orig_width = entry["width"]
                height_scale = float(orig_height) / self._data_shape[0]
                width_scale = float(orig_width) / self._data_shape[1]
                if self._post_affine is not None:
                    affine_mat = self._post_affine(orig_width, orig_height, self._data_shape[1], self._data_shape[0])
            else:
                height_scale, width_scale = (1.0, 1.0)
            # for each bbox detection in each image
            for bbox, label, score in zip(pred_bbox, pred_label, pred_score):
                if label not in self.contiguous_id_to_json:
                    # ignore non-exist class
                    continue
                if score < self.score_thresh:
                    continue
                category_id = self.contiguous_id_to_json[label]
                # rescale bboxes/affine transform bboxes
                if affine_mat is not None:
                    bbox[0:2] = self.affine_transform(bbox[0:2], affine_mat)
                    bbox[2:4] = self.affine_transform(bbox[2:4], affine_mat)
                else:
                    bbox[[0, 2]] *= width_scale
                    bbox[[1, 3]] *= height_scale
                # convert [xmin, ymin, xmax, ymax]  to [xmin, ymin, w, h]
                bbox[2:4] -= (bbox[:2] - 1)
                self.coco_result.append({"image_id": imgid,
                                         "category_id": category_id,
                                         "bbox": bbox[:4].tolist(),
                                         "score": score})

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
        det_bboxes = []
        det_ids = []
        det_scores = []
        for x_rr, y in zip(preds, labels):
            bboxes = x_rr.slice_axis(axis=-1, begin=0, end=4)
            ids = x_rr.slice_axis(axis=-1, begin=4, end=5).squeeze(axis=2)
            scores = x_rr.slice_axis(axis=-1, begin=5, end=6).squeeze(axis=2)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, self.img_height))
        self.update2(det_bboxes, det_ids, det_scores)

    @staticmethod
    def affine_transform(pt, t):
        """
        Apply affine transform to a bounding box given transform matrix t.

        Parameters
        ----------
        pt : np.ndarray
            Bounding box with shape (1, 2).
        t : np.ndarray
            Transformation matrix with shape (2, 3).

        Returns
        -------
        np.ndarray
            New bounding box with shape (1, 2).
        """
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]


class VOCMApMetric(mx.metric.EvalMetric):
    """
    Calculate mean AP for object detection task

    Parameters:
    ---------
    iou_thresh : float
        IOU overlap threshold for TP
    class_names : list of str
        optional, if provided, will print out AP for each class
    name : str, default 'mAP'
        Name of this metric instance for display.
    """
    def __init__(self,
                 iou_thresh=0.5,
                 class_names=None,
                 name="mAP"):
        super(VOCMApMetric, self).__init__(name=name)
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.name = list(class_names) + ["mAP"]
            self.num = num + 1
        self.reset()
        self.iou_thresh = iou_thresh
        self.class_names = class_names

    def reset(self):
        """
        Clear the internal statistics to initial state.
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        self._n_pos = defaultdict(int)
        self._score = defaultdict(list)
        self._match = defaultdict(list)

    def get(self):
        """
        Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        self._update()  # update metric at this time
        if self.num is None:
            if self.num_inst == 0:
                return self.name, float("nan")
            else:
                return self.name, self.sum_metric / self.num_inst
        else:
            names = ["%s" % self.name[i] for i in range(self.num)]
            values = [x / y if y != 0 else float("nan") for x, y in zip(self.sum_metric, self.num_inst)]
            return names, values

    def update(self,
               pred_bboxes,
               pred_labels,
               pred_scores,
               gt_bboxes,
               gt_labels,
               gt_difficults=None):
        """
        Update internal buffer with latest prediction and gt pairs.

        Parameters
        ----------
        pred_bboxes : mxnet.NDArray or np.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or np.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or np.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        gt_bboxes : mxnet.NDArray or np.ndarray
            Ground-truth bounding boxes with shape `B, M, 4`.
            Where B is the size of mini-batch, M is the number of ground-truths.
        gt_labels : mxnet.NDArray or np.ndarray
            Ground-truth bounding boxes labels with shape `B, M`.
        gt_difficults : mxnet.NDArray or np.ndarray, optional, default is None
            Ground-truth bounding boxes difficulty labels with shape `B, M`.
        """
        def as_numpy(a):
            """
            Convert a (list of) mx.NDArray into np.ndarray.
            """
            if isinstance(a, (list, tuple)):
                out = [x.asnumpy() if isinstance(x, mx.nd.NDArray) else x for x in a]
                try:
                    out = np.concatenate(out, axis=0)
                except ValueError:
                    out = np.array(out)
                return out
            elif isinstance(a, mx.nd.NDArray):
                a = a.asnumpy()
            return a

        if gt_difficults is None:
            gt_difficults = [None for _ in as_numpy(gt_labels)]

        if isinstance(gt_labels, list):
            gt_diff_shape = gt_difficults[0].shape[0] if hasattr(gt_difficults[0], "shape") else 0
            if len(gt_difficults) * gt_diff_shape != \
                    len(gt_labels) * gt_labels[0].shape[0]:
                gt_difficults = [None] * len(gt_labels) * gt_labels[0].shape[0]

        for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in zip(
                *[as_numpy(x) for x in [pred_bboxes, pred_labels, pred_scores,
                                        gt_bboxes, gt_labels, gt_difficults]]):
            # strip padding -1 for pred and gt
            valid_pred = np.where(pred_label.flat >= 0)[0]
            pred_bbox = pred_bbox[valid_pred, :]
            pred_label = pred_label.flat[valid_pred].astype(int)
            pred_score = pred_score.flat[valid_pred]
            valid_gt = np.where(gt_label.flat >= 0)[0]
            gt_bbox = gt_bbox[valid_gt, :]
            gt_label = gt_label.flat[valid_gt].astype(int)
            if gt_difficult is None:
                gt_difficult = np.zeros(gt_bbox.shape[0])
            else:
                gt_difficult = gt_difficult.flat[valid_gt]

            for ll in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):
                pred_mask_l = pred_label == ll
                pred_bbox_l = pred_bbox[pred_mask_l]
                pred_score_l = pred_score[pred_mask_l]
                # sort by score
                order = pred_score_l.argsort()[::-1]
                pred_bbox_l = pred_bbox_l[order]
                pred_score_l = pred_score_l[order]

                gt_mask_l = gt_label == ll
                gt_bbox_l = gt_bbox[gt_mask_l]
                gt_difficult_l = gt_difficult[gt_mask_l]

                self._n_pos[ll] += np.logical_not(gt_difficult_l).sum()
                self._score[ll].extend(pred_score_l)

                if len(pred_bbox_l) == 0:
                    continue
                if len(gt_bbox_l) == 0:
                    self._match[ll].extend((0,) * pred_bbox_l.shape[0])
                    continue

                # VOC evaluation follows integer typed bounding boxes.
                pred_bbox_l = pred_bbox_l.copy()
                pred_bbox_l[:, 2:] += 1
                gt_bbox_l = gt_bbox_l.copy()
                gt_bbox_l[:, 2:] += 1

                iou = self.bbox_iou(pred_bbox_l, gt_bbox_l)
                gt_index = iou.argmax(axis=1)
                # set -1 if there is no matching ground truth
                gt_index[iou.max(axis=1) < self.iou_thresh] = -1
                del iou

                selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
                for gt_idx in gt_index:
                    if gt_idx >= 0:
                        if gt_difficult_l[gt_idx]:
                            self._match[ll].append(-1)
                        else:
                            if not selec[gt_idx]:
                                self._match[ll].append(1)
                            else:
                                self._match[ll].append(0)
                        selec[gt_idx] = True
                    else:
                        self._match[ll].append(0)

    def _update(self):
        """
        Update num_inst and sum_metric.
        """
        aps = []
        recall, precs = self._recall_prec()
        for ll, rec, prec in zip(range(len(precs)), recall, precs):
            ap = self._average_precision(rec, prec)
            aps.append(ap)
            if self.num is not None and ll < (self.num - 1):
                self.sum_metric[ll] = ap
                self.num_inst[ll] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.nanmean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.nanmean(aps)

    def _recall_prec(self):
        """
        Get recall and precision from internal records.
        """
        n_fg_class = max(self._n_pos.keys()) + 1
        prec = [None] * n_fg_class
        rec = [None] * n_fg_class

        for ll in self._n_pos.keys():
            score_l = np.array(self._score[ll])
            match_l = np.array(self._match[ll], dtype=np.int32)

            order = score_l.argsort()[::-1]
            match_l = match_l[order]

            tp = np.cumsum(match_l == 1)
            fp = np.cumsum(match_l == 0)

            # If an element of fp + tp is 0,
            # the corresponding element of prec[ll] is nan.
            with np.errstate(divide="ignore", invalid="ignore"):
                prec[ll] = tp / (fp + tp)
            # If n_pos[ll] is 0, rec[ll] is None.
            if self._n_pos[ll] > 0:
                rec[ll] = tp / self._n_pos[ll]

        return rec, prec

    def _average_precision(self,
                           rec,
                           prec):
        """
        Calculate average precision.

        Params:
        ----------
        rec : np.array
            cumulated recall
        prec : np.array
            cumulated precision

        Returns:
        ----------
        float
            AP
        """
        if rec is None or prec is None:
            return np.nan

        # append sentinel values at both ends
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], np.nan_to_num(prec), [0.0]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    @staticmethod
    def bbox_iou(bbox_a, bbox_b, offset=0):
        """
        Calculate Intersection-Over-Union(IOU) of two bounding boxes.

        Parameters
        ----------
        bbox_a : np.ndarray
            An ndarray with shape :math:`(N, 4)`.
        bbox_b : np.ndarray
            An ndarray with shape :math:`(M, 4)`.
        offset : float or int, default is 0
            The ``offset`` is used to control the whether the width(or height) is computed as
            (right - left + ``offset``).
            Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

        Returns
        -------
        np.ndarray
            An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
            bounding boxes in `bbox_a` and `bbox_b`.
        """
        if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
            raise IndexError("Bounding boxes axis 1 must have at least length 4")

        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

        area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
        area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
        return area_i / (area_a[:, None] + area_b - area_i)


class VOC07MApMetric(VOCMApMetric):
    """
    Mean average precision metric for PASCAL V0C 07 dataset.

    Parameters:
    ---------
    iou_thresh : float
        IOU overlap threshold for TP
    class_names : list of str
        optional, if provided, will print out AP for each class
    """
    def __init__(self, *args, **kwargs):
        super(VOC07MApMetric, self).__init__(*args, **kwargs)

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric

        Params:
        ----------
        rec : np.array
            cumulated recall
        prec : np.array
            cumulated precision

        Returns:
        ----------
        float
            AP
        """
        if rec is None or prec is None:
            return np.nan
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(np.nan_to_num(prec)[rec >= t])
            ap += p / 11.0
        return ap


class WiderfaceDetMetric(mx.metric.EvalMetric):
    """
    Detection metric for WIDER FACE detection task.

    Parameters
    ----------
    receptive_field_center_starts : list of int
        The start location of the first receptive field of each scale.
    receptive_field_strides : list of int
        Receptive field stride for each scale.
    bbox_factors : list of float
        A half of bbox upper bound for each scale.
    output_dir_path : str
        Output file path.
    name : str, default 'WF'
        Name of this metric instance for display.
    """
    def __init__(self,
                 receptive_field_center_starts,
                 receptive_field_strides,
                 bbox_factors,
                 output_dir_path,
                 name="WF"):
        super(WiderfaceDetMetric, self).__init__(name=name)
        self.receptive_field_center_starts = receptive_field_center_starts
        self.receptive_field_strides = receptive_field_strides
        self.bbox_factors = bbox_factors
        self.output_dir_path = output_dir_path
        self.num_output_scales = len(self.bbox_factors)

        self.score_threshold = 0.11
        self.nms_threshold = 0.4
        self.top_k = 10000

    def reset(self):
        pass

    def get(self):
        """
        Get evaluation metrics.
        """
        return self.name, 1.0

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
        for x_rr, label in zip(preds, labels):
            outputs = []
            for output in x_rr:
                outputs.append(output.asnumpy())

            label_split = label.split("/")

            resize_scale = float(label_split[2])
            image_size = (int(label_split[3]), int(label_split[4]))

            bboxes, _ = self.predict(outputs, resize_scale, image_size)

            event_name = label_split[0]
            event_dir_name = os.path.join(self.output_dir_path, event_name)
            if not os.path.exists(event_dir_name):
                os.makedirs(event_dir_name)
            file_stem = label_split[1]
            fout = open(os.path.join(event_dir_name, file_stem + ".txt"), "w")
            fout.write(file_stem + "\n")
            fout.write(str(len(bboxes)) + "\n")
            for bbox in bboxes:
                fout.write("%d %d %d %d %.03f" % (math.floor(bbox[0]),
                                                  math.floor(bbox[1]),
                                                  math.ceil(bbox[2] - bbox[0]),
                                                  math.ceil(bbox[3] - bbox[1]),
                                                  bbox[4] if bbox[4] <= 1 else 1) + "\n")
            fout.close()

    def predict(self, outputs, resize_scale, image_size):

        bbox_collection = []

        for i in range(self.num_output_scales):
            score_map = np.squeeze(outputs[i * 2], (0, 1))
            bbox_map = np.squeeze(outputs[i * 2 + 1], 0)

            RF_center_Xs = np.array(
                [self.receptive_field_center_starts[i] + self.receptive_field_strides[i] * x for x in
                 range(score_map.shape[1])])
            RF_center_Xs_mat = np.tile(RF_center_Xs, [score_map.shape[0], 1])
            RF_center_Ys = np.array(
                [self.receptive_field_center_starts[i] + self.receptive_field_strides[i] * y for y in
                 range(score_map.shape[0])])
            RF_center_Ys_mat = np.tile(RF_center_Ys, [score_map.shape[1], 1]).T

            x_lt_mat = RF_center_Xs_mat - bbox_map[0, :, :] * self.bbox_factors[i]
            y_lt_mat = RF_center_Ys_mat - bbox_map[1, :, :] * self.bbox_factors[i]
            x_rb_mat = RF_center_Xs_mat - bbox_map[2, :, :] * self.bbox_factors[i]
            y_rb_mat = RF_center_Ys_mat - bbox_map[3, :, :] * self.bbox_factors[i]

            x_lt_mat = x_lt_mat / resize_scale
            x_lt_mat[x_lt_mat < 0] = 0
            y_lt_mat = y_lt_mat / resize_scale
            y_lt_mat[y_lt_mat < 0] = 0
            x_rb_mat = x_rb_mat / resize_scale
            x_rb_mat[x_rb_mat > image_size[1]] = image_size[1]
            y_rb_mat = y_rb_mat / resize_scale
            y_rb_mat[y_rb_mat > image_size[0]] = image_size[0]

            select_index = np.where(score_map > self.score_threshold)
            for idx in range(select_index[0].size):
                bbox_collection.append((x_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        y_lt_mat[select_index[0][idx], select_index[1][idx]],
                                        x_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        y_rb_mat[select_index[0][idx], select_index[1][idx]],
                                        score_map[select_index[0][idx], select_index[1][idx]]))

        # NMS
        bbox_collection = sorted(bbox_collection, key=lambda item: item[-1], reverse=True)
        if len(bbox_collection) > self.top_k:
            bbox_collection = bbox_collection[0:self.top_k]
        bbox_collection_numpy = np.array(bbox_collection, dtype=np.float32)

        final_bboxes = self.nms(bbox_collection_numpy, self.nms_threshold)
        final_bboxes_ = []
        for i in range(final_bboxes.shape[0]):
            final_bboxes_.append((final_bboxes[i, 0], final_bboxes[i, 1], final_bboxes[i, 2], final_bboxes[i, 3],
                                  final_bboxes[i, 4]))

        return final_bboxes_

    @staticmethod
    def nms(boxes, overlap_threshold):
        if boxes.shape[0] == 0:
            return boxes

        if boxes.dtype != np.float32:
            boxes = boxes.astype(np.float32)

        pick = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        sc = boxes[:, 4]
        widths = x2 - x1
        heights = y2 - y1

        area = heights * widths
        idxs = np.argsort(sc)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))

        return boxes[pick]
