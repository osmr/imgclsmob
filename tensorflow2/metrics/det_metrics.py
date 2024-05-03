"""
Evaluation Metrics for Object Detection.
"""

import warnings
import numpy as np
import mxnet as mx

__all__ = ['CocoDetMApMetric']


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
        pred_bboxes : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes with shape `B, N, 4`.
            Where B is the size of mini-batch, N is the number of bboxes.
        pred_labels : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes labels with shape `B, N`.
        pred_scores : mxnet.NDArray or numpy.ndarray
            Prediction bounding boxes scores with shape `B, N`.
        """
        def as_numpy(a):
            """
            Convert a (list of) mx.NDArray into numpy.ndarray
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
        pt : numpy.ndarray
            Bounding box with shape (1, 2).
        t : numpy.ndarray
            Transformation matrix with shape (2, 3).

        Returns
        -------
        numpy.ndarray
            New bounding box with shape (1, 2).
        """
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]
