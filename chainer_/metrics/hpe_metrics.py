"""
Evaluation Metrics for Human Pose Estimation.
"""

import numpy as np
from .metric import EvalMetric

__all__ = ['CocoHpeOksApMetric']


class CocoHpeOksApMetric(EvalMetric):
    """
    Detection metric for COCO bbox task.

    Parameters
    ----------
    coco_annotations_file_path : str
        COCO anotation file path.
    pose_postprocessing_fn : func
        An function for pose post-processing.
    use_file : bool, default False
        Whether to use temporary file for estimation.
    validation_ids : bool, default False
        Whether to use temporary file for estimation.
    name : str, default 'CocoOksAp'
        Name of this metric instance for display.
    """
    def __init__(self,
                 coco_annotations_file_path,
                 pose_postprocessing_fn,
                 validation_ids=None,
                 use_file=False,
                 name="CocoOksAp"):
        super(CocoHpeOksApMetric, self).__init__(name=name)
        self.coco_annotations_file_path = coco_annotations_file_path
        self.pose_postprocessing_fn = pose_postprocessing_fn
        self.validation_ids = validation_ids
        self.use_file = use_file
        self.coco_result = []

    def reset(self):
        self.coco_result = []

    def get(self):
        """
        Get evaluation metrics.
        """
        import copy
        from pycocotools.coco import COCO
        gt = COCO(self.coco_annotations_file_path)

        if self.use_file:
            import tempfile
            import json
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as f:
                json.dump(self.coco_result, f)
                f.flush()
                pred = gt.loadRes(f.name)
        else:
            def calc_pred(coco, anns):
                import numpy as np
                import copy

                pred = COCO()
                pred.dataset["images"] = [img for img in coco.dataset["images"]]

                annsImgIds = [ann["image_id"] for ann in anns]
                assert set(annsImgIds) == (set(annsImgIds) & set(coco.getImgIds()))

                pred.dataset["categories"] = copy.deepcopy(coco.dataset["categories"])
                for id, ann in enumerate(anns):
                    s = ann["keypoints"]
                    x = s[0::3]
                    y = s[1::3]
                    x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                    ann["area"] = (x1 - x0) * (y1 - y0)
                    ann["id"] = id + 1
                    ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]

                pred.dataset["annotations"] = anns
                pred.createIndex()
                return pred
            pred = calc_pred(gt, copy.deepcopy(self.coco_result))

        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(gt, pred, "keypoints")
        if self.validation_ids is not None:
            coco_eval.params.imgIds = self.validation_ids
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return self.name, tuple(coco_eval.stats[:3])

    def update(self, labels, preds):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        labels : xp.array
            The labels of the data.
        preds : xp.array
            Predicted values.
        """
        label = np.expand_dims(labels, axis=0)
        pred = np.expand_dims(preds, axis=0)

        pred_pts_score, pred_person_score, label_img_id = self.pose_postprocessing_fn(pred, label)

        for idx in range(len(pred_pts_score)):
            image_id = int(label_img_id[idx])
            kpt = pred_pts_score[idx].flatten().tolist()
            score = float(pred_person_score[idx])
            self.coco_result.append({
                "image_id": image_id,
                "category_id": 1,
                "keypoints": kpt,
                "score": score})
