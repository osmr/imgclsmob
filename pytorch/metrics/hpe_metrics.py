"""
Evaluation Metrics for Human Pose Estimation.
"""

import numpy as np
from .metric import EvalMetric

__all__ = ['CocoHpeOksApMetric', 'MpiiHpePckhMetric']


class CocoHpeOksApMetric(EvalMetric):
    """
    Detection metric for COCO bbox task.

    Parameters
    ----------
    coco : object
        An instance of pycocotools object.
    recalc_pose_fn : func
        An function for pose recalculation.
    name : str, default 'CocoOksAp'
        Name of this metric instance for display.
    in_vis_thresh : float
        Detection results with confident scores smaller than `in_vis_thresh` will
        be discarded before saving to results.
    """
    def __init__(self,
                 coco,
                 recalc_pose_fn,
                 name="CocoOksAp",
                 in_vis_thresh=0.2):
        super(CocoHpeOksApMetric, self).__init__(name=name)
        self.coco = coco
        self.recalc_pose_fn = recalc_pose_fn
        self._in_vis_thresh = in_vis_thresh
        self._results = []

    def reset(self):
        self._results = []

    def get(self):
        """
        Get evaluation metrics.
        """

        def calc_pred(coco, anns):
            from pycocotools.coco import COCO
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

        gt = self.coco
        pred = calc_pred(self.coco, self._results)

        from pycocotools.cocoeval import COCOeval
        coco_eval = COCOeval(gt, pred, "keypoints")
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # from collections import OrderedDict
        # stats_names = ["AP", "Ap .5", "AP .75", "AP (M)", "AP (L)",
        #                "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"]
        # info_str = []
        # for ind, name in enumerate(stats_names):
        #     info_str.append((name, coco_eval.stats[ind]))
        # name_value = OrderedDict(info_str)
        # return name_value, name_value["AP"]

        return self.name, tuple(coco_eval.stats[:3])

    def update(self, labels, preds):
        label = labels.cpu().detach().numpy()
        pred = preds.cpu().detach().numpy()

        label_score = label[:, 0]
        label_img_id = label[:, 1].astype(np.int32)
        label_bbs = label[:, 2:6]
        pred_keypoints = pred[:, :, :2]
        pred_score = pred[:, :, 2]

        pred_keypoints = self.recalc_pose_fn(pred_keypoints, label_bbs)

        num_joints = pred_keypoints.shape[1]
        for idx, kpt in enumerate(pred_keypoints):
            kpt = []
            kpt_score = 0
            count = 0
            for i in range(num_joints):
                kpt += pred_keypoints[idx][i].tolist()
                mval = float(pred_score[idx][i])
                kpt.append(mval)
                if mval > self._in_vis_thresh:
                    kpt_score += mval
                    count += 1

            if count > 0:
                kpt_score /= count
            rescore = kpt_score * float(label_score[idx])

            self._results.append({
                "image_id": int(label_img_id[idx]),
                "category_id": 1,
                "keypoints": kpt,
                "score": rescore})


class MpiiHpePckhMetric(EvalMetric):
    """
    Detection metric for MPII bbox task.

    Parameters
    ----------
    recalc_pose_fn : func
        An function for pose recalculation.
    name : str, default 'MpiiPCKh'
        Name of this metric instance for display.
    """
    def __init__(self,
                 recalc_pose_fn,
                 name="MpiiPCKh"):
        super(MpiiHpePckhMetric, self).__init__(name=name)
        self.recalc_pose_fn = recalc_pose_fn
        self._results = []

    def reset(self):
        self._results = []

    def get(self):
        """
        Get evaluation metrics.
        """

        return self.name, None

    def update(self, labels, preds):
        pass
