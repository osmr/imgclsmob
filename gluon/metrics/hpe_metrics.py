"""
Evaluation Metrics for Human Pose Estimationn.
"""

import mxnet as mx

__all__ = ['CocoHpeOksApMetric']


class CocoHpeOksApMetric(mx.metric.EvalMetric):
    """
    Detection metric for COCO bbox task.

    Parameters
    ----------
    coco : object
        An instance of pycocotools object.
    name : str, default 'CocoOksAp'
        Name of this metric instance for display.
    in_vis_thresh : float
        Detection results with confident scores smaller than `in_vis_thresh` will
        be discarded before saving to results.
    """
    def __init__(self,
                 coco,
                 name="CocoOksAp",
                 in_vis_thresh=0.2):
        super(CocoHpeOksApMetric, self).__init__(name=name)
        self.coco = coco
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

        return self.name, coco_eval.stats[0]

    def update(self,
               preds,
               maxvals,
               score,
               img_id,
               *args,
               **kwargs):
        num_joints = preds.shape[1]
        in_vis_thresh = self._in_vis_thresh
        for idx, kpt in enumerate(preds):
            kpt = []
            kpt_score = 0
            count = 0
            for i in range(num_joints):
                kpt += preds[idx][i].asnumpy().tolist()
                mval = float(maxvals[idx][i].asscalar())
                kpt.append(mval)
                if mval > in_vis_thresh:
                    kpt_score += mval
                    count += 1

            if count > 0:
                kpt_score /= count
            rescore = kpt_score * score[idx].asscalar()

            self._results.append({
                "image_id": int(img_id[idx].asscalar()),
                "category_id": 1,
                "keypoints": kpt,
                "score": rescore})
