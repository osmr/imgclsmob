"""
Evaluation Metrics for Human Pose Estimationn.
"""

from collections import OrderedDict
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
    data_shape : tuple of 2 int, default is None
        If `data_shape` is provided as (height, width), we will rescale bounding boxes when
        saving the predictions.
        This is helpful when SSD/YOLO box predictions cannot be rescaled conveniently. Note that
        the data_shape must be fixed for all validation images.
    """
    def __init__(self,
                 coco,
                 name="CocoOksAp",
                 in_vis_thresh=0.2,
                 data_shape=None):
        super(CocoHpeOksApMetric, self).__init__(name=name)
        self.coco = coco
        self._in_vis_thresh = in_vis_thresh
        self._data_shape = data_shape
        self._img_ids = sorted(self.coco.getImgIds())
        self._results = []

    def reset(self):
        self._results = []

    def get(self):
        """
        Get evaluation metrics.
        """
        from pycocotools.coco import COCO
        from pycocotools import mask as maskUtils
        import numpy as np
        import copy
        import time

        def loadRes(coco, resFile):
            """
            Load result file and return a result api object.
            :param   resFile (str)     : file name of result file
            :return: res (obj)         : result api object
            """
            res = COCO()
            res.dataset['images'] = [img for img in coco.dataset['images']]

            print('Loading and preparing results...')
            tic = time.time()
            anns = resFile
            annsImgIds = [ann['image_id'] for ann in anns]
            assert set(annsImgIds) == (set(annsImgIds) & set(coco.getImgIds())),\
                'Results do not correspond to current coco set'
            if 'caption' in anns[0]:
                imgIds = set([img['id'] for img in res.dataset['images']]) & set([ann['image_id'] for ann in anns])
                res.dataset['images'] = [img for img in res.dataset['images'] if img['id'] in imgIds]
                for id, ann in enumerate(anns):
                    ann['id'] = id + 1
            elif 'bbox' in anns[0] and not anns[0]['bbox'] == []:
                res.dataset['categories'] = copy.deepcopy(coco.dataset['categories'])
                for id, ann in enumerate(anns):
                    bb = ann['bbox']
                    x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                    if 'segmentation' not in ann:
                        ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                    ann['area'] = bb[2] * bb[3]
                    ann['id'] = id + 1
                    ann['iscrowd'] = 0
            elif 'segmentation' in anns[0]:
                res.dataset['categories'] = copy.deepcopy(coco.dataset['categories'])
                for id, ann in enumerate(anns):
                    # now only support compressed RLE format as segmentation results
                    ann['area'] = maskUtils.area(ann['segmentation'])
                    if 'bbox' not in ann:
                        ann['bbox'] = maskUtils.toBbox(ann['segmentation'])
                    ann['id'] = id + 1
                    ann['iscrowd'] = 0
            elif 'keypoints' in anns[0]:
                res.dataset['categories'] = copy.deepcopy(coco.dataset['categories'])
                for id, ann in enumerate(anns):
                    s = ann['keypoints']
                    x = s[0::3]
                    y = s[1::3]
                    x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                    ann['area'] = (x1 - x0) * (y1 - y0)
                    ann['id'] = id + 1
                    ann['bbox'] = [x0, y0, x1 - x0, y1 - y0]
            print('DONE (t={:0.2f}s)'.format(time.time() - tic))

            res.dataset['annotations'] = anns
            res.createIndex()
            return res

        pred = loadRes(self.coco, self._results)
        gt = self.coco

        from pycocotools.cocoeval import COCOeval

        coco_eval = COCOeval(gt, pred, "keypoints")
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        stats_names = ["AP", "Ap .5", "AP .75", "AP (M)", "AP (L)",
                       "AR", "AR .5", "AR .75", "AR (M)", "AR (L)"]

        info_str = []
        for ind, name in enumerate(stats_names):
            info_str.append((name, coco_eval.stats[ind]))
        name_value = OrderedDict(info_str)
        return name_value, name_value["AP"]

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
