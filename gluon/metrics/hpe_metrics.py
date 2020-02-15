"""
Evaluation Metrics for Human Pose Estimationn.
"""

import cv2
import numpy as np
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

    def update(self, labels, preds):
        for label, pred in zip(labels, preds):
            # print("pred.s={}".format(pred.shape))
            label = label.asnumpy()
            pred = pred.asnumpy()

            label_score = label[:, 0]
            label_img_id = label[:, 1].astype(np.int32)
            label_center = label[:, 2:4]
            label_scale = label[:, 4:6]
            pred_keypoints = pred[:, :, :2]
            pred_score = pred[:, :, 2]

            pred_keypoints = self.recalc_pose(pred_keypoints, label_center, label_scale)

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

    @staticmethod
    def affine_transform(pt, t):
        new_pt = np.array([pt[0], pt[1], 1.]).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    @staticmethod
    def get_dir(src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    @staticmethod
    def get_3rd_point(a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    @staticmethod
    def get_affine_transform(center,
                             scale,
                             rot,
                             output_size,
                             shift=np.array([0, 0], dtype=np.float32),
                             inv=0):
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale])

        scale_tmp = scale
        src_w = scale_tmp[0]
        dst_w = output_size[0]
        dst_h = output_size[1]

        rot_rad = np.pi * rot / 180
        src_dir = CocoHpeOksApMetric.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * shift
        src[1, :] = center + src_dir + scale_tmp * shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

        src[2:, :] = CocoHpeOksApMetric.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = CocoHpeOksApMetric.get_3rd_point(dst[0, :], dst[1, :])

        if inv:
            trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
        else:
            trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

        return trans

    @staticmethod
    def transform_preds(coords, center, scale, output_size):
        target_coords = np.zeros(coords.shape)
        trans = CocoHpeOksApMetric.get_affine_transform(center, scale, 0, output_size, inv=1)
        for p in range(coords.shape[0]):
            target_coords[p, 0:2] = CocoHpeOksApMetric.affine_transform(coords[p, 0:2], trans)
        return target_coords

    @staticmethod
    def recalc_pose(keypoints, center, scale):
        heatmap_height = 256 // 4
        heatmap_width = 192 // 4
        output_size = [heatmap_width, heatmap_height]

        preds = np.zeros_like(keypoints)

        for i in range(keypoints.shape[0]):
            preds[i] = CocoHpeOksApMetric.transform_preds(keypoints[i], center[i], scale[i], output_size)

        return preds
