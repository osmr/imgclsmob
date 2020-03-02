"""
Evaluation Metrics for Human Pose Estimation (for Lightweight OpenPose).
"""

import numpy as np
import math
import cv2
from operator import itemgetter
from .metric import EvalMetric

__all__ = ['CocoHpe2OksApMetric']


class CocoHpe2OksApMetric(EvalMetric):
    """
    Detection metric for COCO bbox task #2.

    Parameters
    ----------
    annotations_file_path : str
        COCO anotation file path.
    name : str, default 'CocoOksAp'
        Name of this metric instance for display.
    """
    def __init__(self,
                 annotations_file_path,
                 name="CocoOksAp"):
        super(CocoHpe2OksApMetric, self).__init__(name=name)
        self.annotations_file_path = annotations_file_path
        self.coco_result = []

    def reset(self):
        self.coco_result = []

    def get(self):
        """
        Get evaluation metrics.
        """
        from pycocotools.coco import COCO

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

        gt = COCO(self.annotations_file_path)
        pred = calc_pred(gt, self.coco_result)

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
        # return self.name, 0.0

    def update(self, labels, preds):
        # label = labels.cpu().detach().numpy().astype(np.int32)
        # pred = preds.cpu().detach().numpy()
        label = np.expand_dims(labels.astype(np.int32), axis=0)
        pred = np.expand_dims(preds, axis=0)

        image_ids = label[:, 0]
        pads = label[:, 1:5]
        heights = label[:, 5]
        widths = label[:, 6]

        keypoints = 19
        stride = 8

        heatmap2ds = pred[:, :keypoints]
        paf2ds = pred[:, keypoints:(3 * keypoints)]

        batch = 1
        for batch_i in range(batch):
            image_id = int(image_ids[batch_i])
            pad = list(pads[batch_i])
            height = int(heights[batch_i])
            width = int(widths[batch_i])
            heatmap2d = heatmap2ds[batch_i]
            paf2d = paf2ds[batch_i]

            heatmaps = np.transpose(heatmap2d, (1, 2, 0))
            heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            heatmaps = heatmaps[pad[0]:heatmaps.shape[0] - pad[2], pad[1]:heatmaps.shape[1] - pad[3]:, :]
            heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)

            pafs = np.transpose(paf2d, (1, 2, 0))
            pafs = cv2.resize(pafs, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
            pafs = pafs[pad[0]:pafs.shape[0] - pad[2], pad[1]:pafs.shape[1] - pad[3], :]
            pafs = cv2.resize(pafs, (width, height), interpolation=cv2.INTER_CUBIC)

            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(18):  # 19th for bg
                total_keypoints_num += self.extract_keypoints(
                    heatmaps[:, :, kpt_idx],
                    all_keypoints_by_type,
                    total_keypoints_num)

            pose_entries, all_keypoints = self.group_keypoints(
                all_keypoints_by_type,
                pafs)

            coco_keypoints, scores = self.convert_to_coco_format(
                pose_entries,
                all_keypoints)

            for idx in range(len(coco_keypoints)):
                self.coco_result.append({
                    'image_id': image_id,
                    'category_id': 1,
                    'keypoints': coco_keypoints[idx],
                    'score': scores[idx]
                })

    @staticmethod
    def extract_keypoints(heatmap,
                          all_keypoints,
                          total_keypoint_num):
        heatmap[heatmap < 0.1] = 0
        heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")
        heatmap_center = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 1:heatmap_with_borders.shape[1] - 1]
        heatmap_left = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 2:heatmap_with_borders.shape[1]]
        heatmap_right = heatmap_with_borders[1:heatmap_with_borders.shape[0] - 1, 0:heatmap_with_borders.shape[1] - 2]
        heatmap_up = heatmap_with_borders[2:heatmap_with_borders.shape[0], 1:heatmap_with_borders.shape[1] - 1]
        heatmap_down = heatmap_with_borders[0:heatmap_with_borders.shape[0] - 2, 1:heatmap_with_borders.shape[1] - 1]

        heatmap_peaks = (heatmap_center > heatmap_left) &\
                        (heatmap_center > heatmap_right) &\
                        (heatmap_center > heatmap_up) &\
                        (heatmap_center > heatmap_down)
        heatmap_peaks = heatmap_peaks[1:heatmap_center.shape[0] - 1, 1:heatmap_center.shape[1] - 1]
        keypoints = list(zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0]))  # (w, h)
        keypoints = sorted(keypoints, key=itemgetter(0))

        suppressed = np.zeros(len(keypoints), np.uint8)
        keypoints_with_score_and_id = []
        keypoint_num = 0
        for i in range(len(keypoints)):
            if suppressed[i]:
                continue
            for j in range(i + 1, len(keypoints)):
                if math.sqrt((keypoints[i][0] - keypoints[j][0]) ** 2 + (keypoints[i][1] - keypoints[j][1]) ** 2) < 6:
                    suppressed[j] = 1
            keypoint_with_score_and_id = (
                keypoints[i][0],
                keypoints[i][1],
                heatmap[keypoints[i][1], keypoints[i][0]],
                total_keypoint_num + keypoint_num)
            keypoints_with_score_and_id.append(keypoint_with_score_and_id)
            keypoint_num += 1
        all_keypoints.append(keypoints_with_score_and_id)
        return keypoint_num

    @staticmethod
    def group_keypoints(all_keypoints_by_type,
                        pafs,
                        pose_entry_size=20,
                        min_paf_score=0.05):

        def linspace2d(start, stop, n=10):
            points = 1 / (n - 1) * (stop - start)
            return points[:, None] * np.arange(n) + start[:, None]

        BODY_PARTS_KPT_IDS = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                              [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]
        BODY_PARTS_PAF_IDS = ([12, 13], [20, 21], [14, 15], [16, 17], [22, 23], [24, 25], [0, 1], [2, 3], [4, 5],
                              [6, 7], [8, 9], [10, 11], [28, 29], [30, 31], [34, 35], [32, 33], [36, 37], [18, 19],
                              [26, 27])

        pose_entries = []
        all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
        for part_id in range(len(BODY_PARTS_PAF_IDS)):
            part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
            kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
            kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
            num_kpts_a = len(kpts_a)
            num_kpts_b = len(kpts_b)
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]

            if num_kpts_a == 0 and num_kpts_b == 0:  # no keypoints for such body part
                continue
            elif num_kpts_a == 0:  # body part has just 'b' keypoints
                for i in range(num_kpts_b):
                    num = 0
                    for j in range(len(pose_entries)):  # check if already in some pose, was added by another body part
                        if pose_entries[j][kpt_b_id] == kpts_b[i][3]:
                            num += 1
                            continue
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_b_id] = kpts_b[i][3]  # keypoint idx
                        pose_entry[-1] = 1                   # num keypoints in pose
                        pose_entry[-2] = kpts_b[i][2]        # pose score
                        pose_entries.append(pose_entry)
                continue
            elif num_kpts_b == 0:  # body part has just 'a' keypoints
                for i in range(num_kpts_a):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == kpts_a[i][3]:
                            num += 1
                            continue
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = kpts_a[i][3]
                        pose_entry[-1] = 1
                        pose_entry[-2] = kpts_a[i][2]
                        pose_entries.append(pose_entry)
                continue

            connections = []
            for i in range(num_kpts_a):
                kpt_a = np.array(kpts_a[i][0:2])
                for j in range(num_kpts_b):
                    kpt_b = np.array(kpts_b[j][0:2])
                    mid_point = [(), ()]
                    mid_point[0] = (int(round((kpt_a[0] + kpt_b[0]) * 0.5)),
                                    int(round((kpt_a[1] + kpt_b[1]) * 0.5)))
                    mid_point[1] = mid_point[0]

                    vec = [kpt_b[0] - kpt_a[0], kpt_b[1] - kpt_a[1]]
                    vec_norm = math.sqrt(vec[0] ** 2 + vec[1] ** 2)
                    if vec_norm == 0:
                        continue
                    vec[0] /= vec_norm
                    vec[1] /= vec_norm
                    cur_point_score = (vec[0] * part_pafs[mid_point[0][1], mid_point[0][0], 0] +
                                       vec[1] * part_pafs[mid_point[1][1], mid_point[1][0], 1])

                    height_n = pafs.shape[0] // 2
                    success_ratio = 0
                    point_num = 10  # number of points to integration over paf
                    if cur_point_score > -100:
                        passed_point_score = 0
                        passed_point_num = 0
                        x, y = linspace2d(kpt_a, kpt_b)
                        for point_idx in range(point_num):
                            px = int(round(x[point_idx]))
                            py = int(round(y[point_idx]))
                            paf = part_pafs[py, px, 0:2]
                            cur_point_score = vec[0] * paf[0] + vec[1] * paf[1]
                            if cur_point_score > min_paf_score:
                                passed_point_score += cur_point_score
                                passed_point_num += 1
                        success_ratio = passed_point_num / point_num
                        ratio = 0
                        if passed_point_num > 0:
                            ratio = passed_point_score / passed_point_num
                        ratio += min(height_n / vec_norm - 1, 0)
                    if ratio > 0 and success_ratio > 0.8:
                        score_all = ratio + kpts_a[i][2] + kpts_b[j][2]
                        connections.append([i, j, ratio, score_all])
            if len(connections) > 0:
                connections = sorted(connections, key=itemgetter(2), reverse=True)

            num_connections = min(num_kpts_a, num_kpts_b)
            has_kpt_a = np.zeros(num_kpts_a, dtype=np.int32)
            has_kpt_b = np.zeros(num_kpts_b, dtype=np.int32)
            filtered_connections = []
            for row in range(len(connections)):
                if len(filtered_connections) == num_connections:
                    break
                i, j, cur_point_score = connections[row][0:3]
                if not has_kpt_a[i] and not has_kpt_b[j]:
                    filtered_connections.append([kpts_a[i][3], kpts_b[j][3], cur_point_score])
                    has_kpt_a[i] = 1
                    has_kpt_b[j] = 1
            connections = filtered_connections
            if len(connections) == 0:
                continue

            if part_id == 0:
                pose_entries = [np.ones(pose_entry_size) * -1 for _ in range(len(connections))]
                for i in range(len(connections)):
                    pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                    pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                    pose_entries[i][-1] = 2
                    pose_entries[i][-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
            elif part_id == 17 or part_id == 18:
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                for i in range(len(connections)):
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0] and pose_entries[j][kpt_b_id] == -1:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                        elif pose_entries[j][kpt_b_id] == connections[i][1] and pose_entries[j][kpt_a_id] == -1:
                            pose_entries[j][kpt_a_id] = connections[i][0]
                continue
            else:
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                for i in range(len(connections)):
                    num = 0
                    for j in range(len(pose_entries)):
                        if pose_entries[j][kpt_a_id] == connections[i][0]:
                            pose_entries[j][kpt_b_id] = connections[i][1]
                            num += 1
                            pose_entries[j][-1] += 1
                            pose_entries[j][-2] += all_keypoints[connections[i][1], 2] + connections[i][2]
                    if num == 0:
                        pose_entry = np.ones(pose_entry_size) * -1
                        pose_entry[kpt_a_id] = connections[i][0]
                        pose_entry[kpt_b_id] = connections[i][1]
                        pose_entry[-1] = 2
                        pose_entry[-2] = np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                        pose_entries.append(pose_entry)

        filtered_entries = []
        for i in range(len(pose_entries)):
            if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
                continue
            filtered_entries.append(pose_entries[i])
        pose_entries = np.asarray(filtered_entries)
        return pose_entries, all_keypoints

    @staticmethod
    def convert_to_coco_format(pose_entries, all_keypoints):
        coco_keypoints = []
        scores = []
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            keypoints = [0] * 17 * 3
            to_coco_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            person_score = pose_entries[n][-2]
            position_id = -1
            for keypoint_id in pose_entries[n][:-2]:
                position_id += 1
                if position_id == 1:  # no 'neck' in COCO
                    continue

                cx, cy, score, visibility = 0, 0, 0, 0  # keypoint not found
                if keypoint_id != -1:
                    cx, cy, score = all_keypoints[int(keypoint_id), 0:3]
                    cx = cx + 0.5
                    cy = cy + 0.5
                    visibility = 1
                keypoints[to_coco_map[position_id] * 3 + 0] = cx
                keypoints[to_coco_map[position_id] * 3 + 1] = cy
                keypoints[to_coco_map[position_id] * 3 + 2] = visibility
            coco_keypoints.append(keypoints)
            scores.append(person_score * max(0, (pose_entries[n][-1] - 1)))  # -1 for 'neck'
        return coco_keypoints, scores
