"""
Evaluation Metrics for Image Retrieval.
"""

import numpy as np
import torch
from .metric import EvalMetric

__all__ = ['PointDetectionMatchRatio', 'PointDescriptionMatchRatio']


class PointDetectionMatchRatio(EvalMetric):
    """
    Computes point detection match ratio (with mean residual).

    Parameters
    ----------
    pts_max_count : int
        Maximal count of points.
    axis : int, default 1
        The axis that represents classes
    name : str, default 'accuracy'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 pts_max_count,
                 axis=1,
                 name="pt_det_ratio",
                 output_names=None,
                 label_names=None):
        super(PointDetectionMatchRatio, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = axis
        self.pts_max_count = pts_max_count
        self.resudual_sum = 0.0
        self.resudual_count = 0

    def update_alt(self,
                   homography,
                   src_pts,
                   dst_pts,
                   src_confs,
                   dst_confs,
                   src_img_size,
                   dst_img_size):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        homography : torch.Tensor
            Homography (from source image to destination one).
        src_pts : torch.Tensor
            Detected points for the first (source) image.
        dst_pts : torch.Tensor
            Detected points for the second (destination) image.
        src_confs : torch.Tensor
            Confidences for detected points on the source image.
        dst_confs : torch.Tensor
            Confidences for detected points on the destination image.
        src_img_size : tuple of 2 int
            Size (H, W) of the source image.
        dst_img_size : tuple of 2 int
            Size (H, W) of the destination image.
        """
        assert (src_confs.argsort(descending=True).cpu().detach().numpy() == np.arange(src_confs.shape[0])).all()
        assert (dst_confs.argsort(descending=True).cpu().detach().numpy() == np.arange(dst_confs.shape[0])).all()

        max_dist_sat_value = 1e5
        eps = 1e-5

        # print("src_img_size={}".format(src_img_size))
        # print("dst_img_size={}".format(dst_img_size))

        homography = homography.to(src_pts.device)
        self.normalize_homography(homography)
        homography_inv = self.calc_homography_inv(homography)

        # print("homography={}".format(homography))
        # print("homography_inv={}".format(homography_inv))

        # print("src_pts={}".format(src_pts[:10, :].int()))

        src_pts = src_pts.flip(dims=(1,))
        dst_pts = dst_pts.flip(dims=(1,))

        # print("src_pts={}".format(src_pts[:10, :].int()))

        # print("src_pts.shape={}".format(src_pts.shape))
        # print("dst_pts.shape={}".format(dst_pts.shape))
        # print("src_pts={}".format(src_pts[:10, :].int()))
        # print("dst_pts={}".format(dst_pts[:10, :].int()))

        # with torch.no_grad():
        src_hmg_pts = self.calc_homogeneous_coords(src_pts.float())
        dst_hmg_pts = self.calc_homogeneous_coords(dst_pts.float())

        # print("src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))
        # print("dst_hmg_pts={}".format(dst_hmg_pts[:10, :].int()))

        src_hmg_pts, src_confs = self.filter_inside_points(
            src_hmg_pts,
            src_confs,
            homography,
            dst_img_size)
        dst_hmg_pts, dst_confs = self.filter_inside_points(
            dst_hmg_pts,
            dst_confs,
            homography_inv,
            src_img_size)

        # print("src_hmg_pts.shape={}".format(src_hmg_pts.shape))
        # print("dst_hmg_pts.shape={}".format(dst_hmg_pts.shape))
        #
        # print("src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))
        # print("dst_hmg_pts={}".format(dst_hmg_pts[:10, :].int()))

        src_pts_count = src_hmg_pts.shape[0]
        dst_pts_count = dst_hmg_pts.shape[0]

        src_pts_count2 = min(src_pts_count, self.pts_max_count)
        src_hmg_pts, conf_thr = self.filter_best_points(
            hmg_pts=src_hmg_pts,
            confs=src_confs,
            max_count=src_pts_count2,
            min_conf=None)

        dst_pts_count2 = min(dst_pts_count, self.pts_max_count)
        dst_hmg_pts, _ = self.filter_best_points(
            hmg_pts=dst_hmg_pts,
            confs=dst_confs,
            max_count=dst_pts_count2,
            min_conf=conf_thr)

        # print("src_hmg_pts.shape={}".format(src_hmg_pts.shape))
        # print("dst_hmg_pts.shape={}".format(dst_hmg_pts.shape))

        # print("src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))
        # print("dst_hmg_pts={}".format(dst_hmg_pts[:10, :].int()))

        preds_dst_hmg_pts = self.transform_points(
            src_hmg_pts,
            homography)

        # print("preds_dst_hmg_pts={}".format(preds_dst_hmg_pts[:10, :].int()))

        cost = self.calc_pairwise_distances(x=preds_dst_hmg_pts, y=dst_hmg_pts).cpu().detach().numpy()
        self.saturate_distance_matrix(
            dist_mat=cost,
            max_dist_thr=8.0,
            max_dist_sat=max_dist_sat_value)

        # print("cost.shape={}".format(cost.shape))

        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        # print("row_ind.shape={}".format(row_ind.shape))
        # print("col_ind.shape={}".format(col_ind.shape))

        resuduals = cost[row_ind, col_ind]
        resuduals = resuduals[resuduals < (max_dist_sat_value - eps)]
        resudual_count = len(resuduals)

        self.sum_metric += resudual_count
        self.global_sum_metric += resudual_count
        self.num_inst += src_pts_count2
        self.global_num_inst += src_pts_count2

        print("ratio_resudual={}".format(float(resudual_count) / src_pts_count2))

        if resudual_count != 0:
            self.resudual_sum += resuduals.sum()
            self.resudual_count += resudual_count

    @staticmethod
    def normalize_homography(homography):
        homography /= homography[2, 2]

    @staticmethod
    def calc_homography_inv(homography):
        homography_inv = homography.inverse()
        PointDetectionMatchRatio.normalize_homography(homography_inv)
        return homography_inv

    @staticmethod
    def calc_homogeneous_coords(pts):
        hmg_pts = torch.cat((pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)), dim=1)
        return hmg_pts

    @staticmethod
    def calc_cartesian_coords(hmg_pts):
        pts = hmg_pts[:, :2]
        return pts

    @staticmethod
    def transform_points(src_hmg_pts,
                         homography):
        # print("transform_points -> src_hmg_pts.shape={}".format(src_hmg_pts.shape))
        # print("transform_points -> homography.shape={}".format(homography.shape))

        # print("homography={}".format(homography))
        # print("transform_points -> src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))

        dst_hmg_pts = torch.matmul(src_hmg_pts, homography.t())

        # print("transform_points -> dst_hmg_pts={}".format(dst_hmg_pts[:10, :].int()))
        # print("transform_points -> dst_hmg_pts.shape={}".format(dst_hmg_pts.shape))

        dst_hmg_pts /= dst_hmg_pts[:, 2:]
        return dst_hmg_pts

    @staticmethod
    def calc_inside_pts_mask(pts,
                             img_size):
        eps = 1e-3
        border_size = 1.0
        border = border_size - eps
        mask = (pts[:, 0] >= border) & (pts[:, 0] < img_size[0] - border) &\
               (pts[:, 1] >= border) & (pts[:, 1] < img_size[1] - border)
        return mask

    @staticmethod
    def filter_inside_points(src_hmg_pts,
                             src_confs,
                             homography,
                             dst_img_size):
        # print("fip->src_hmg_pts.shape={}".format(src_hmg_pts.shape))
        # print("fip->src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))

        # print("fip->src_confs.shape={}".format(src_confs.shape))
        # print("fip->src_confs={}".format(src_confs[:10]))
        # print("homography_inv={}".format(homography))

        dst_hmg_pts = PointDetectionMatchRatio.transform_points(src_hmg_pts, homography)

        # print("fip->dst_hmg_pts.shape={}".format(dst_hmg_pts.shape))
        # print("fip->dst_hmg_pts={}".format(dst_hmg_pts[:10, :]))

        mask = PointDetectionMatchRatio.calc_inside_pts_mask(dst_hmg_pts, dst_img_size)

        # print("fip->mask={}".format(mask[:10]))
        # print("fip->mask.sum()={}".format(mask.sum()))

        return src_hmg_pts[mask], src_confs[mask]

    @staticmethod
    def filter_best_points(hmg_pts,
                           confs,
                           max_count,
                           min_conf=None):
        if min_conf is not None:
            max_ind = (confs < min_conf).nonzero()[0, 0].item()
            max_count = max(max_count, max_ind)
        inds = confs.argsort(descending=True)[:max_count]
        return hmg_pts[inds], confs[inds][-1]

    @staticmethod
    def calc_pairwise_distances(x, y):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.sum(diff * diff, dim=-1).sqrt()

    @staticmethod
    def saturate_distance_matrix(dist_mat,
                                 max_dist_thr,
                                 max_dist_sat):
        dist_mat[dist_mat > max_dist_thr] = max_dist_sat


class PointDescriptionMatchRatio(EvalMetric):
    """
    Computes point description match ratio.

    Parameters
    ----------
    pts_max_count : int
        Maximal count of points.
    dist_ratio_thr : float, default 0.9
        Distance ratio threshold for point filtering.
    axis : int, default 1
        The axis that represents classes
    name : str, default 'accuracy'
        Name of this metric instance for display.
    output_names : list of str, or None, default None
        Name of predictions that should be used when updating with update_dict.
        By default include all predictions.
    label_names : list of str, or None, default None
        Name of labels that should be used when updating with update_dict.
        By default include all labels.
    """
    def __init__(self,
                 pts_max_count,
                 dist_ratio_thr=0.95,
                 axis=1,
                 name="pt_desc_ratio",
                 output_names=None,
                 label_names=None):
        super(PointDescriptionMatchRatio, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = axis
        self.pts_max_count = pts_max_count
        self.dist_ratio_thr = dist_ratio_thr
        self.resudual_sum = 0.0
        self.resudual_count = 0

    def update_alt(self,
                   homography,
                   src_pts,
                   dst_pts,
                   src_descs,
                   dst_descs,
                   src_img_size,
                   dst_img_size):
        """
        Updates the internal evaluation result.

        Parameters
        ----------
        homography : torch.Tensor
            Homography (from source image to destination one).
        src_pts : torch.Tensor
            Detected points for the first (source) image.
        dst_pts : torch.Tensor
            Detected points for the second (destination) image.
        src_descs : torch.Tensor
            Descriptors for detected points on the source image.
        dst_descs : torch.Tensor
            Descriptors for detected points on the destination image.
        src_img_size : tuple of 2 int
            Size (H, W) of the source image.
        dst_img_size : tuple of 2 int
            Size (H, W) of the destination image.
        """
        # max_dist_sat_value = 1e5
        # eps = 1e-5

        homography = homography.to(src_pts.device)
        self.normalize_homography(homography)
        homography_inv = self.calc_homography_inv(homography)

        src_pts = src_pts.flip(dims=(1,))
        dst_pts = dst_pts.flip(dims=(1,))

        src_hmg_pts = self.calc_homogeneous_coords(src_pts.float())
        dst_hmg_pts = self.calc_homogeneous_coords(dst_pts.float())

        src_hmg_pts = self.filter_inside_points(
            src_hmg_pts,
            homography,
            dst_img_size)
        dst_hmg_pts = self.filter_inside_points(
            dst_hmg_pts,
            homography_inv,
            src_img_size)

        src_pts_count = src_hmg_pts.shape[0]
        dst_pts_count = dst_hmg_pts.shape[0]

        src_pts_count2 = min(src_pts_count, self.pts_max_count * 10)
        src_hmg_pts, src_descs = self.filter_best_points(
            hmg_pts=src_hmg_pts,
            descs=src_descs,
            max_count=src_pts_count2)

        dst_pts_count2 = min(dst_pts_count, self.pts_max_count * 10)
        dst_hmg_pts, dst_descs = self.filter_best_points(
            hmg_pts=dst_hmg_pts,
            descs=dst_descs,
            max_count=dst_pts_count2)

        dist_mat = self.calc_pairwise_distances(x=src_descs, y=dst_descs)
        vals, inds = dist_mat.topk(k=2, dim=1, largest=True, sorted=True)
        inds = inds[:, 0][(vals[:, 1] / vals[:, 0]) < 0.95]

        src_hmg_pts = src_hmg_pts[inds]
        preds_dst_hmg_pts = self.transform_points(
            src_hmg_pts,
            homography)

        print(preds_dst_hmg_pts)

        # self.saturate_distance_matrix(
        #     dist_mat=cost,
        #     max_dist_thr=8.0,
        #     max_dist_sat=max_dist_sat_value)
        #
        # # print("cost.shape={}".format(cost.shape))
        #
        # from scipy.optimize import linear_sum_assignment
        # row_ind, col_ind = linear_sum_assignment(cost)
        #
        # # print("row_ind.shape={}".format(row_ind.shape))
        # # print("col_ind.shape={}".format(col_ind.shape))
        #
        # resuduals = cost[row_ind, col_ind]
        # resuduals = resuduals[resuduals < (max_dist_sat_value - eps)]
        # resudual_count = len(resuduals)

        resudual_count = 1

        self.sum_metric += resudual_count
        self.global_sum_metric += resudual_count
        self.num_inst += src_pts_count2
        self.global_num_inst += src_pts_count2

        print("ratio_resudual={}".format(float(resudual_count) / src_pts_count2))

    @staticmethod
    def normalize_homography(homography):
        homography /= homography[2, 2]

    @staticmethod
    def calc_homography_inv(homography):
        homography_inv = homography.inverse()
        PointDetectionMatchRatio.normalize_homography(homography_inv)
        return homography_inv

    @staticmethod
    def calc_homogeneous_coords(pts):
        hmg_pts = torch.cat((pts, torch.ones((pts.shape[0], 1), dtype=pts.dtype, device=pts.device)), dim=1)
        return hmg_pts

    @staticmethod
    def calc_cartesian_coords(hmg_pts):
        pts = hmg_pts[:, :2]
        return pts

    @staticmethod
    def transform_points(src_hmg_pts,
                         homography):
        # print("transform_points -> src_hmg_pts.shape={}".format(src_hmg_pts.shape))
        # print("transform_points -> homography.shape={}".format(homography.shape))

        # print("homography={}".format(homography))
        # print("transform_points -> src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))

        dst_hmg_pts = torch.matmul(src_hmg_pts, homography.t())

        # print("transform_points -> dst_hmg_pts={}".format(dst_hmg_pts[:10, :].int()))
        # print("transform_points -> dst_hmg_pts.shape={}".format(dst_hmg_pts.shape))

        dst_hmg_pts /= dst_hmg_pts[:, 2:]
        return dst_hmg_pts

    @staticmethod
    def calc_inside_pts_mask(pts,
                             img_size):
        eps = 1e-3
        border_size = 1.0
        border = border_size - eps
        mask = (pts[:, 0] >= border) & (pts[:, 0] < img_size[0] - border) &\
               (pts[:, 1] >= border) & (pts[:, 1] < img_size[1] - border)
        return mask

    @staticmethod
    def filter_inside_points(src_hmg_pts,
                             homography,
                             dst_img_size):
        dst_hmg_pts = PointDetectionMatchRatio.transform_points(src_hmg_pts, homography)
        mask = PointDetectionMatchRatio.calc_inside_pts_mask(dst_hmg_pts, dst_img_size)
        return src_hmg_pts[mask]

    @staticmethod
    def filter_best_points(hmg_pts,
                           descs,
                           max_count):
        return hmg_pts[:max_count], descs[:max_count]

    @staticmethod
    def calc_pairwise_distances(x, y):
        diff = x.unsqueeze(1) - y.unsqueeze(0)
        return torch.sum(diff * diff, dim=-1).sqrt()

    @staticmethod
    def saturate_distance_matrix(dist_mat,
                                 max_dist_thr,
                                 max_dist_sat):
        dist_mat[dist_mat > max_dist_thr] = max_dist_sat
