"""
Evaluation Metrics for Image Retrieval.
"""

# import numpy as np
import torch
from .metric import EvalMetric

__all__ = ['PointDetectionMeanResidual']


class PointDetectionMeanResidual(EvalMetric):
    """
    Computes mean residual for point detection.

    Parameters
    ----------
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
                 axis=1,
                 name="pt_det_mean_res",
                 output_names=None,
                 label_names=None):
        super(PointDetectionMeanResidual, self).__init__(
            name,
            axis=axis,
            output_names=output_names,
            label_names=label_names,
            has_global_stats=True)
        self.axis = axis

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
        # from scipy.spatial.distance import pdist
        from scipy.optimize import linear_sum_assignment

        print("src_img_size={}".format(src_img_size))
        print("dst_img_size={}".format(dst_img_size))

        self.normalize_homography(homography)
        homography_inv = self.calc_homography_inv(homography)

        print("homography={}".format(homography))
        print("homography_inv={}".format(homography_inv))

        print("src_pts.shape={}".format(src_pts.shape))
        print("dst_pts.shape={}".format(dst_pts.shape))
        print("src_pts={}".format(src_pts[:10, :].int()))
        print("dst_pts={}".format(dst_pts[:10, :].int()))

        # with torch.no_grad():
        src_hmg_pts = self.calc_homogeneous_coords(src_pts.float())
        dst_hmg_pts = self.calc_homogeneous_coords(dst_pts.float())

        print("src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))
        print("dst_hmg_pts={}".format(dst_hmg_pts[:10, :].int()))

        self.filter_inside_points(
            src_hmg_pts,
            src_confs,
            homography,
            dst_img_size)
        self.filter_inside_points(
            dst_hmg_pts,
            dst_confs,
            homography_inv,
            src_img_size)

        print("src_hmg_pts.shape={}".format(src_hmg_pts.shape))
        print("dst_hmg_pts.shape={}".format(dst_hmg_pts.shape))

        src_pts_count = src_hmg_pts.shape[0]
        dst_pts_count = dst_hmg_pts.shape[0]
        pts_count = min(src_pts_count, dst_pts_count, 100)
        assert (pts_count > 0)
        src_hmg_pts = self.filter_best_points(
            src_hmg_pts,
            src_confs,
            pts_count)
        dst_hmg_pts = self.filter_best_points(
            dst_hmg_pts,
            dst_confs,
            pts_count)

        preds_dst_hmg_pts = self.transform_points(
            src_hmg_pts,
            homography)

        cost = torch.pairwise_distance(x1=preds_dst_hmg_pts, x2=dst_hmg_pts).cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(cost)
        mean_resudual = cost[row_ind, col_ind].sum()
        mean_resudual *= (100.0 / dst_img_size[0])

        self.sum_metric += mean_resudual
        self.global_sum_metric += mean_resudual
        self.num_inst += 1
        self.global_num_inst += 1

    @staticmethod
    def normalize_homography(homography):
        homography /= homography[2, 2]

    @staticmethod
    def calc_homography_inv(homography):
        homography_inv = homography.inverse()
        PointDetectionMeanResidual.normalize_homography(homography_inv)
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

        print("homography={}".format(homography))
        print("transform_points -> src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))

        dst_hmg_pts = torch.matmul(src_hmg_pts, homography.t())

        print("transform_points -> dst_hmg_pts={}".format(dst_hmg_pts[:10, :].int()))
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
                             homography_inv,
                             dst_img_size):
        print("fip->src_hmg_pts.shape={}".format(src_hmg_pts.shape))
        print("fip->src_hmg_pts={}".format(src_hmg_pts[:10, :].int()))
        print("fip->src_confs.shape={}".format(src_confs.shape))
        print("fip->src_confs={}".format(src_confs[:10]))
        print("homography_inv={}".format(homography_inv))

        dst_hmg_pts = PointDetectionMeanResidual.transform_points(src_hmg_pts, homography_inv)

        print("fip->dst_hmg_pts.shape={}".format(dst_hmg_pts.shape))
        print("fip->dst_hmg_pts={}".format(dst_hmg_pts[:10, :]))

        mask = PointDetectionMeanResidual.calc_inside_pts_mask(dst_hmg_pts, dst_img_size)

        print("fip->mask={}".format(mask[:10]))
        print("fip->mask.sum()={}".format(mask.sum()))

        src_hmg_pts[:] = src_hmg_pts[mask]
        src_confs[:] = src_confs[mask]

    @staticmethod
    def filter_best_points(hmg_pts,
                           confs,
                           max_count):
        inds = confs.argsort(descending=True)[:max_count]
        return hmg_pts[inds]
