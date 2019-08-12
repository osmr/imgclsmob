"""
Evaluation Metrics for Image Retrieval.
"""

import numpy as np
import torch
from torch import functional as F
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

    def update(self,
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
        from scipy.optimize import linear_sum_assignment

        with torch.no_grad():
            src_hmg_pts = self.calc_homogeneous_coords(src_pts)
            dst_hmg_pts = self.calc_homogeneous_coords(dst_pts)
            self.filter_inside_points(
                src_hmg_pts,
                src_confs,
                homography,
                dst_img_size)
            self.filter_inside_points(
                dst_hmg_pts,
                dst_confs,
                homography.inverse(),
                src_img_size)
            src_pts_count = src_hmg_pts.shape[1]
            dst_pts_count = dst_hmg_pts.shape[1]
            pts_count = min(src_pts_count, dst_pts_count, 100)
            assert (pts_count > 0)
            self.filter_best_points(
                src_hmg_pts,
                src_confs,
                pts_count)
            self.filter_best_points(
                dst_hmg_pts,
                dst_confs,
                pts_count)

            preds_dst_hmg_pts = self.transform_points(
                src_hmg_pts,
                homography)

            cost = F.pairwise_distance(preds_dst_hmg_pts, dst_hmg_pts, 2, 1e-6, False).cpu().detach().numpy()
            row_ind, col_ind = linear_sum_assignment(cost)
            mean_resudual = cost[row_ind, col_ind].sum()
            mean_resudual *= (100.0 / dst_img_size[0])

            self.sum_metric += mean_resudual
            self.global_sum_metric += mean_resudual
            self.num_inst += 1
            self.global_num_inst += 1

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
        dst_hmg_pts = np.dot(src_hmg_pts, homography).squeeze(axis=2)
        dst_hmg_pts /= dst_hmg_pts[:, 2:]
        return dst_hmg_pts

    @staticmethod
    def calc_outside_pts_mask(pts,
                              img_size):
        mask = (pts[:, 0] >= 0) & (pts[:, 0] < img_size[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < img_size[1])
        return mask

    @staticmethod
    def filter_inside_points(src_hmg_pts,
                             src_confs,
                             homography,
                             dst_img_size):
        dst_hmg_pts = PointDetectionMeanResidual.transform_points(src_hmg_pts, homography)
        mask = PointDetectionMeanResidual.calc_outside_pts_mask(dst_hmg_pts, dst_img_size)
        src_hmg_pts[:] = src_hmg_pts[mask, :]
        src_confs[:] = src_confs[mask, :]

    @staticmethod
    def filter_best_points(hmg_pts,
                           confs,
                           max_count):
        inds = confs.argsort()[::-1][:max_count]
        hmg_pts[:] = hmg_pts[inds, :]
        confs[:] = confs[inds]
