"""
    Script for evaluating trained image matching model on PyTorch (under development).
"""

import os
import time
import logging
import argparse
import numpy as np
import torch
from common.logger_utils import initialize_logging
from pytorch.utils import prepare_pt_context, prepare_model
from pytorch.dataset_utils import get_dataset_metainfo
from pytorch.dataset_utils import get_val_data_source
from pytorch.metrics.ret_metrics import PointDescriptionMatchRatio


def add_eval_parser_arguments(parser):
    """
    Create python script parameters (for eval specific subpart).

    Parameters:
    ----------
    parser : ArgumentParser
        ArgumentParser instance.
    """
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model from github repo")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="base data type for tensors")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume from previously saved parameters")
    parser.add_argument(
        "--calc-flops",
        dest="calc_flops",
        action="store_true",
        help="calculate FLOPs")
    parser.add_argument(
        "--calc-flops-only",
        dest="calc_flops_only",
        action="store_true",
        help="calculate FLOPs without quality estimation")
    parser.add_argument(
        "--data-subset",
        type=str,
        default="val",
        help="data subset. options are val and test")

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of gpus to use")
    parser.add_argument(
        "-j",
        "--num-data-workers",
        dest="num_workers",
        default=4,
        type=int,
        help="number of preprocessing workers")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="training batch size per device (CPU/GPU)")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="directory of saved models and log-files")
    parser.add_argument(
        "--logging-file-name",
        type=str,
        default="train.log",
        help="filename of training log")

    parser.add_argument(
        "--log-packages",
        type=str,
        default="mxnet, numpy",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="mxnet-cu100",
        help="list of pip packages for logging")

    parser.add_argument(
        "--disable-cudnn-autotune",
        action="store_true",
        help="disable cudnn autotune for segmentation models")
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="show progress bar")


def parse_args():
    """
    Parse python script parameters (common part).

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image matching (PyTorch/HPatches)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=str,
        default="HPatches",
        help="dataset name")
    parser.add_argument(
        "--work-dir",
        type=str,
        default=os.path.join("..", "imgclsmob_data"),
        help="path to working directory only for dataset root path preset")

    args, _ = parser.parse_known_args()
    dataset_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    dataset_metainfo.add_dataset_parser_arguments(
        parser=parser,
        work_dir_path=args.work_dir)

    add_eval_parser_arguments(parser)

    args = parser.parse_args()
    return args


class SuperPointFrontend(object):
    """ Wrapper around pytorch net to help with pre and post image processing. """

    def __init__(self,
                 nms_dist=4,
                 conf_thresh=0.015,
                 nn_thresh=0.7,
                 cuda=True):
        self.nms_dist = nms_dist
        self.conf_thresh = conf_thresh
        self.nn_thresh = nn_thresh  # L2 descriptor distance for good match.
        self.cell = 8  # Size of each output cell. Keep this fixed.
        self.border_remove = 4  # Remove points this close to the border.

    def nms_fast(self, in_corners, H, W, dist_thresh):
        """
        Run a faster approximate Non-Max-Suppression on numpy corners shaped:
          3xN [x_i,y_i,conf_i]^T

        Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
        are zeros. Iterate through all the 1's and convert them either to -1 or 0.
        Suppress points by setting nearby values to 0.

        Grid Value Legend:
        -1 : Kept.
         0 : Empty or suppressed.
         1 : To be processed (converted to either kept or supressed).

        NOTE: The NMS first rounds points to integers, so NMS distance might not
        be exactly dist_thresh. It also assumes points are within image boundaries.

        Inputs
          in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          H - Image height.
          W - Image width.
          dist_thresh - Distance to suppress, measured as an infinty norm distance.
        Returns
          nmsed_corners - 3xN numpy matrix with surviving corners.
          nmsed_inds - N length numpy vector with surviving corner indices.
        """
        grid = np.zeros((H, W)).astype(int)  # Track NMS data.
        inds = np.zeros((H, W)).astype(int)  # Store indices of points.
        # Sort by confidence and round to nearest int.
        inds1 = np.argsort(-in_corners[2, :])
        corners = in_corners[:, inds1]
        rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
        # Check for edge case of 0 or 1 corners.
        if rcorners.shape[1] == 0:
            return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
        if rcorners.shape[1] == 1:
            out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
            return out, np.zeros((1)).astype(int)
        # Initialize the grid.
        for i, rc in enumerate(rcorners.T):
            grid[rcorners[1, i], rcorners[0, i]] = 1
            inds[rcorners[1, i], rcorners[0, i]] = i
        # Pad the border of the grid, so that we can NMS points near the border.
        pad = dist_thresh
        grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
        # Iterate through points, highest to lowest conf, suppress neighborhood.
        count = 0
        for i, rc in enumerate(rcorners.T):
            # Account for top and left padding.
            pt = (rc[0] + pad, rc[1] + pad)
            if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
                grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
                grid[pt[1], pt[0]] = -1
                count += 1
        # Get all surviving -1's and return sorted array of remaining corners.
        keepy, keepx = np.where(grid == -1)
        keepy, keepx = keepy - pad, keepx - pad
        inds_keep = inds[keepy, keepx]
        out = corners[:, inds_keep]
        values = out[-1, :]
        inds2 = np.argsort(-values)
        out = out[:, inds2]
        out_inds = inds1[inds_keep[inds2]]
        return out, out_inds

    def run(self, net, img):
        """ Process a numpy image to extract points and descriptors.
        Input
          img - HxW numpy float32 input image in range [0,1].
        Output
          corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
          desc - 256xN numpy array of corresponding unit normalized descriptors.
          heatmap - HxW numpy heatmap in range [0,1] of point confidences.
          """
        import torch.nn as nn

        # assert img.ndim == 2, 'Image must be grayscale.'
        # assert img.dtype == np.float32, 'Image must be float32.'
        # H, W = img.shape[0], img.shape[1]
        # in_channels = img.copy()
        # in_channels = (in_channels.reshape(1, H, W))
        # in_channels = torch.from_numpy(in_channels)
        # in_channels = torch.autograd.Variable(in_channels).view(1, 1, H, W)
        # if self.cuda:
        #     in_channels = in_channels.cuda()
        inp = img
        H, W = img.shape[2], img.shape[3]
        # Forward pass of network.
        outs = net.forward(inp)
        semi, coarse_desc = outs[0], outs[1]
        # Convert pytorch -> numpy.
        semi = semi.data.cpu().numpy().squeeze()
        # --- Process points.
        dense = np.exp(semi)  # Softmax.
        dense = dense / (np.sum(dense, axis=0) + .00001)  # Should sum to 1.
        # Remove dustbin.
        nodust = dense[:-1, :, :]
        # Reshape to get full resolution heatmap.
        Hc = int(H / self.cell)
        Wc = int(W / self.cell)
        nodust = nodust.transpose(1, 2, 0)
        heatmap = np.reshape(nodust, [Hc, Wc, self.cell, self.cell])
        heatmap = np.transpose(heatmap, [0, 2, 1, 3])
        heatmap = np.reshape(heatmap, [Hc * self.cell, Wc * self.cell])
        xs, ys = np.where(heatmap >= self.conf_thresh)  # Confidence threshold.
        if len(xs) == 0:
            return np.zeros((3, 0)), None, None
        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = self.nms_fast(pts, H, W, dist_thresh=self.nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = self.border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]
        # --- Process descriptor.
        D = coarse_desc.shape[1]
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            # Interpolate into descriptor map using 2D point locations.
            samp_pts = torch.from_numpy(pts[:2, :].copy())
            samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
            samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
            samp_pts = samp_pts.transpose(0, 1).contiguous()
            samp_pts = samp_pts.view(1, 1, -1, 2)
            samp_pts = samp_pts.float()
            # if self.cuda:
            #     samp_pts = samp_pts.cuda()
            samp_pts = samp_pts.cuda()
            desc = nn.functional.grid_sample(coarse_desc, samp_pts)
            desc = desc.data.cpu().numpy().reshape(D, -1)
            desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]
        return pts, desc, heatmap


def warp_keypoints(src_pts, homography):
    src_hmg_pts = np.concatenate([src_pts, np.ones((src_pts.shape[0], 1))], axis=1)
    dst_hmg_pts = np.dot(src_hmg_pts, np.transpose(homography)).squeeze(axis=2)
    dst_pts = dst_hmg_pts[:, :2] / dst_hmg_pts[:, 2:]
    return dst_pts


def calc_filter_mask(pts, shape):
    mask = (pts[:, 0] >= 0) & (pts[:, 0] < shape[0]) & (pts[:, 1] >= 0) & (pts[:, 1] < shape[1])
    return mask


def select_k_best(pts,
                  confs,
                  max_count=300):
    inds = confs.argsort()[::-1][:max_count]
    return pts[inds, :], confs[inds]


def calc_repeatability_np(src_pts,
                          src_confs,
                          dst_pts,
                          dst_confs,
                          homography,
                          src_shape,
                          dst_shape,
                          distance_thresh=3):

    pred_src_pts = warp_keypoints(dst_pts, np.linalg.inv(homography))
    pred_src_mask = calc_filter_mask(pred_src_pts, src_shape)
    label_dst_pts, label_dst_confs = dst_pts[pred_src_mask, :], dst_confs[pred_src_mask]

    pred_dst_pts = warp_keypoints(src_pts, homography)
    pred_dst_mask = calc_filter_mask(pred_dst_pts, dst_shape)
    pred_dst_pts, pred_dst_confs = pred_dst_pts[pred_dst_mask, :], src_confs[pred_dst_mask]

    label_dst_pts, label_dst_confs = select_k_best(label_dst_pts, label_dst_confs)
    pred_dst_pts, pred_dst_confs = select_k_best(pred_dst_pts, pred_dst_confs)

    n_pred = pred_dst_pts.shape[0]
    n_label = label_dst_pts.shape[0]

    label_dst_pts = np.stack([label_dst_pts[:, 0], label_dst_pts[:, 1], label_dst_confs], axis=1)
    pred_dst_pts = np.stack([pred_dst_pts[:, 0], pred_dst_pts[:, 1], pred_dst_confs], axis=1)

    pred_dst_pts = np.expand_dims(pred_dst_pts, 1)
    label_dst_pts = np.expand_dims(label_dst_pts, 0)
    norm = np.linalg.norm(pred_dst_pts - label_dst_pts, ord=None, axis=2)

    count1 = 0
    count2 = 0
    if n_label != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
    if n_pred != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
    if n_pred + n_label > 0:
        repeatability = (count1 + count2) / (n_pred + n_label)
    else:
        repeatability = 0

    return n_pred, n_label, repeatability


def calc_detector_repeatability(test_data,
                                net,
                                use_cuda):
    tic = time.time()
    repeatabilities = []
    n1s = []
    n2s = []
    # det_metric = PointDetectionMatchRatio(pts_max_count=100)
    # det_metric.reset()
    desc_metric = PointDescriptionMatchRatio(pts_max_count=10)
    desc_metric.reset()
    with torch.no_grad():
        for data_src, data_dst, target in test_data:
            if use_cuda:
                data_src = data_src.cuda(non_blocking=True)
                data_dst = data_dst.cuda(non_blocking=True)
            # spf = SuperPointFrontend()
            # src_pts, src_confs, src_desc_map = spf.run(net, data_src)
            # dst_pts, dst_confs, dst_desc_map = spf.run(net, data_dst)
            # src_pts = [src_pts.transpose()[:, [1, 0]].astype(np.int32)]
            # dst_pts = [dst_pts.transpose()[:, [1, 0]].astype(np.int32)]

            src_pts, src_confs, src_desc_map = net(data_src)
            dst_pts, dst_confs, dst_desc_map = net(data_dst)

            src_shape = data_src.cpu().detach().numpy().shape[2:]
            dst_shape = data_dst.cpu().detach().numpy().shape[2:]
            # print("data_src.shape={}".format(data_src.shape))
            # print("data_dst.shape={}".format(data_dst.shape))

            # import cv2
            # scale_factor = 0.5
            # num_pts = 100
            #
            # src_img = data_src.squeeze(0).transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
            # src_img = cv2.cvtColor(src_img, cv2.COLOR_GRAY2RGB)
            # for i in range(min(src_pts[0].shape[0], num_pts)):
            #     assert (src_pts[0][i, 0] < src_shape[0])
            #     assert (src_pts[0][i, 1] < src_shape[1])
            #
            #     cv2.circle(
            #         src_img,
            #         (src_pts[0][i, 1], src_pts[0][i, 0]),
            #         5,
            #         (0, 0, 255),
            #         -1)
            # cv2.imshow(
            #     winname="src_img",
            #     mat=cv2.resize(
            #         src=src_img,
            #         dsize=None,
            #         fx=scale_factor,
            #         fy=scale_factor,
            #         interpolation=cv2.INTER_NEAREST))
            #
            # dst_img = data_dst.squeeze(0).transpose(0, 2).transpose(0, 1).cpu().detach().numpy()
            # dst_img = cv2.cvtColor(dst_img, cv2.COLOR_GRAY2RGB)
            # for i in range(min(dst_pts[0].shape[0], num_pts)):
            #     assert (dst_pts[0][i, 0] < dst_shape[0])
            #     assert (dst_pts[0][i, 1] < dst_shape[1])
            #
            #     cv2.circle(
            #         dst_img,
            #         (dst_pts[0][i, 1], dst_pts[0][i, 0]),
            #         5,
            #         (0, 0, 255),
            #         -1)
            # cv2.imshow(
            #     winname="dst_img",
            #     mat=cv2.resize(
            #         src=dst_img,
            #         dsize=None,
            #         fx=scale_factor,
            #         fy=scale_factor,
            #         interpolation=cv2.INTER_NEAREST))
            #
            # cv2.waitKey(0)

            # for i in range(len(src_pts)):
            #     homography = target.cpu().detach().numpy()
            #
            #     src_pts_np = src_pts[i].cpu().detach().numpy()
            #     src_confs_np = src_confs[i].cpu().detach().numpy()
            #
            #     dst_pts_np = dst_pts[i].cpu().detach().numpy()
            #     dst_confs_np = dst_confs[i].cpu().detach().numpy()
            #
            #     n1, n2, repeatability = calc_repeatability_np(
            #         src_pts_np,
            #         src_confs_np,
            #         dst_pts_np,
            #         dst_confs_np,
            #         homography,
            #         src_shape,
            #         dst_shape)
            #     n1s.append(n1)
            #     n2s.append(n2)
            #     repeatabilities.append(repeatability)

            # det_metric.update_alt(
            #     homography=target[0],
            #     src_pts=src_pts[0],
            #     dst_pts=dst_pts[0],
            #     src_confs=src_confs[0],
            #     dst_confs=dst_confs[0],
            #     src_img_size=src_shape,
            #     dst_img_size=dst_shape)
            desc_metric.update_alt(
                homography=target[0],
                src_pts=src_pts[0],
                dst_pts=dst_pts[0],
                src_descs=src_desc_map[0],
                dst_descs=dst_desc_map[0],
                src_img_size=src_shape,
                dst_img_size=dst_shape)

    logging.info("Average number of points in the first image: {}".format(np.mean(n1s)))
    logging.info("Average number of points in the second image: {}".format(np.mean(n2s)))
    logging.info("The repeatability: {:.4f}".format(np.mean(repeatabilities)))
    logging.info("Time cost: {:.4f} sec".format(time.time() - tic))


def main():
    """
    Main body of script.
    """
    args = parse_args()

    os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
    assert (args.batch_size == 1)

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    ds_metainfo = get_dataset_metainfo(dataset_name=args.dataset)
    ds_metainfo.update(args=args)

    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_cuda=use_cuda,
        net_extra_kwargs=ds_metainfo.test_net_extra_kwargs,
        load_ignore_extra=False,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        remove_module=False)

    test_data = get_val_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    calc_detector_repeatability(
        test_data=test_data,
        net=net,
        use_cuda=use_cuda)


if __name__ == "__main__":
    main()
