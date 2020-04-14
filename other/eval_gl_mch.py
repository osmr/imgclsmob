"""
    Script for evaluating trained image matching model on MXNet/Gluon (under development).
"""

import os
import time
import logging
import argparse
import numpy as np
import mxnet as mx
from mxnet.gluon.utils import split_and_load
from common.logger_utils import initialize_logging
from gluon.utils import prepare_mx_context, prepare_model
from gluon.dataset_utils import get_dataset_metainfo
from gluon.dataset_utils import get_val_data_source


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
        description="Evaluate a model for image matching (Gluon/HPatches)",
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


def warp_keypoints(keypoints, H):
    num_points = keypoints.shape[0]
    homogeneous_points = np.concatenate([keypoints, np.ones((num_points, 1))], axis=1)
    warped_points = np.dot(homogeneous_points, np.transpose(H)).squeeze(axis=2)
    return warped_points[:, :2] / warped_points[:, 2:]


def keep_true_keypoints(points, H, shape):
    warped_points = warp_keypoints(points[:, [1, 0]], H)
    warped_points[:, [0, 1]] = warped_points[:, [1, 0]]
    mask = (warped_points[:, 0] >= 0) & (warped_points[:, 0] < shape[0]) &\
           (warped_points[:, 1] >= 0) & (warped_points[:, 1] < shape[1])
    return points[mask, :]


def filter_keypoints(points, shape):
    mask = (points[:, 0] >= 0) & (points[:, 0] < shape[0]) &\
           (points[:, 1] >= 0) & (points[:, 1] < shape[1])
    return points[mask, :]


def select_k_best(conf_pts,
                  max_count=300):
    sorted_pts = conf_pts[conf_pts[:, 2].argsort(), :2]
    start = min(max_count, conf_pts.shape[0])
    return sorted_pts[-start:, :]


def calc_repeatability_np(src_pts,
                          src_confs,
                          dst_conf_pts,
                          homography,
                          src_shape,
                          dst_shape):
    distance_thresh = 3

    filtered_warped_keypoints = keep_true_keypoints(dst_conf_pts, np.linalg.inv(homography), src_shape)

    true_warped_keypoints = warp_keypoints(src_pts[:, [1, 0]], homography)
    true_warped_keypoints = np.stack([true_warped_keypoints[:, 1], true_warped_keypoints[:, 0], src_confs], axis=-1)
    true_warped_keypoints = filter_keypoints(true_warped_keypoints, dst_shape)

    filtered_warped_keypoints = select_k_best(filtered_warped_keypoints)
    true_warped_keypoints = select_k_best(true_warped_keypoints)

    n1 = true_warped_keypoints.shape[0]
    n2 = filtered_warped_keypoints.shape[0]
    true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
    filtered_warped_keypoints = np.expand_dims(filtered_warped_keypoints, 0)
    norm = np.linalg.norm(true_warped_keypoints - filtered_warped_keypoints, ord=None, axis=2)
    count1 = 0
    count2 = 0
    if n2 != 0:
        min1 = np.min(norm, axis=1)
        count1 = np.sum(min1 <= distance_thresh)
    if n1 != 0:
        min2 = np.min(norm, axis=0)
        count2 = np.sum(min2 <= distance_thresh)
    if n1 + n2 > 0:
        repeatability = (count1 + count2) / (n1 + n2)
    else:
        repeatability = 0

    return n1, n2, repeatability


def batch_fn(batch, ctx):
    data_src = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    data_dst = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    label = split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
    return data_src, data_dst, label


def calc_detector_repeatability(test_data,
                                net,
                                ctx):
    tic = time.time()
    repeatabilities = []
    n1s = []
    n2s = []
    for batch in test_data:
        data_src_list, data_dst_list, labels_list = batch_fn(batch, ctx)
        outputs_src_list = [net(X) for X in data_src_list]
        outputs_dst_list = [net(X) for X in data_dst_list]
        for i in range(len(data_src_list)):
            homography = labels_list[i].asnumpy()

            data_src_i = data_src_list[i]
            data_dst_i = data_dst_list[i]

            src_shape = data_src_i.shape[2:]
            dst_shape = data_dst_i.shape[2:]

            src_pts, src_confs, src_desc_map = outputs_src_list[i]
            dst_pts, dst_confs, dst_desc_map = outputs_dst_list[i]

            # src_conf_pts = mx.nd.concat(src_pts[0], src_confs[0].reshape(shape=(-1, 1)), dim=1).asnumpy()
            src_pts_np = src_pts[0].asnumpy()
            src_confs_np = src_confs[0].asnumpy()
            dst_conf_pts = mx.nd.concat(dst_pts[0], dst_confs[0].reshape(shape=(-1, 1)), dim=1).asnumpy()

            n1, n2, repeatability = calc_repeatability_np(
                src_pts_np,
                src_confs_np,
                dst_conf_pts,
                homography,
                src_shape,
                dst_shape)
            n1s.append(n1)
            n2s.append(n2)
            repeatabilities.append(repeatability)

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

    ctx, batch_size = prepare_mx_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        dtype=args.dtype,
        net_extra_kwargs=ds_metainfo.test_net_extra_kwargs,
        load_ignore_extra=False,
        classes=args.num_classes,
        in_channels=args.in_channels,
        do_hybridize=False,
        ctx=ctx)

    test_data = get_val_data_source(
        ds_metainfo=ds_metainfo,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    calc_detector_repeatability(
        test_data=test_data,
        net=net,
        ctx=ctx)


if __name__ == "__main__":
    main()
