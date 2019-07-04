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


def select_k_best(points, k):
    sorted_prob = points[points[:, 2].argsort(), :2]
    start = min(k, points.shape[0])
    return sorted_prob[-start:, :]


def batch_fn(batch, ctx):
    data_src = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    data_dst = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    label = split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
    return data_src, data_dst, label


def calc_detector_repeatability(test_data,
                                net,
                                ctx):
    tic = time.time()
    keep_k_points = 300
    distance_thresh = 3
    repeatability = []
    N1s = []
    N2s = []
    for batch in test_data:
        data_src_list, data_dst_list, labels_list = batch_fn(batch, ctx)
        outputs_src_list = [net(X) for X in data_src_list]
        outputs_dst_list = [net(X) for X in data_dst_list]
        for i in range(len(data_src_list)):
            data_src_i = data_src_list[i]
            data_dst_i = data_dst_list[i]
            H = labels_list[i].asnumpy()

            shape = data_dst_i.shape[2:]

            src_pts, src_confs, src_desc_map = outputs_src_list[i]
            dst_pts, dst_confs, dst_desc_map = outputs_dst_list[i]

            keypoints = src_pts[0].asnumpy()
            prob = src_confs[0].asnumpy()
            warped_keypoints0 = mx.nd.concat(dst_pts[0], dst_confs[0].reshape(shape=(-1, 1)), dim=1).asnumpy()
            warped_keypoints = keep_true_keypoints(warped_keypoints0, np.linalg.inv(H), data_src_i.shape[2:])

            true_warped_keypoints = warp_keypoints(keypoints[:, [1, 0]], H)
            true_warped_keypoints = np.stack([true_warped_keypoints[:, 1], true_warped_keypoints[:, 0], prob], axis=-1)
            true_warped_keypoints = filter_keypoints(true_warped_keypoints, shape)

            warped_keypoints = select_k_best(warped_keypoints, keep_k_points)
            true_warped_keypoints = select_k_best(true_warped_keypoints, keep_k_points)

            N1 = true_warped_keypoints.shape[0]
            N2 = warped_keypoints.shape[0]
            N1s.append(N1)
            N2s.append(N2)
            true_warped_keypoints = np.expand_dims(true_warped_keypoints, 1)
            warped_keypoints = np.expand_dims(warped_keypoints, 0)
            norm = np.linalg.norm(true_warped_keypoints - warped_keypoints, ord=None, axis=2)
            count1 = 0
            count2 = 0
            if N2 != 0:
                min1 = np.min(norm, axis=1)
                count1 = np.sum(min1 <= distance_thresh)
            if N1 != 0:
                min2 = np.min(norm, axis=0)
                count2 = np.sum(min2 <= distance_thresh)
            if N1 + N2 > 0:
                repeatability.append((count1 + count2) / (N1 + N2))

    logging.info("Average number of points in the first image: {}".format(np.mean(N1s)))
    logging.info("Average number of points in the second image: {}".format(np.mean(N2s)))
    logging.info("The repeatability: {:.4f}".format(np.mean(repeatability)))
    logging.info("Time cost: {:.4f} sec".format(time.time() - tic))


def main():
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
        net_extra_kwargs=ds_metainfo.net_extra_kwargs,
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
