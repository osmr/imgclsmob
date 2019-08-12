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
from pytorch.metrics.ret_metrics import PointDetectionMeanResidual


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
    metric = PointDetectionMeanResidual()
    metric.reset()
    with torch.no_grad():
        for data_src, data_dst, target in test_data:
            if use_cuda:
                data_src = data_src.cuda(non_blocking=True)
                data_dst = data_dst.cuda(non_blocking=True)
            src_pts, src_confs, src_desc_map = net(data_src)
            dst_pts, dst_confs, dst_desc_map = net(data_dst)
            src_shape = data_src.cpu().detach().numpy().shape[2:]
            dst_shape = data_dst.cpu().detach().numpy().shape[2:]
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
            metric.update(
                homography=target,
                src_pts=src_pts[0],
                dst_pts=dst_pts[0],
                src_confs=src_confs[0],
                dst_confs=dst_confs[0],
                src_img_size=src_shape,
                dst_img_size=dst_shape)

    logging.info("Average number of points in the first image: {}".format(np.mean(n1s)))
    logging.info("Average number of points in the second image: {}".format(np.mean(n2s)))
    logging.info("The repeatability: {:.4f}".format(np.mean(repeatabilities)))
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

    use_cuda, batch_size = prepare_pt_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        use_cuda=use_cuda,
        net_extra_kwargs=ds_metainfo.net_extra_kwargs,
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
