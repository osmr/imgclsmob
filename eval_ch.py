import os
import argparse
import numpy as np

from chainer import iterators
from chainer import cuda
import chainer.functions as F

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset
from chainercv.utils import apply_to_iterator
from chainercv.utils import ProgressHook

from common.logger_utils import initialize_logging
from chainer_.model_utils import get_model
from chainer_.imagenet_predictor import ImagenetPredictor
from chainer_.top_k_accuracy import top_k_accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a model for image classification (Chainer)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/imagenet',
        help='training and validation pictures to use.')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='type of model to use. see vision_model for options.')
    parser.add_argument(
        '--use-pretrained',
        action='store_true',
        help='enable using pretrained model from gluon.')
    parser.add_argument(
        '--resume',
        type=str,
        default='',
        help='resume from previously saved parameters if not None')

    parser.add_argument(
        '--gpu-num',
        type=int,
        default=0,
        help='number of GPU to use.')
    parser.add_argument(
        '-j',
        '--num-data-workers',
        dest='num_workers',
        default=4,
        type=int,
        help='number of preprocessing workers')

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='training batch size per device (CPU/GPU).')

    parser.add_argument(
        '--save-dir',
        type=str,
        default='',
        help='directory of saved models and log-files')
    parser.add_argument(
        '--logging-file-name',
        type=str,
        default='train.log',
        help='filename of training log')

    parser.add_argument(
        '--log-packages',
        type=str,
        default='chainer, chainercv',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='cupy-cuda92, chainer, chainercv',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def get_data_iterator(data_dir,
                      batch_size,
                      num_workers,
                      num_classes):
    val_dir_path = os.path.join(data_dir, 'val')
    val_dataset = DirectoryParsingLabelDataset(val_dir_path)
    val_dataset_len = len(val_dataset)
    assert(len(directory_parsing_label_names(val_dir_path)) == num_classes)
    val_iterator = iterators.MultiprocessIterator(
        dataset=val_dataset,
        batch_size=batch_size,
        repeat=False,
        shuffle=False,
        n_processes=num_workers,
        shared_mem=300000000)
    return val_iterator, val_dataset_len


def main():
    args = parse_args()

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    net = get_model(args.model)
    predictor = ImagenetPredictor(base_model=net)

    num_classes = 1000
    val_iterator, val_dataset_len = get_data_iterator(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=num_classes)
    print("-->3")

    if args.gpu_num >= 0:
        cuda.get_device(args.gpu_num).use()
        predictor.to_gpu()

    print('Model has been prepared. Evaluation starts.')
    in_values, out_values, rest_values = apply_to_iterator(
        predictor.predict,
        val_iterator,
        hook=ProgressHook(val_dataset_len))
    del in_values

    pred_probs, = out_values
    gt_labels, = rest_values

    y = np.array(list(pred_probs))
    t = np.array(list(gt_labels))

    # print("type(y)={}".format(type(y)))
    # print("y.shape={}".format(y.shape))
    # print("t.shape={}".format(t.shape))

    top1_acc = F.accuracy(
        y=y,
        t=t).data
    top5_acc = top_k_accuracy(
        y=y,
        t=t,
        k=5).data
    print('Top 1 Error {}'.format(1. - top1_acc))
    print('Top 5 Error {}'.format(1. - top5_acc))


if __name__ == '__main__':
    main()

