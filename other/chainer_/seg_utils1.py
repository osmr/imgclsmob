"""
    Segmentation datasets (VOC2012/ADE20K/Cityscapes/COCO) routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_test_dataset', 'get_metainfo', 'SegPredictor']

import numpy as np
import chainer
from chainer import Chain
from chainer_.datasets.voc_seg_dataset import VOCSegDataset
from chainer_.datasets.ade20k_seg_dataset import ADE20KSegDataset
from chainer_.datasets.cityscapes_seg_dataset import CityscapesSegDataset
from chainer_.datasets.coco_seg_dataset import CocoSegDataset


def add_dataset_parser_arguments(parser,
                                 dataset_name):
    if dataset_name == "VOC":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/voc',
            help='path to directory with Pascal VOC2012 dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=21,
            help='number of classes')
    elif dataset_name == "ADE20K":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/ade20k',
            help='path to directory with ADE20K dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=150,
            help='number of classes')
    elif dataset_name == "Cityscapes":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/cityscapes',
            help='path to directory with Cityscapes dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=19,
            help='number of classes')
    elif dataset_name == "COCO":
        parser.add_argument(
            '--data-dir',
            type=str,
            default='../imgclsmob_data/coco',
            help='path to directory with COCO dataset')
        parser.add_argument(
            '--num-classes',
            type=int,
            default=21,
            help='number of classes')
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')
    parser.add_argument(
        '--image-base-size',
        type=int,
        default=520,
        help='base image size')
    parser.add_argument(
        '--image-crop-size',
        type=int,
        default=480,
        help='crop image size')


class SegPredictor(Chain):

    def __init__(self,
                 base_model,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        super(SegPredictor, self).__init__()
        self.mean = np.array(mean, np.float32)[:, np.newaxis, np.newaxis]
        self.std = np.array(std, np.float32)[:, np.newaxis, np.newaxis]
        with self.init_scope():
            self.model = base_model

    def _preprocess(self, img):
        dtype = chainer.get_dtype(None)
        img = img.transpose(2, 0, 1)
        img = img.astype(dtype)
        img *= 1.0 / 255.0

        img -= self.mean
        img /= self.std
        return img

    def predict(self, imgs):
        imgs = self.xp.asarray([self._preprocess(img) for img in imgs])

        with chainer.using_config("train", False), chainer.function.no_backprop_mode():
            imgs = chainer.Variable(imgs)
            predictions = self.model(imgs)

        output = chainer.backends.cuda.to_cpu(predictions.array)
        # output = np.argmax(output, axis=1).astype(np.int32)

        return output


def get_metainfo(dataset_name):
    if dataset_name == "VOC":
        return {
            "vague_idx": VOCSegDataset.vague_idx,
            "use_vague": VOCSegDataset.use_vague,
            "background_idx": VOCSegDataset.background_idx,
            "ignore_bg": VOCSegDataset.ignore_bg}
    elif dataset_name == "ADE20K":
        return {
            "vague_idx": ADE20KSegDataset.vague_idx,
            "use_vague": ADE20KSegDataset.use_vague,
            "background_idx": ADE20KSegDataset.background_idx,
            "ignore_bg": ADE20KSegDataset.ignore_bg}
    elif dataset_name == "Cityscapes":
        return {
            "vague_idx": CityscapesSegDataset.vague_idx,
            "use_vague": CityscapesSegDataset.use_vague,
            "background_idx": CityscapesSegDataset.background_idx,
            "ignore_bg": CityscapesSegDataset.ignore_bg}
    elif dataset_name == "COCO":
        return {
            "vague_idx": CocoSegDataset.vague_idx,
            "use_vague": CocoSegDataset.use_vague,
            "background_idx": CocoSegDataset.background_idx,
            "ignore_bg": CocoSegDataset.ignore_bg}
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))


def get_test_dataset(dataset_name,
                     dataset_dir):

    if dataset_name == "VOC":
        dataset_class = VOCSegDataset
    elif dataset_name == "ADE20K":
        dataset_class = ADE20KSegDataset
    elif dataset_name == "Cityscapes":
        dataset_class = CityscapesSegDataset
    elif dataset_name == "COCO":
        dataset_class = CocoSegDataset
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))

    dataset = dataset_class(
        root=dataset_dir,
        mode="test",
        transform=None)

    return dataset
