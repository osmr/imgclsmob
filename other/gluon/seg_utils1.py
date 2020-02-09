"""
    Segmentation datasets (VOC2012/ADE20K/Cityscapes/COCO) routines.
"""

__all__ = ['add_dataset_parser_arguments', 'batch_fn', 'get_test_data_source', 'get_num_training_samples', 'validate1',
           'get_metainfo']

from tqdm import tqdm
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from gluon.datasets.voc_seg_dataset import VOCSegDataset
from gluon.datasets.ade20k_seg_dataset import ADE20KSegDataset
from gluon.datasets.cityscapes_seg_dataset import CityscapesSegDataset
from gluon.datasets.coco_seg_dataset import CocoSegDataset
# from gluoncv.data.mscoco.segmentation import COCOSegmentation


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


def batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
    return data, label


def get_num_training_samples(dataset_name):
    if dataset_name == "ADE20K":
        return None
    else:
        raise Exception('Unrecognized dataset: {}'.format(dataset_name))


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


def get_test_data_source(dataset_name,
                         dataset_dir,
                         batch_size,
                         num_workers):
    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb)
    ])

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
        transform=transform_val)

    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)


def validate1(accuracy_metric,
              net,
              val_data,
              batch_fn,
              data_source_needs_reset,
              dtype,
              ctx):
    if data_source_needs_reset:
        val_data.reset()
    accuracy_metric.reset()
    for batch in tqdm(val_data):
        data_list, labels_list = batch_fn(batch, ctx)
        outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
        accuracy_metric.update(labels_list, outputs_list)
    accuracy_info = accuracy_metric.get()
    return accuracy_info
