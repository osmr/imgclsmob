"""
    Segmentation datasets (VOC2012/ADE20K/Cityscapes/COCO) routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_test_data_loader', 'validate1', 'get_metainfo']

from tqdm import tqdm
import torch.utils.data
import torchvision.transforms as transforms
from pytorch.datasets.voc_seg_dataset import VOCSegDataset
from pytorch.datasets.ade20k_seg_dataset import ADE20KSegDataset
from pytorch.datasets.cityscapes_seg_dataset import CityscapesSegDataset
from pytorch.datasets.coco_seg_dataset import CocoSegDataset
# import torchvision.datasets as datasets


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


def get_test_data_loader(dataset_name,
                         dataset_dir,
                         batch_size,
                         num_workers):
    mean_rgb = (0.485, 0.456, 0.406)
    std_rgb = (0.229, 0.224, 0.225)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb),
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

    val_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return val_loader


def validate1(accuracy_metrics,
              net,
              val_data,
              use_cuda):
    net.eval()
    for metric in accuracy_metrics:
        metric.reset()
    with torch.no_grad():
        for data, target in tqdm(val_data):
            if use_cuda:
                target = target.cuda(non_blocking=True)
            output = net(data)
            for metric in accuracy_metrics:
                metric.update(target, output)
    accuracy_info = [metric.get() for metric in accuracy_metrics]
    return accuracy_info
