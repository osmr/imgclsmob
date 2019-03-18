import os
import argparse

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon Segmentation')

    parser.add_argument('--model', type=str, default='fcn', help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet50', help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascalaug', help='dataset name (default: pascal)')
    parser.add_argument('--dataset-dir', type=str, default='../imgclsmob_data/voc', help='dataset path')
    parser.add_argument('--workers', type=int, default=16, metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520, help='base image size')
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')

    parser.add_argument('--batch-size', type=int, default=1, metavar='N', help='input batch size for testing')

    parser.add_argument('--ngpus', type=int, default=len(mx.test_utils.list_gpus()), help='number of GPUs (default: 4)')

    # checking point
    parser.add_argument('--resume', type=str, default=None, help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default', help='set the checkpoint name')
    parser.add_argument('--model-zoo', type=str, default=None, help='evaluating on model zoo model')

    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.ngpus == 0:
        args.ctx = [mx.cpu(0)]
    else:
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    print(args)
    return args


def test(args):

    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    testset = get_segmentation_dataset(
        args.dataset,
        split='val',
        mode='testval',
        transform=input_transform,
        root=args.dataset_dir)
    test_data = gluon.data.DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        shuffle=False,
        last_batch='keep',
        batchify_fn=ms_batchify_fn,
        num_workers=args.workers)

    # create network
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(
            model=args.model,
            dataset=args.dataset,
            ctx=args.ctx,
            backbone=args.backbone,
            norm_layer=mx.gluon.nn.BatchNorm,
            norm_kwargs={},
            aux=False,
            base_size=args.base_size,
            crop_size=args.crop_size)
        # load pretrained weight
        assert args.resume is not None, '=> Please provide the checkpoint using --resume'
        if os.path.isfile(args.resume):
            model.load_parameters(
                filename=args.resume,
                ctx=args.ctx)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    evaluator = MultiEvalModel(
        module=model,
        nclass=testset.num_class,
        ctx_list=args.ctx)
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    for i, (data, dsts) in enumerate(test_data):
        predicts = [pred for pred in evaluator.parallel_forward(data)]
        targets = [target.as_in_context(predicts[0].context).expand_dims(0) for target in dsts]
        metric.update(targets, predicts)
        pixAcc, mIoU = metric.get()
        print('batch={}, pixAcc: {}, mIoU: {}'.format(i, pixAcc, mIoU))


if __name__ == "__main__":
    args = parse_args()
    print('Testing model: ', args.resume)
    test(args)
