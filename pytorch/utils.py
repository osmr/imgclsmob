import math
import logging
import os
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .pytorchcv.model_provider import get_model


def prepare_pt_context(num_gpus,
                       batch_size):
    use_cuda = (num_gpus > 0)
    batch_size *= max(1, num_gpus)
    return use_cuda, batch_size


def get_data_loader(data_dir,
                    batch_size,
                    num_workers,
                    input_image_size=224,
                    resize_inv_factor=0.875):
    assert (resize_inv_factor > 0.0)
    resize_value = int(math.ceil(float(input_image_size) / resize_inv_factor))

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    jitter_param = 0.4

    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            root=os.path.join(data_dir, 'train'),
            transform=transforms.Compose([
                transforms.RandomResizedCrop(input_image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=jitter_param,
                    contrast=jitter_param,
                    saturation=jitter_param),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        dataset=datasets.ImageFolder(
            root=os.path.join(data_dir, 'val'),
            transform=transforms.Compose([
                transforms.Resize(resize_value),
                transforms.CenterCrop(input_image_size),
                transforms.ToTensor(),
                normalize,
            ])),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, val_loader


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  use_cuda,
                  use_data_parallel=True,
                  ignore_extra=False,
                  remap_to_cpu=False):
    kwargs = {'pretrained': use_pretrained}

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        checkpoint = torch.load(
            pretrained_model_file_path,
            map_location=(None if use_cuda and not remap_to_cpu else 'cpu'))
        if (type(checkpoint) == dict) and ('state_dict' in checkpoint):
            checkpoint = checkpoint['state_dict']

        if ignore_extra:
            pretrained_state = checkpoint
            model_dict = net.state_dict()
            pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
            net.load_state_dict(pretrained_state)
        else:
            net.load_state_dict(checkpoint)

    if use_data_parallel and use_cuda:
        net = torch.nn.DataParallel(net)

    if use_cuda:
        net = net.cuda()

    return net


def calc_net_weight_count(net):
    net.train()
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def validate(acc_top1,
             acc_top5,
             net,
             val_data,
             use_cuda):
    net.eval()
    acc_top1.reset()
    acc_top5.reset()
    with torch.no_grad():
        for data, target in val_data:
            if use_cuda:
                target = target.cuda(non_blocking=True)
            output = net(data)
            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            acc_top1.update(prec1[0], data.size(0))
            acc_top5.update(prec5[0], data.size(0))
    top1 = acc_top1.avg.item()
    top5 = acc_top5.avg.item()
    return 1.0 - top1, 1.0 - top5
