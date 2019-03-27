import logging
import os
import numpy as np

import torch.utils.data

from .pytorchcv.model_provider import get_model


def prepare_pt_context(num_gpus,
                       batch_size):
    use_cuda = (num_gpus > 0)
    batch_size *= max(1, num_gpus)
    return use_cuda, batch_size


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  use_cuda,
                  use_data_parallel=True,
                  net_extra_kwargs=None,
                  ignore_extra=False,
                  remap_to_cpu=False,
                  remove_module=False):
    kwargs = {'pretrained': use_pretrained}
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

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
            if remove_module:
                net_tmp = torch.nn.DataParallel(net)
                net_tmp.load_state_dict(checkpoint)
                net.load_state_dict(net_tmp.module.cpu().state_dict())
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


def validate1(accuracy_metric,
              net,
              val_data,
              use_cuda):
    net.eval()
    accuracy_metric.reset()
    with torch.no_grad():
        for data, target in val_data:
            if use_cuda:
                target = target.cuda(non_blocking=True)
            output = net(data)
            accuracy_value = accuracy(output, target)
            accuracy_metric.update(accuracy_value[0], data.size(0))
    accuracy_value = accuracy_metric.avg.item()
    return 1.0 - accuracy_value
