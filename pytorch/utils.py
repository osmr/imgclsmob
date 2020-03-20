"""
    Main routines shared between training and evaluation scripts.
"""

import logging
import os
import numpy as np
import torch.utils.data
from .pytorchcv.model_provider import get_model
from .metrics.metric import EvalMetric, CompositeEvalMetric
from .metrics.cls_metrics import Top1Error, TopKError
from .metrics.seg_metrics import PixelAccuracyMetric, MeanIoUMetric
from .metrics.det_metrics import CocoDetMApMetric
from .metrics.hpe_metrics import CocoHpeOksApMetric


def prepare_pt_context(num_gpus,
                       batch_size):
    """
    Correct batch size.

    Parameters
    ----------
    num_gpus : int
        Number of GPU.
    batch_size : int
        Batch size for each GPU.

    Returns
    -------
    bool
        Whether to use CUDA.
    int
        Batch size for all GPUs.
    """
    use_cuda = (num_gpus > 0)
    batch_size *= max(1, num_gpus)
    return use_cuda, batch_size


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  use_cuda,
                  use_data_parallel=True,
                  net_extra_kwargs=None,
                  load_ignore_extra=False,
                  num_classes=None,
                  in_channels=None,
                  remap_to_cpu=False,
                  remove_module=False):
    """
    Create and initialize model by name.

    Parameters
    ----------
    model_name : str
        Model name.
    use_pretrained : bool
        Whether to use pretrained weights.
    pretrained_model_file_path : str
        Path to file with pretrained weights.
    use_cuda : bool
        Whether to use CUDA.
    use_data_parallel : bool, default True
        Whether to use parallelization.
    net_extra_kwargs : dict, default None
        Extra parameters for model.
    load_ignore_extra : bool, default False
        Whether to ignore extra layers in pretrained model.
    num_classes : int, default None
        Number of classes.
    in_channels : int, default None
        Number of input channels.
    remap_to_cpu : bool, default False
        Whether to remape model to CPU during loading.
    remove_module : bool, default False
        Whether to remove module from loaded model.

    Returns
    -------
    Module
        Model.
    """
    kwargs = {"pretrained": use_pretrained}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
    if in_channels is not None:
        kwargs["in_channels"] = in_channels
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))
        checkpoint = torch.load(
            pretrained_model_file_path,
            map_location=(None if use_cuda and not remap_to_cpu else "cpu"))
        if (type(checkpoint) == dict) and ("state_dict" in checkpoint):
            checkpoint = checkpoint["state_dict"]

        if load_ignore_extra:
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
    """
    Calculate number of model trainable parameters.

    Parameters
    ----------
    net : Module
        Model.

    Returns
    -------
    int
        Number of parameters.
    """
    net.train()
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def validate(metric,
             net,
             val_data,
             use_cuda):
    """
    Core validation/testing routine.

    Parameters:
    ----------
    metric : EvalMetric
        Metric object instance.
    net : Module
        Model.
    val_data : DataLoader
        Data loader.
    use_cuda : bool
        Whether to use CUDA.

    Returns
    -------
    EvalMetric
        Metric object instance.
    """
    net.eval()
    metric.reset()
    with torch.no_grad():
        for data, target in val_data:
            if use_cuda:
                target = target.cuda(non_blocking=True)
            output = net(data)
            metric.update(target, output)
    return metric


def report_accuracy(metric,
                    extended_log=False):
    """
    Make report string for composite metric.

    Parameters:
    ----------
    metric : EvalMetric
        Metric object instance.
    extended_log : bool, default False
        Whether to log more precise accuracy values.

    Returns
    -------
    str
        Report string.
    """
    def create_msg(name, value):
        if type(value) in [list, tuple]:
            if extended_log:
                return "{}={} ({})".format("{}", "/".join(["{:.4f}"] * len(value)), "/".join(["{}"] * len(value))).\
                    format(name, *(value + value))
            else:
                return "{}={}".format("{}", "/".join(["{:.4f}"] * len(value))).format(name, *value)
        else:
            if extended_log:
                return "{name}={value:.4f} ({value})".format(name=name, value=value)
            else:
                return "{name}={value:.4f}".format(name=name, value=value)

    metric_info = metric.get()
    if isinstance(metric, CompositeEvalMetric):
        msg = ", ".join([create_msg(name=m[0], value=m[1]) for m in zip(*metric_info)])
    elif isinstance(metric, EvalMetric):
        msg = create_msg(name=metric_info[0], value=metric_info[1])
    else:
        raise Exception("Wrong metric type: {}".format(type(metric)))
    return msg


def get_metric(metric_name, metric_extra_kwargs):
    """
    Get metric by name.

    Parameters:
    ----------
    metric_name : str
        Metric name.
    metric_extra_kwargs : dict
        Metric extra parameters.

    EvalMetric
    -------
    EvalMetric
        Metric object instance.
    """
    if metric_name == "Top1Error":
        return Top1Error(**metric_extra_kwargs)
    elif metric_name == "TopKError":
        return TopKError(**metric_extra_kwargs)
    elif metric_name == "PixelAccuracyMetric":
        return PixelAccuracyMetric(**metric_extra_kwargs)
    elif metric_name == "MeanIoUMetric":
        return MeanIoUMetric(**metric_extra_kwargs)
    elif metric_name == "CocoDetMApMetric":
        return CocoDetMApMetric(**metric_extra_kwargs)
    elif metric_name == "CocoHpeOksApMetric":
        return CocoHpeOksApMetric(**metric_extra_kwargs)
    else:
        raise Exception("Wrong metric name: {}".format(metric_name))


def get_composite_metric(metric_names, metric_extra_kwargs):
    """
    Get composite metric by list of metric names.

    Parameters:
    ----------
    metric_names : list of str
        Metric name list.
    metric_extra_kwargs : list of dict
        Metric extra parameters list.

    Returns
    -------
    CompositeEvalMetric
        Metric object instance.
    """
    if len(metric_names) == 1:
        metric = get_metric(metric_names[0], metric_extra_kwargs[0])
    else:
        metric = CompositeEvalMetric()
        for name, extra_kwargs in zip(metric_names, metric_extra_kwargs):
            metric.add(get_metric(name, extra_kwargs))
    return metric


def get_metric_name(metric, index):
    """
    Get metric name by index in the composite metric.

    Parameters:
    ----------
    metric : CompositeEvalMetric or EvalMetric
        Metric object instance.
    index : int
        Index.

    Returns
    -------
    str
        Metric name.
    """
    if isinstance(metric, CompositeEvalMetric):
        return metric.metrics[index].name
    elif isinstance(metric, EvalMetric):
        assert (index == 0)
        return metric.name
    else:
        raise Exception("Wrong metric type: {}".format(type(metric)))
