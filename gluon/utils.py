"""
    Main routines shared between training and evaluation scripts.
"""

__all__ = ['prepare_mx_context', 'get_initializer', 'prepare_model', 'calc_net_weight_count', 'validate',
           'report_accuracy', 'get_composite_metric', 'get_metric_name', 'get_loss']

import os
import re
import logging
import numpy as np
import mxnet as mx
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from .gluoncv2.model_provider import get_model
from .metrics.cls_metrics import Top1Error, TopKError
from .metrics.seg_metrics import PixelAccuracyMetric, MeanIoUMetric
from .metrics.det_metrics import CocoDetMApMetric, VOC07MApMetric, WiderfaceDetMetric
from .metrics.hpe_metrics import CocoHpeOksApMetric
from .losses import SegSoftmaxCrossEntropyLoss, MixSoftmaxCrossEntropyLoss


def prepare_mx_context(num_gpus,
                       batch_size):
    """
    Prepare MXNet context and correct batch size.

    Parameters
    ----------
    num_gpus : int
        Number of GPU.
    batch_size : int
        Batch size for each GPU.

    Returns
    -------
    Context
        MXNet context.
    int
        Batch size for all GPUs.
    """
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    batch_size *= max(1, num_gpus)
    return ctx, batch_size


def get_initializer(initializer_name):
    """
    Get initializer by name.

    Parameters
    ----------
    initializer_name : str
        Initializer name.

    Returns
    -------
    Initializer
        Initializer.
    """
    if initializer_name == "MSRAPrelu":
        return mx.init.MSRAPrelu()
    elif initializer_name == "Xavier":
        return mx.init.Xavier()
    elif initializer_name == "Xavier-gaussian-out-2":
        return mx.init.Xavier(
            rnd_type="gaussian",
            factor_type="out",
            magnitude=2)
    else:
        return None


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  dtype,
                  net_extra_kwargs=None,
                  load_ignore_extra=False,
                  tune_layers=None,
                  classes=None,
                  in_channels=None,
                  do_hybridize=True,
                  initializer=mx.init.MSRAPrelu(),
                  ctx=mx.cpu()):
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
    dtype : str
        Base data type for tensors.
    net_extra_kwargs : dict, default None
        Extra parameters for model.
    load_ignore_extra : bool, default False
        Whether to ignore extra layers in pretrained model.
    tune_layers : dict, default False
        Layers for tuning (all other will be frozen).
    classes : int, default None
        Number of classes.
    in_channels : int, default None
        Number of input channels.
    do_hybridize : bool, default True
        Whether to hybridize model.
    initializer : Initializer
        Initializer.
    ctx : Context, default CPU
        MXNet context.

    Returns
    -------
    HybridBlock
        Model.
    """
    kwargs = {"ctx": ctx,
              "pretrained": use_pretrained}
    if classes is not None:
        kwargs["classes"] = classes
    if in_channels is not None:
        kwargs["in_channels"] = in_channels
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))
        net.load_parameters(
            filename=pretrained_model_file_path,
            ctx=ctx,
            ignore_extra=load_ignore_extra)

    net.cast(dtype)

    if do_hybridize:
        net.hybridize(
            static_alloc=True,
            static_shape=True)

    if pretrained_model_file_path or use_pretrained:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize(initializer, ctx=ctx)
    else:
        net.initialize(initializer, ctx=ctx)

    if (tune_layers is not None) and tune_layers:
        tune_layers_pattern = re.compile(tune_layers)
        for k, v in net._collect_params_with_prefix().items():
            if tune_layers_pattern.match(k):
                logging.info("Fine-tune parameter: {}".format(k))
            else:
                v.grad_req = "null"
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize(initializer, ctx=ctx)

    return net


def calc_net_weight_count(net):
    """
    Calculate number of model trainable parameters.

    Parameters
    ----------
    net : HybridBlock
        Model.

    Returns
    -------
    int
        Number of parameters.
    """
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def validate(metric,
             net,
             val_data,
             batch_fn,
             data_source_needs_reset,
             dtype,
             ctx):
    """
    Core validation/testing routine.

    Parameters:
    ----------
    metric : EvalMetric
        Metric object instance.
    net : HybridBlock
        Model.
    val_data : DataLoader or ImageRecordIter
        Data loader or ImRec-iterator.
    batch_fn : func
        Function for splitting data after extraction from data loader.
    data_source_needs_reset : bool
        Whether to reset data (if test_data is ImageRecordIter).
    dtype : str
        Base data type for tensors.
    ctx : Context
        MXNet context.

    Returns
    -------
    EvalMetric
        Metric object instance.
    """
    if data_source_needs_reset:
        val_data.reset()
    metric.reset()
    for batch in val_data:
        data_list, labels_list = batch_fn(batch, ctx)
        outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
        metric.update(labels_list, outputs_list)
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
    if isinstance(metric, mx.metric.CompositeEvalMetric):
        msg = ", ".join([create_msg(name=m[0], value=m[1]) for m in zip(*metric_info)])
    elif isinstance(metric, mx.metric.EvalMetric):
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

    Returns
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
    elif metric_name == "VOC07MApMetric":
        return VOC07MApMetric(**metric_extra_kwargs)
    elif metric_name == "WiderfaceDetMetric":
        return WiderfaceDetMetric(**metric_extra_kwargs)
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
        metric = mx.metric.CompositeEvalMetric()
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
    if isinstance(metric, mx.metric.CompositeEvalMetric):
        return metric.metrics[index].name
    elif isinstance(metric, mx.metric.EvalMetric):
        assert (index == 0)
        return metric.name
    else:
        raise Exception("Wrong metric type: {}".format(type(metric)))


def get_loss(loss_name, loss_extra_kwargs):
    """
    Get loss by name.

    Parameters:
    ----------
    loss_name : str
        Loss name.
    loss_extra_kwargs : dict
        Loss extra parameters.

    Returns
    -------
    Loss
        Loss object instance.
    """
    if loss_name == "SoftmaxCrossEntropy":
        return SoftmaxCrossEntropyLoss(**loss_extra_kwargs)
    if loss_name == "SegSoftmaxCrossEntropy":
        return SegSoftmaxCrossEntropyLoss(**loss_extra_kwargs)
    if loss_name == "MixSoftmaxCrossEntropy":
        return MixSoftmaxCrossEntropyLoss(**loss_extra_kwargs)
    else:
        raise Exception("Wrong loss name: {}".format(loss_name))
