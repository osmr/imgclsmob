__all__ = ['prepare_model']

import os
import logging
import tensorflow as tf
from .tf2cv.model_provider import get_model
from .metrics.metric import EvalMetric, CompositeEvalMetric
from .metrics.cls_metrics import Top1Error, TopKError
from .metrics.seg_metrics import PixelAccuracyMetric, MeanIoUMetric
from .metrics.det_metrics import CocoDetMApMetric
from .metrics.hpe_metrics import CocoHpeOksApMetric


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  net_extra_kwargs=None,
                  load_ignore_extra=False,
                  batch_size=None,
                  use_cuda=True):
    kwargs = {"pretrained": use_pretrained}
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)
    # kwargs["input_shape"] = (1, 224, 224, 3)

    # my_devices = tf.config.experimental.list_physical_devices(device_type="CPU")
    # tf.config.experimental.set_visible_devices(devices=my_devices, device_type="CPU")
    # tf.debugging.set_log_device_placement(True)

    if not use_cuda:
        with tf.device("/cpu:0"):
            net = get_model(model_name, **kwargs)
            # input_shape = ((1, 3, net.in_size[0], net.in_size[1]) if
            #                net.data_format == "channels_first" else (1, net.in_size[0], net.in_size[1], 3))
            # net.build(input_shape=input_shape)
    else:
        net = get_model(model_name, **kwargs)
        # input_shape = ((batch_size, 3, net.in_size[0], net.in_size[1]) if
        #                net.data_format == "channels_first" else (batch_size, net.in_size[0], net.in_size[1], 3))
        # net.build(input_shape=input_shape)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info("Loading model: {}".format(pretrained_model_file_path))

        input_shape = ((batch_size, 3, net.in_size[0], net.in_size[1]) if
                       net.data_format == "channels_first" else (batch_size, net.in_size[0], net.in_size[1], 3))
        net.build(input_shape=input_shape)
        if load_ignore_extra:
            net.load_weights(
                filepath=pretrained_model_file_path,
                by_name=True,
                skip_mismatch=True)
        else:
            net.load_weights(
                filepath=pretrained_model_file_path)

    return net


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
