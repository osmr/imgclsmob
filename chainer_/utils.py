import logging
import os
from chainer import cuda
from chainer import using_config, Variable
from chainer.function import no_backprop_mode
from chainer.backends.cuda import to_cpu
from chainer.serializers import load_npz
from .chainercv2.model_provider import get_model
from .metric import EvalMetric, CompositeEvalMetric
from .cls_metrics import Top1Error, TopKError
from .seg_metrics import PixelAccuracyMetric, MeanIoUMetric


def prepare_ch_context(num_gpus):
    use_gpus = (num_gpus > 0)
    if use_gpus:
        cuda.get_device(0).use()
    return use_gpus


class Predictor(object):
    """
    Model predictor with preprocessing.

    Parameters
    ----------
    model : Chain
        Base model.
    transform : callable, optional
        A function that transforms the image.
    """
    def __init__(self,
                 model,
                 transform=None):
        super(Predictor, self).__init__()
        self.model = model
        self.transform = transform

    def do_transform(self, img):
        if self.transform is not None:
            return self.transform(img)
        else:
            return img

    def __call__(self, imgs):
        imgs = self.model.xp.asarray([self.do_transform(img) for img in imgs])

        with using_config("train", False), no_backprop_mode():
            imgs = Variable(imgs)
            predictions = self.model(imgs)

        output = to_cpu(predictions.array)
        return output


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  use_gpus=False,
                  net_extra_kwargs=None,
                  num_classes=None,
                  in_channels=None):
    kwargs = {'pretrained': use_pretrained}
    if num_classes is not None:
        kwargs["classes"] = num_classes
    if in_channels is not None:
        kwargs["in_channels"] = in_channels
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        load_npz(
            file=pretrained_model_file_path,
            obj=net)

    if use_gpus:
        net.to_gpu()

    return net


def report_accuracy(metric,
                    extended_log=False):
    metric_info = metric.get()
    if extended_log:
        msg_pattern = "{name}={value:.4f} ({value})"
    else:
        msg_pattern = "{name}={value:.4f}"
    if isinstance(metric, CompositeEvalMetric):
        msg = ""
        for m in zip(*metric_info):
            if msg != "":
                msg += ", "
            msg += msg_pattern.format(name=m[0], value=m[1])
    elif isinstance(metric, EvalMetric):
        msg = msg_pattern.format(name=metric_info[0], value=metric_info[1])
    else:
        raise Exception("Wrong metric type: {}".format(type(metric)))
    return msg


def get_metric(metric_name, metric_extra_kwargs):
    if metric_name == "Top1Error":
        return Top1Error(**metric_extra_kwargs)
    elif metric_name == "TopKError":
        return TopKError(**metric_extra_kwargs)
    elif metric_name == "PixelAccuracyMetric":
        return PixelAccuracyMetric(**metric_extra_kwargs)
    elif metric_name == "MeanIoUMetric":
        return MeanIoUMetric(**metric_extra_kwargs)
    else:
        raise Exception("Wrong metric name: {}".format(metric_name))


def get_composite_metric(metric_names, metric_extra_kwargs):
    if len(metric_names) == 1:
        metric = get_metric(metric_names[0], metric_extra_kwargs[0])
    else:
        metric = CompositeEvalMetric()
        for name, extra_kwargs in zip(metric_names, metric_extra_kwargs):
            metric.add(get_metric(name, extra_kwargs))
    return metric


def get_metric_name(metric, index):
    if isinstance(metric, CompositeEvalMetric):
        return metric.metrics[index].name
    elif isinstance(metric, EvalMetric):
        assert (index == 0)
        return metric.name
    else:
        raise Exception("Wrong metric type: {}".format(type(metric)))
