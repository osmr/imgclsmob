import logging
import os
from chainer.serializers import load_npz
from .chainercv2.model_provider import get_model


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  use_gpus=False,
                  net_extra_kwargs=None,
                  num_classes=None,
                  in_channels=None):
    kwargs = {'pretrained': use_pretrained}
    if num_classes is not None:
        kwargs["num_classes"] = num_classes
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
