import logging
import os
from chainer.serializers import load_npz
from .chainercv2.model_provider import get_model


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  net_extra_kwargs=None,
                  num_gpus=0):
    kwargs = {'pretrained': use_pretrained}
    if net_extra_kwargs is not None:
        kwargs.update(net_extra_kwargs)

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        load_npz(
            file=pretrained_model_file_path,
            obj=net)

    if num_gpus > 0:
        net.to_gpu()

    return net
