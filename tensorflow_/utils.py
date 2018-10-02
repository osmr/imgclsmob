import logging
import os

from tensorpack.tfutils import get_model_loader

from .model_provider import get_model


def prepare_tf_context(num_gpus,
                       batch_size):
    batch_size *= max(1, num_gpus)
    return batch_size


def prepare_model(model_name,
                  pretrained_model_file_path):

    net = get_model(model_name)

    inputs_desc = None
    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        inputs_desc = get_model_loader(pretrained_model_file_path)

    return net, inputs_desc
