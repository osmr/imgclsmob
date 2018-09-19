import logging
import os
import numpy as np

import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .model_provider import get_model


def prepare_model(model_name,
                  classes,
                  use_pretrained,
                  pretrained_model_file_path):
    kwargs = {'pretrained': use_pretrained,
              'classes': classes}

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        net.load_weights(filepath=pretrained_model_file_path)
        #from keras.models import load_model
        #net = load_model(pretrained_model_file_path)


    return net
