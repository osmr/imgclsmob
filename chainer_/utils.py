import logging
import os
import numpy as np

from chainer import iterators
from chainer.serializers import load_npz

from chainercv.datasets import directory_parsing_label_names
from chainercv.datasets import DirectoryParsingLabelDataset

from .model_utils import get_model


def get_data_iterator(data_dir,
                      batch_size,
                      num_workers,
                      num_classes):
    val_dir_path = os.path.join(data_dir, 'val')
    val_dataset = DirectoryParsingLabelDataset(val_dir_path)
    val_dataset_len = len(val_dataset)
    assert(len(directory_parsing_label_names(val_dir_path)) == num_classes)
    val_iterator = iterators.MultiprocessIterator(
        dataset=val_dataset,
        batch_size=batch_size,
        repeat=False,
        shuffle=False,
        n_processes=num_workers,
        shared_mem=300000000)
    return val_iterator, val_dataset_len


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
        load_npz(
            file=pretrained_model_file_path,
            obj=net)

    return net

