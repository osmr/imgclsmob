import argparse
import time
import logging
import os
import sys
import numpy as np

import mxnet as mx
from mxnet import gluon
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import LRScheduler
from gluoncv import utils as gutils

from gluon.models.resnet import *
from gluon.models.preresnet import *
from gluon.models.squeezenet import *
from gluon.models.squeezenext import *
from gluon.models.mobilenet import *
from gluon.models.shufflenet import *
from gluon.models.menet import *
from gluon.models.nasnet import *
from gluon.models.darknet import *


def parse_args():
    parser = argparse.ArgumentParser(description='Convert models (Gluon)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--src-model',
        type=str,
        required=True,
        help='source model name')
    parser.add_argument(
        '--dst-model',
        type=str,
        required=True,
        help='destination model name')
    parser.add_argument(
        '--src-params',
        type=str,
        default='',
        help='source model parameter file path')
    parser.add_argument(
        '--dst-params',
        type=str,
        default='',
        help='destination model parameter file path')
    args = parser.parse_args()
    return args


def _get_model(name, **kwargs):
    models = {
        'resnet18': resnet18,
        'preresnet18': preresnet18,

        'squeezenet1_0': squeezenet1_0,
        'squeezenet1_1': squeezenet1_1,
        'squeezeresnet1_0': squeezeresnet1_0,
        'squeezeresnet1_1': squeezeresnet1_1,

        'sqnxt23_1_0': sqnxt23_1_0,
        'sqnxt23_1_5': sqnxt23_1_5,
        'sqnxt23_2_0': sqnxt23_2_0,
        'sqnxt23v5_1_0': sqnxt23v5_1_0,
        'sqnxt23v5_1_5': sqnxt23v5_1_5,
        'sqnxt23v5_2_0': sqnxt23v5_2_0,

        'nasnet_a_mobile': nasnet_a_mobile,

        'darknet_ref': darknet_ref,
        'darknet_tiny': darknet_tiny,
        'darknet19': darknet19,

        'mobilenet1_0': mobilenet1_0,
        'mobilenet0_75': mobilenet0_75,
        'mobilenet0_5': mobilenet0_5,
        'mobilenet0_25': mobilenet0_25,
        'fd_mobilenet1_0': fd_mobilenet1_0,
        'fd_mobilenet0_75': fd_mobilenet0_75,
        'fd_mobilenet0_5': fd_mobilenet0_5,
        'fd_mobilenet0_25': fd_mobilenet0_25,
        'shufflenet1_0_g1': shufflenet1_0_g1,
        'shufflenet1_0_g2': shufflenet1_0_g2,
        'shufflenet1_0_g3': shufflenet1_0_g3,
        'shufflenet1_0_g4': shufflenet1_0_g4,
        'shufflenet1_0_g8': shufflenet1_0_g8,
        'shufflenet0_5_g1': shufflenet0_5_g1,
        'shufflenet0_5_g3': shufflenet0_5_g3,
        'shufflenet0_25_g1': shufflenet0_25_g1,
        'shufflenet0_25_g3': shufflenet0_25_g3,
        'menet108_8x1_g3': menet108_8x1_g3,
        'menet128_8x1_g4': menet128_8x1_g4,
        'menet160_8x1_g8': menet160_8x1_g8,
        'menet228_12x1_g3': menet228_12x1_g3,
        'menet256_12x1_g4': menet256_12x1_g4,
        'menet348_12x1_g3': menet348_12x1_g3,
        'menet352_12x1_g8': menet352_12x1_g8,
        'menet456_24x1_g3': menet456_24x1_g3,
    }
    try:
        net = get_model(name, **kwargs)
        return net
    except ValueError as e:
        upstream_supported = str(e)
    name = name.lower()
    if name not in models:
        raise ValueError('%s\n\t%s' % (upstream_supported, '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net


def prepare_model(model_name,
                  classes,
                  use_pretrained,
                  pretrained_model_file_path,
                  batch_norm,
                  use_se,
                  last_gamma,
                  dtype,
                  ctx):
    kwargs = {'ctx': ctx,
              'pretrained': use_pretrained,
              'classes': classes}

    if model_name.startswith('vgg'):
        kwargs['batch_norm'] = batch_norm
    elif model_name.startswith('resnext'):
        kwargs['use_se'] = use_se

    if last_gamma:
        kwargs['last_gamma'] = True

    net = _get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        net.load_parameters(
            filename=pretrained_model_file_path,
            ctx=ctx)

    net.cast(dtype)

    net.hybridize(
        static_alloc=True,
        static_shape=True)

    if pretrained_model_file_path or use_pretrained:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    else:
        net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    return net


def main():
    args = parse_args()
    ctx = mx.cpu()
    num_classes = 1000

    src_net = prepare_model(
        model_name=args.src_model,
        classes=num_classes,
        use_pretrained=False,
        pretrained_model_file_path=args.src_params,
        batch_norm=False,
        use_se=False,
        last_gamma=False,
        dtype=np.float32,
        ctx=ctx)

    dst_net = prepare_model(
        model_name=args.dst_model,
        classes=num_classes,
        use_pretrained=False,
        pretrained_model_file_path="",
        batch_norm=False,
        use_se=False,
        last_gamma=False,
        dtype=np.float32,
        ctx=ctx)

    src_params = src_net._collect_params_with_prefix()
    dst_params = dst_net._collect_params_with_prefix()
    src_param_keys = list(src_params.keys())
    dst_param_keys = list(dst_params.keys())
    for i, dst_key in enumerate(dst_param_keys):
        #dst_params[dst_key]._load_init(src_params[src_param_keys[i+4]]._data[0], ctx)
        dst_params[dst_key]._load_init(src_params[src_param_keys[i]]._data[0], ctx)
    dst_net.save_parameters(args.dst_params)


if __name__ == '__main__':
    main()

