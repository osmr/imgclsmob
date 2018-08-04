import argparse
import time
import logging
import os
import sys
import numpy as np

import mxnet as mx
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Convert models (from Gluon to PyTorch)',
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

    parser.add_argument(
        '--save-dir',
        type=str,
        default='',
        help='directory of saved models and log-files')
    parser.add_argument(
        '--logging-file-name',
        type=str,
        default='train.log',
        help='filename of training log')
    args = parser.parse_args()
    return args


def prepare_logger(log_dir_path,
                   logging_file_name):
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_exist = False
    if log_dir_path is not None and log_dir_path:
        log_file_path = os.path.join(log_dir_path, logging_file_name)
        if not os.path.exists(log_dir_path):
            os.makedirs(log_dir_path)
            log_file_exist = False
        else:
            log_file_exist = (os.path.exists(log_file_path) and os.path.getsize(log_file_path) > 0)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        if log_file_exist:
            logging.info('--------------------------------')
    return logger, log_file_exist


def _get_model_gl(name, **kwargs):
    from gluoncv.model_zoo import get_model
    import gluon.models.resnet as gl_resnet
    import gluon.models.preresnet as gl_preresnet
    import gluon.models.squeezenet as gl_squeezenet
    import gluon.models.darknet as gl_darknet
    import gluon.models.mobilenet as gl_mobilenet

    models = {
        'resnet18': gl_resnet.resnet18,
        'resnet34': gl_resnet.resnet34,
        'resnet50': gl_resnet.resnet50,
        'resnet50b': gl_resnet.resnet50b,
        'resnet101': gl_resnet.resnet101,
        'resnet101b': gl_resnet.resnet101b,
        'resnet152': gl_resnet.resnet152,
        'resnet152b': gl_resnet.resnet152b,
        'resnet200': gl_resnet.resnet200,
        'resnet200b': gl_resnet.resnet200b,

        'preresnet18': gl_preresnet.preresnet18,
        'preresnet34': gl_preresnet.preresnet34,
        'preresnet50': gl_preresnet.preresnet50,
        'preresnet50b': gl_preresnet.preresnet50b,
        'preresnet101': gl_preresnet.preresnet101,
        'preresnet101b': gl_preresnet.preresnet101b,
        'preresnet152': gl_preresnet.preresnet152,
        'preresnet152b': gl_preresnet.preresnet152b,
        'preresnet200': gl_preresnet.preresnet200,
        'preresnet200b': gl_preresnet.preresnet200b,

        'squeezenet_v1_0': gl_squeezenet.squeezenet_v1_0,
        'squeezenet_v1_1': gl_squeezenet.squeezenet_v1_1,
        'squeezeresnet_v1_0': gl_squeezenet.squeezeresnet_v1_0,
        'squeezeresnet_v1_1': gl_squeezenet.squeezeresnet_v1_1,

        'darknet_ref': gl_darknet.darknet_ref,
        'darknet_tiny': gl_darknet.darknet_tiny,
        'darknet19': gl_darknet.darknet19,

        'mobilenet_w1': gl_mobilenet.mobilenet_w1,
        'mobilenet_w3d4': gl_mobilenet.mobilenet_w3d4,
        'mobilenet_wd2': gl_mobilenet.mobilenet_wd2,
        'mobilenet_wd4': gl_mobilenet.mobilenet_wd4,
        'fdmobilenet_w1': gl_mobilenet.fdmobilenet_w1,
        'fdmobilenet_w3d4': gl_mobilenet.fdmobilenet_w3d4,
        'fdmobilenet_wd2': gl_mobilenet.fdmobilenet_wd2,
        'fdmobilenet_wd4': gl_mobilenet.fdmobilenet_wd4,

        # 'squeezenet1_0': squeezenet1_0,
        # 'squeezenet1_1': squeezenet1_1,
        # 'squeezeresnet1_0': squeezeresnet1_0,
        # 'squeezeresnet1_1': squeezeresnet1_1,
        #
        # 'sqnxt23_1_0': sqnxt23_1_0,
        # 'sqnxt23_1_5': sqnxt23_1_5,
        # 'sqnxt23_2_0': sqnxt23_2_0,
        # 'sqnxt23v5_1_0': sqnxt23v5_1_0,
        # 'sqnxt23v5_1_5': sqnxt23v5_1_5,
        # 'sqnxt23v5_2_0': sqnxt23v5_2_0,
        #
        # 'nasnet_a_mobile': nasnet_a_mobile,
        #
        # 'mobilenet1_0': mobilenet1_0,
        # 'mobilenet0_75': mobilenet0_75,
        # 'mobilenet0_5': mobilenet0_5,
        # 'mobilenet0_25': mobilenet0_25,
        # 'fd_mobilenet1_0': fd_mobilenet1_0,
        # 'fd_mobilenet0_75': fd_mobilenet0_75,
        # 'fd_mobilenet0_5': fd_mobilenet0_5,
        # 'fd_mobilenet0_25': fd_mobilenet0_25,
        # 'shufflenet1_0_g1': shufflenet1_0_g1,
        # 'shufflenet1_0_g2': shufflenet1_0_g2,
        # 'shufflenet1_0_g3': shufflenet1_0_g3,
        # 'shufflenet1_0_g4': shufflenet1_0_g4,
        # 'shufflenet1_0_g8': shufflenet1_0_g8,
        # 'shufflenet0_5_g1': shufflenet0_5_g1,
        # 'shufflenet0_5_g3': shufflenet0_5_g3,
        # 'shufflenet0_25_g1': shufflenet0_25_g1,
        # 'shufflenet0_25_g3': shufflenet0_25_g3,
        # 'menet108_8x1_g3': menet108_8x1_g3,
        # 'menet128_8x1_g4': menet128_8x1_g4,
        # 'menet160_8x1_g8': menet160_8x1_g8,
        # 'menet228_12x1_g3': menet228_12x1_g3,
        # 'menet256_12x1_g4': menet256_12x1_g4,
        # 'menet348_12x1_g3': menet348_12x1_g3,
        # 'menet352_12x1_g8': menet352_12x1_g8,
        # 'menet456_24x1_g3': menet456_24x1_g3,
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


def _get_model_pt(name, **kwargs):
    import torchvision.models as models
    import pytorch.models.preresnet as pt_preresnet
    import pytorch.models.resnet as pt_resnet
    import pytorch.models.squeezenet as pt_squeezenet
    import pytorch.models.darknet as pt_darknet
    import pytorch.models.mobilenet as pt_mobilenet

    slk_models = {
        'resnet18': pt_resnet.resnet18,
        'resnet34': pt_resnet.resnet34,
        'resnet50': pt_resnet.resnet50,
        'resnet50b': pt_resnet.resnet50b,
        'resnet101': pt_resnet.resnet101,
        'resnet101b': pt_resnet.resnet101b,
        'resnet152': pt_resnet.resnet152,
        'resnet152b': pt_resnet.resnet152b,
        'resnet200': pt_resnet.resnet200,
        'resnet200b': pt_resnet.resnet200b,

        'preresnet18': pt_preresnet.preresnet18,
        'preresnet34': pt_preresnet.preresnet34,
        'preresnet50': pt_preresnet.preresnet50,
        'preresnet50b': pt_preresnet.preresnet50b,
        'preresnet101': pt_preresnet.preresnet101,
        'preresnet101b': pt_preresnet.preresnet101b,
        'preresnet152': pt_preresnet.preresnet152,
        'preresnet152b': pt_preresnet.preresnet152b,
        'preresnet200': pt_preresnet.preresnet200,
        'preresnet200b': pt_preresnet.preresnet200b,

        'squeezenet_v1_0': pt_squeezenet.squeezenet_v1_0,
        'squeezenet_v1_1': pt_squeezenet.squeezenet_v1_1,
        'squeezeresnet_v1_0': pt_squeezenet.squeezeresnet_v1_0,
        'squeezeresnet_v1_1': pt_squeezenet.squeezeresnet_v1_1,

        'darknet_ref': pt_darknet.darknet_ref,
        'darknet_tiny': pt_darknet.darknet_tiny,
        'darknet19': pt_darknet.darknet19,

        'mobilenet_w1': pt_mobilenet.mobilenet_w1,
        'mobilenet_w3d4': pt_mobilenet.mobilenet_w3d4,
        'mobilenet_wd2': pt_mobilenet.mobilenet_wd2,
        'mobilenet_wd4': pt_mobilenet.mobilenet_wd4,
        'fdmobilenet_w1': pt_mobilenet.fdmobilenet_w1,
        'fdmobilenet_w3d4': pt_mobilenet.fdmobilenet_w3d4,
        'fdmobilenet_wd2': pt_mobilenet.fdmobilenet_wd2,
        'fdmobilenet_wd4': pt_mobilenet.fdmobilenet_wd4,

        # 'oth_mobilenet1_0': oth_mobilenet1_0,
        # 'oth_mobilenet0_75': oth_mobilenet0_75,
        # 'oth_mobilenet0_5': oth_mobilenet0_5,
        # 'oth_mobilenet0_25': oth_mobilenet0_25,
        # 'oth_fd_mobilenet1_0': oth_fd_mobilenet1_0,
        # 'oth_fd_mobilenet0_75': oth_fd_mobilenet0_75,
        # 'oth_fd_mobilenet0_5': oth_fd_mobilenet0_5,
        # 'oth_fd_mobilenet0_25': oth_fd_mobilenet0_25,
        # 'oth_shufflenet1_0_g1': oth_shufflenet1_0_g1,
        # 'oth_shufflenet1_0_g8': oth_shufflenet1_0_g8,
        # 'oth_menet108_8x1_g3': oth_menet108_8x1_g3,
        # 'oth_menet128_8x1_g4': oth_menet128_8x1_g4,
        # 'oth_menet160_8x1_g8': oth_menet160_8x1_g8,
        # 'oth_menet228_12x1_g3': oth_menet228_12x1_g3,
        # 'oth_menet256_12x1_g4': oth_menet256_12x1_g4,
        # 'oth_menet348_12x1_g3': oth_menet348_12x1_g3,
        # 'oth_menet352_12x1_g8': oth_menet352_12x1_g8,
        # 'oth_menet456_24x1_g3': oth_menet456_24x1_g3,
        #
        # 'slk_squeezenet1_0': squeezenet1_0,
        # 'slk_squeezenet1_1': squeezenet1_1,
        # 'mobilenet1_0': mobilenet1_0,
        # 'mobilenet0_75': mobilenet0_75,
        # 'mobilenet0_5': mobilenet0_5,
        # 'mobilenet0_25': mobilenet0_25,
        # 'fd_mobilenet1_0': fd_mobilenet1_0,
        # 'fd_mobilenet0_75': fd_mobilenet0_75,
        # 'fd_mobilenet0_5': fd_mobilenet0_5,
        # 'fd_mobilenet0_25': fd_mobilenet0_25,
        # 'shufflenet1_0_g1': shufflenet1_0_g1,
        # 'shufflenet1_0_g2': shufflenet1_0_g2,
        # 'shufflenet1_0_g3': shufflenet1_0_g3,
        # 'shufflenet1_0_g4': shufflenet1_0_g4,
        # 'shufflenet1_0_g8': shufflenet1_0_g8,
        # 'shufflenet0_5_g1': shufflenet0_5_g1,
        # 'shufflenet0_5_g3': shufflenet0_5_g3,
        # 'shufflenet0_25_g1': shufflenet0_25_g1,
        # 'shufflenet0_25_g3': shufflenet0_25_g3,
        #
        # 'menet108_8x1_g3': menet108_8x1_g3,
        # 'menet128_8x1_g4': menet128_8x1_g4,
        # 'menet160_8x1_g8': menet160_8x1_g8,
        # 'menet228_12x1_g3': menet228_12x1_g3,
        # 'menet256_12x1_g4': menet256_12x1_g4,
        # 'menet348_12x1_g3': menet348_12x1_g3,
        # 'menet352_12x1_g8': menet352_12x1_g8,
        # 'menet456_24x1_g3': menet456_24x1_g3,
    }
    try:
        net = models.__dict__[name](**kwargs)
        return net
    except KeyError as e:
        upstream_supported = str(e)
    name = name.lower()
    if name not in slk_models:
        raise ValueError('%s\n\t%s' % (upstream_supported, '\n\t'.join(sorted(slk_models.keys()))))
    net = slk_models[name](**kwargs)
    return net


def prepare_model_gl(model_name,
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

    net = _get_model_gl(model_name, **kwargs)

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


def prepare_model_pt(model_name,
                     classes,
                     use_pretrained,
                     pretrained_model_file_path,
                     use_cuda):
    kwargs = {'pretrained': use_pretrained,
              'num_classes': classes}

    net = _get_model_pt(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        checkpoint = torch.load(pretrained_model_file_path)
        if type(checkpoint) == dict:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)

    if use_cuda:
        net = net.cuda()

    return net


def main():
    args = parse_args()

    _, log_file_exist = prepare_logger(
        log_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name)
    logging.info("Script command line:\n{}".format(" ".join(sys.argv)))
    logging.info("Script arguments:\n{}".format(args))

    ctx = mx.cpu()
    num_classes = 1000

    src_net = prepare_model_gl(
        model_name=args.src_model,
        classes=num_classes,
        use_pretrained=False,
        pretrained_model_file_path=args.src_params,
        batch_norm=False,
        use_se=False,
        last_gamma=False,
        dtype=np.float32,
        ctx=ctx)

    dst_net = prepare_model_pt(
        model_name=args.dst_model,
        classes=num_classes,
        use_pretrained=False,
        pretrained_model_file_path="",
        use_cuda=False)

    src_params = src_net._collect_params_with_prefix()
    dst_params = dst_net.state_dict()
    src_param_keys = list(src_params.keys())
    dst_param_keys = list(dst_params.keys())
    dst_param_keys = [key for key in dst_param_keys if not key.endswith("num_batches_tracked")]
    assert (len(src_param_keys) == len(dst_param_keys))
    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        dst_params[dst_key] = torch.from_numpy(src_params[src_param_keys[i]]._data[0].asnumpy())

    torch.save(
        obj=dst_params,
        f=args.dst_params)

    logging.info('Convert gl-model {} into pt-model {}'.format(args.src_model, args.dst_model))


if __name__ == '__main__':
    main()

