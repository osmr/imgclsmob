import argparse
import logging
import os
import sys
import numpy as np

import mxnet as mx
import torch

from common.env_stats import get_env_stats


def parse_args():
    parser = argparse.ArgumentParser(description='Convert models (Gluon/PyTorch/MXNet)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--src-fwk',
        type=str,
        required=True,
        help='source model framework name')
    parser.add_argument(
        '--dst-fwk',
        type=str,
        required=True,
        help='destination model framework name')
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
    import gluon.models.mobilenetv2 as gl_mobilenetv2
    import gluon.models.shufflenet as gl_shufflenet
    import gluon.models.menet as gl_menet
    import gluon.models.squeezenext as gl_squeezenext
    import gluon.models.densenet as gl_densenet
    # import gluon.models.menet1 as gl_meneta
    # import gluon.models.squeezenext1 as gl_squeezenext1

    models = {
        'resnet10': gl_resnet.resnet10,
        'resnet12': gl_resnet.resnet12,
        'resnet14': gl_resnet.resnet14,
        'resnet16': gl_resnet.resnet16,
        'resnet18': gl_resnet.resnet18,
        'resnet18_w3d4': gl_resnet.resnet18_w3d4,
        'resnet18_wd2': gl_resnet.resnet18_wd2,
        'resnet18_wd4': gl_resnet.resnet18_wd4,
        'resnet34': gl_resnet.resnet34,
        'resnet50': gl_resnet.resnet50,
        'resnet50b': gl_resnet.resnet50b,
        'resnet101': gl_resnet.resnet101,
        'resnet101b': gl_resnet.resnet101b,
        'resnet152': gl_resnet.resnet152,
        'resnet152b': gl_resnet.resnet152b,
        'resnet200': gl_resnet.resnet200,
        'resnet200b': gl_resnet.resnet200b,

        'preresnet10': gl_preresnet.preresnet10,
        'preresnet12': gl_preresnet.preresnet12,
        'preresnet14': gl_preresnet.preresnet14,
        'preresnet16': gl_preresnet.preresnet16,
        'preresnet18': gl_preresnet.preresnet18,
        'preresnet18_w3d4': gl_preresnet.preresnet18_w3d4,
        'preresnet18_wd2': gl_preresnet.preresnet18_wd2,
        'preresnet18_wd4': gl_preresnet.preresnet18_wd4,
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

        'mobilenetv2_w1': gl_mobilenetv2.mobilenetv2_w1,
        'mobilenetv2_w3d4': gl_mobilenetv2.mobilenetv2_w3d4,
        'mobilenetv2_wd2': gl_mobilenetv2.mobilenetv2_wd2,
        'mobilenetv2_wd4': gl_mobilenetv2.mobilenetv2_wd4,

        'shufflenet_g1_w1': gl_shufflenet.shufflenet_g1_w1,
        'shufflenet_g2_w1': gl_shufflenet.shufflenet_g2_w1,
        'shufflenet_g3_w1': gl_shufflenet.shufflenet_g3_w1,
        'shufflenet_g4_w1': gl_shufflenet.shufflenet_g4_w1,
        'shufflenet_g8_w1': gl_shufflenet.shufflenet_g8_w1,
        'shufflenet_g1_w3d4': gl_shufflenet.shufflenet_g1_w3d4,
        'shufflenet_g3_w3d4': gl_shufflenet.shufflenet_g3_w3d4,
        'shufflenet_g1_wd2': gl_shufflenet.shufflenet_g1_wd2,
        'shufflenet_g3_wd2': gl_shufflenet.shufflenet_g3_wd2,
        'shufflenet_g1_wd4': gl_shufflenet.shufflenet_g1_wd4,
        'shufflenet_g3_wd4': gl_shufflenet.shufflenet_g3_wd4,

        'menet108_8x1_g3': gl_menet.menet108_8x1_g3,
        'menet128_8x1_g4': gl_menet.menet128_8x1_g4,
        'menet160_8x1_g8': gl_menet.menet160_8x1_g8,
        'menet228_12x1_g3': gl_menet.menet228_12x1_g3,
        'menet256_12x1_g4': gl_menet.menet256_12x1_g4,
        'menet348_12x1_g3': gl_menet.menet348_12x1_g3,
        'menet352_12x1_g8': gl_menet.menet352_12x1_g8,
        'menet456_24x1_g3': gl_menet.menet456_24x1_g3,

        'sqnxt23_w1': gl_squeezenext.sqnxt23_w1,
        'sqnxt23_w3d2': gl_squeezenext.sqnxt23_w3d2,
        'sqnxt23_w2': gl_squeezenext.sqnxt23_w2,
        'sqnxt23v5_w1': gl_squeezenext.sqnxt23v5_w1,
        'sqnxt23v5_w3d2': gl_squeezenext.sqnxt23v5_w3d2,
        'sqnxt23v5_w2': gl_squeezenext.sqnxt23v5_w2,

        'densenet121': gl_densenet.densenet121,
        'densenet161': gl_densenet.densenet161,
        'densenet169': gl_densenet.densenet169,
        'densenet201': gl_densenet.densenet201,

        # 'sqnxt23_1_0': gl_squeezenext1.sqnxt23_1_0,
        # 'sqnxt23_1_5': gl_squeezenext1.sqnxt23_1_5,
        # 'sqnxt23_2_0': gl_squeezenext1.sqnxt23_2_0,
        # 'sqnxt23v5_1_0': gl_squeezenext1.sqnxt23v5_1_0,
        # 'sqnxt23v5_1_5': gl_squeezenext1.sqnxt23v5_1_5,
        # 'sqnxt23v5_2_0': gl_squeezenext1.sqnxt23v5_2_0,

        # 'menet108_8x1_g3a': gl_meneta.menet108_8x1_g3,
        # 'menet128_8x1_g4a': gl_meneta.menet128_8x1_g4,
        # 'menet160_8x1_g8a': gl_meneta.menet160_8x1_g8,
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
    import pytorch.models.mobilenetv2 as pt_mobilenetv2
    import pytorch.models.shufflenet as pt_shufflenet
    import pytorch.models.menet as pt_menet
    import pytorch.models.densenet as pt_densenet

    import pytorch.models.others.MobileNet as pt_oth_mobilenet
    import pytorch.models.others.MENet as pt_oth_menet

    slk_models = {
        'resnet10': pt_resnet.resnet10,
        'resnet12': pt_resnet.resnet12,
        'resnet14': pt_resnet.resnet14,
        'resnet16': pt_resnet.resnet16,
        'slk_resnet18': pt_resnet.resnet18,
        'resnet18_w3d4': pt_resnet.resnet18_w3d4,
        'resnet18_wd2': pt_resnet.resnet18_wd2,
        'resnet18_wd4': pt_resnet.resnet18_wd4,
        'slk_resnet34': pt_resnet.resnet34,
        'slk_resnet50': pt_resnet.resnet50,
        'resnet50b': pt_resnet.resnet50b,
        'slk_resnet101': pt_resnet.resnet101,
        'resnet101b': pt_resnet.resnet101b,
        'slk_resnet152': pt_resnet.resnet152,
        'resnet152b': pt_resnet.resnet152b,
        'resnet200': pt_resnet.resnet200,
        'resnet200b': pt_resnet.resnet200b,

        'preresnet10': pt_preresnet.preresnet10,
        'preresnet12': pt_preresnet.preresnet12,
        'preresnet14': pt_preresnet.preresnet14,
        'preresnet16': pt_preresnet.preresnet16,
        'preresnet18': pt_preresnet.preresnet18,
        'preresnet18_w3d4': pt_preresnet.preresnet18_w3d4,
        'preresnet18_wd2': pt_preresnet.preresnet18_wd2,
        'preresnet18_wd4': pt_preresnet.preresnet18_wd4,
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

        'mobilenetv2_w1': pt_mobilenetv2.mobilenetv2_w1,
        'mobilenetv2_w3d4': pt_mobilenetv2.mobilenetv2_w3d4,
        'mobilenetv2_wd2': pt_mobilenetv2.mobilenetv2_wd2,
        'mobilenetv2_wd4': pt_mobilenetv2.mobilenetv2_wd4,

        'shufflenet_g1_w1': pt_shufflenet.shufflenet_g1_w1,
        'shufflenet_g2_w1': pt_shufflenet.shufflenet_g2_w1,
        'shufflenet_g3_w1': pt_shufflenet.shufflenet_g3_w1,
        'shufflenet_g4_w1': pt_shufflenet.shufflenet_g4_w1,
        'shufflenet_g8_w1': pt_shufflenet.shufflenet_g8_w1,
        'shufflenet_g1_w3d4': pt_shufflenet.shufflenet_g1_w3d4,
        'shufflenet_g3_w3d4': pt_shufflenet.shufflenet_g3_w3d4,
        'shufflenet_g1_wd2': pt_shufflenet.shufflenet_g1_wd2,
        'shufflenet_g3_wd2': pt_shufflenet.shufflenet_g3_wd2,
        'shufflenet_g1_wd4': pt_shufflenet.shufflenet_g1_wd4,
        'shufflenet_g3_wd4': pt_shufflenet.shufflenet_g3_wd4,

        'menet108_8x1_g3': pt_menet.menet108_8x1_g3,
        'menet128_8x1_g4': pt_menet.menet128_8x1_g4,
        'menet160_8x1_g8': pt_menet.menet160_8x1_g8,
        'menet228_12x1_g3': pt_menet.menet228_12x1_g3,
        'menet256_12x1_g4': pt_menet.menet256_12x1_g4,
        'menet348_12x1_g3': pt_menet.menet348_12x1_g3,
        'menet352_12x1_g8': pt_menet.menet352_12x1_g8,
        'menet456_24x1_g3': pt_menet.menet456_24x1_g3,

        'densenet121': pt_densenet.densenet121,
        'densenet161': pt_densenet.densenet161,
        'densenet169': pt_densenet.densenet169,
        'densenet201': pt_densenet.densenet201,

        'oth_fd_mobilenet1_0': pt_oth_mobilenet.oth_fd_mobilenet1_0,
        'oth_fd_mobilenet0_75': pt_oth_mobilenet.oth_fd_mobilenet0_75,
        'oth_fd_mobilenet0_5': pt_oth_mobilenet.oth_fd_mobilenet0_5,
        'oth_fd_mobilenet0_25': pt_oth_mobilenet.oth_fd_mobilenet0_25,

        'oth_menet108_8x1_g3': pt_oth_menet.oth_menet108_8x1_g3,
        'oth_menet128_8x1_g4': pt_oth_menet.oth_menet128_8x1_g4,
        'oth_menet160_8x1_g8': pt_oth_menet.oth_menet160_8x1_g8,
        'oth_menet228_12x1_g3': pt_oth_menet.oth_menet228_12x1_g3,
        'oth_menet256_12x1_g4': pt_oth_menet.oth_menet256_12x1_g4,
        'oth_menet348_12x1_g3': pt_oth_menet.oth_menet348_12x1_g3,
        'oth_menet352_12x1_g8': pt_oth_menet.oth_menet352_12x1_g8,
        'oth_menet456_24x1_g3': pt_oth_menet.oth_menet456_24x1_g3,
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
        checkpoint = torch.load(
            pretrained_model_file_path,
            map_location=(None if use_cuda else 'cpu'))
        if type(checkpoint) == dict:
            net.load_state_dict(checkpoint['state_dict'])
        else:
            net.load_state_dict(checkpoint)

    if use_cuda:
        net = net.cuda()

    return net


def prepare_model_mx(pretrained_model_file_path,
                     ctx):

    # net = mx.mod.Module.load(
    #     prefix=pretrained_model_file_path,
    #     epoch=0,
    #     context=[ctx],
    #     label_names=[])
    # net.bind(
    #     data_shapes=[('data', (1, 3, 224, 224))],
    #     label_shapes=None,
    #     for_training=False)

    sym, arg_params, aux_params = mx.model.load_checkpoint(pretrained_model_file_path, 0)

    return sym, arg_params, aux_params


def main():
    args = parse_args()

    _, log_file_exist = prepare_logger(
        log_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name)
    logging.info("Script command line:\n{}".format(" ".join(sys.argv)))
    logging.info("Script arguments:\n{}".format(args))

    packages = []
    pip_packages = []
    if (args.src_fwk == "gluon") or (args.dst_fwk == "gluon"):
        packages += ['mxnet']
        pip_packages += ['mxnet-cu92', 'gluoncv']
    if (args.src_fwk == "pytorch") or (args.dst_fwk == "pytorch"):
        packages += ['torch', 'torchvision']
    logging.info("Env_stats:\n{}".format(get_env_stats(
        packages=packages,
        pip_packages=pip_packages)))

    ctx = mx.cpu()
    use_cuda = False
    num_classes = 1000

    if args.src_fwk == "gluon":
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
        src_params = src_net._collect_params_with_prefix()
        src_param_keys = list(src_params.keys())
    elif args.src_fwk == "pytorch":
        src_net = prepare_model_pt(
            model_name=args.src_model,
            classes=num_classes,
            use_pretrained=False,
            pretrained_model_file_path=args.src_params,
            use_cuda=use_cuda)
        src_params = src_net.state_dict()
        src_param_keys = list(src_params.keys())
        if args.dst_fwk != "pytorch":
            src_param_keys = [key for key in src_param_keys if not key.endswith("num_batches_tracked")]
    elif args.src_fwk == "mxnet":
        src_sym, src_arg_params, src_aux_params = prepare_model_mx(
            pretrained_model_file_path=args.src_params,
            ctx=ctx)
        src_param_keys = list(src_arg_params.keys())
    else:
        raise ValueError("Unsupported src fwk: {}".format(args.src_fwk))

    if args.dst_fwk == "gluon":
        dst_net = prepare_model_gl(
            model_name=args.dst_model,
            classes=num_classes,
            use_pretrained=False,
            pretrained_model_file_path="",
            batch_norm=False,
            use_se=False,
            last_gamma=False,
            dtype=np.float32,
            ctx=ctx)
        dst_params = dst_net._collect_params_with_prefix()
        dst_param_keys = list(dst_params.keys())
    elif args.dst_fwk == "pytorch":
        dst_net = prepare_model_pt(
            model_name=args.dst_model,
            classes=num_classes,
            use_pretrained=False,
            pretrained_model_file_path="",
            use_cuda=use_cuda)
        dst_params = dst_net.state_dict()
        dst_param_keys = list(dst_params.keys())
        if args.src_fwk != "pytorch":
            dst_param_keys = [key for key in dst_param_keys if not key.endswith("num_batches_tracked")]
    else:
        raise ValueError("Unsupported dst fwk: {}".format(args.dst_fwk))

    # assert (len(src_param_keys) == len(dst_param_keys) + 4)  # preresnet
    assert (len(src_param_keys) == len(dst_param_keys))

    if args.src_fwk == "gluon" and args.dst_fwk == "gluon":
        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            # assert (dst_params[dst_key].shape == src_params[src_param_keys[i+4]].shape)
            # dst_params[dst_key]._load_init(src_params[src_param_keys[i+4]]._data[0], ctx)  # preresnet
            assert (dst_params[dst_key].shape == src_params[src_key].shape)
            dst_params[dst_key]._load_init(src_params[src_key]._data[0], ctx)
        dst_net.save_parameters(args.dst_params)
    elif args.src_fwk == "pytorch" and args.dst_fwk == "pytorch":
        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            assert (tuple(dst_params[dst_key].size()) == tuple(src_params[src_key].size()))
            dst_params[dst_key] = torch.from_numpy(src_params[src_key].numpy())
        torch.save(
            obj=dst_params,
            f=args.dst_params)
    elif args.src_fwk == "gluon" and args.dst_fwk == "pytorch":
        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            assert (tuple(dst_params[dst_key].size()) == src_params[src_key].shape)
            dst_params[dst_key] = torch.from_numpy(src_params[src_key]._data[0].asnumpy())
        torch.save(
            obj=dst_params,
            f=args.dst_params)
    elif args.src_fwk == "pytorch" and args.dst_fwk == "gluon":
        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            assert (dst_params[dst_key].shape == tuple(src_params[src_key].size()))
            dst_params[dst_key]._load_init(mx.nd.array(src_params[src_key].numpy(), ctx), ctx)
        dst_net.save_parameters(args.dst_params)
    elif args.src_fwk == "mxnet" and args.dst_fwk == "gluon":

        dst_params['features.0.bn.beta']._load_init(src_arg_params['bn0_beta'], ctx)
        dst_params['features.0.bn.gamma']._load_init(src_arg_params['bn0_gamma'], ctx)
        dst_params['features.5.bn.beta']._load_init(src_arg_params['bn1_beta'], ctx)
        dst_params['features.5.bn.gamma']._load_init(src_arg_params['bn1_gamma'], ctx)
        dst_params['output.1.bias']._load_init(src_arg_params['fc1_bias'], ctx)
        dst_params['output.1.weight']._load_init(src_arg_params['fc1_weight'], ctx)
        dst_params['features.0.conv.weight']._load_init(src_arg_params['conv0_weight'], ctx)

        src_param_keys = [key for key in src_param_keys if (key.startswith("stage"))]

        dst_param_keys = [key for key in dst_param_keys if (not key.endswith("running_mean") and
                                                            not key.endswith("running_var") and
                                                            key.startswith("features") and
                                                            not key.startswith("features.0") and
                                                            not key.startswith("features.5"))]

        # src_param_keys = [key.replace('stage', 'features.') for key in src_param_keys]
        # src_param_keys = [key.replace('_', '.') for key in src_param_keys]

        src_param_keys = [key.replace('_conv1_', '.conv1.conv.') for key in src_param_keys]
        src_param_keys = [key.replace('_conv2_', '.conv2.conv.') for key in src_param_keys]
        src_param_keys = [key.replace('_conv3_', '.conv3.conv.') for key in src_param_keys]
        src_param_keys = [key.replace('_bn1_', '.conv1.bn.') for key in src_param_keys]
        src_param_keys = [key.replace('_bn2_', '.conv2.bn.') for key in src_param_keys]
        src_param_keys = [key.replace('_bn3_', '.conv3.bn.') for key in src_param_keys]

        src_param_keys.sort()
        src_param_keys.sort(key=lambda var: ['{:10}'.format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        dst_param_keys.sort()
        dst_param_keys.sort(key=lambda var: ['{:10}'.format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        src_param_keys = [key.replace('.conv1.conv.', '_conv1_') for key in src_param_keys]
        src_param_keys = [key.replace('.conv2.conv.', '_conv2_') for key in src_param_keys]
        src_param_keys = [key.replace('.conv3.conv.', '_conv3_') for key in src_param_keys]
        src_param_keys = [key.replace('.conv1.bn.', '_bn1_') for key in src_param_keys]
        src_param_keys = [key.replace('.conv2.bn.', '_bn2_') for key in src_param_keys]
        src_param_keys = [key.replace('.conv3.bn.', '_bn3_') for key in src_param_keys]

        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            dst_params[dst_key]._load_init(src_arg_params[src_key], ctx)

        for param in dst_net.collect_params().values():
            if param._data is not None:
                continue
            print('param={}'.format(param))
            param.initialize(ctx=ctx)

        dst_net.save_parameters(args.dst_params)

    logging.info('Convert {}-model {} into {}-model {}'.format(
        args.src_fwk, args.src_model, args.dst_fwk, args.dst_model))


if __name__ == '__main__':
    main()

