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

    from gluon.utils import get_model
    net = get_model(model_name, **kwargs)

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

    from pytorch.utils import get_model
    net = get_model(model_name, **kwargs)

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

        if args.src_model in ["resnet50_v1", "resnet101_v1", "resnet152_v1"]:
            src_param_keys = [key for key in src_param_keys if
                              not (key.startswith("features.") and key.endswith(".bias"))]

        if args.src_model in ["resnet18_v2", "resnet34_v2", "resnet50_v2", "resnet101_v2", "resnet152_v2"]:
            src_param_keys = src_param_keys[4:]

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

    assert (len(src_param_keys) == len(dst_param_keys))

    if args.src_fwk == "gluon" and args.dst_fwk == "gluon":
        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            assert (dst_params[dst_key].shape == src_params[src_key].shape)
            assert (dst_key.split('.')[-1] == src_key.split('.')[-1])
            dst_params[dst_key]._load_init(src_params[src_key]._data[0], ctx)
        dst_net.save_parameters(args.dst_params)
    elif args.src_fwk == "pytorch" and args.dst_fwk == "pytorch":
        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            assert (tuple(dst_params[dst_key].size()) == tuple(src_params[src_key].size())),\
                "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                    src_key, dst_key, tuple(src_params[src_key].size()), tuple(dst_params[dst_key].size()))
            assert (dst_key.split('.')[-1] == src_key.split('.')[-1])
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

