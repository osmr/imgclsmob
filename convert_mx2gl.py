import argparse
import time
import logging
import os
import sys
import numpy as np

import mxnet as mx
import torch


def parse_args():
    parser = argparse.ArgumentParser(description='Convert models (from MXNet to Gluon)',
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

        #
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
        # 'darknet_ref': darknet_ref,
        # 'darknet_tiny': darknet_tiny,
        # 'darknet19': darknet19,
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


def prepare_model_gl(model_name,
                     classes,
                     use_pretrained,
                     dtype,
                     ctx):

    kwargs = {'ctx': ctx,
              'pretrained': use_pretrained,
              'classes': classes}

    net = _get_model_gl(model_name, **kwargs)

    net.cast(dtype)

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

    src_sym, src_arg_params, src_aux_params = prepare_model_mx(
        pretrained_model_file_path=args.src_params,
        ctx=ctx)

    dst_net = prepare_model_gl(
        model_name=args.src_model,
        classes=num_classes,
        use_pretrained=False,
        dtype=np.float32,
        ctx=ctx)

    src_param_keys = list(src_arg_params.keys())
    src_param_keys.sort()

    import re
    src_param_keys.sort(key=lambda var: ['{:10}'.format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    dst_params = dst_net._collect_params_with_prefix()
    dst_param_keys = list(dst_params.keys())
    for i, dst_key in enumerate(dst_param_keys):
        #dst_params[dst_key] = 1
        dst_key_split = dst_key.split('.')
        if dst_key_split[0] == 'features':
            pass
        else:
            raise Exception()

        pass


    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        dst_params[dst_key] = torch.from_numpy(src_params[src_param_keys[i]]._data[0].asnumpy())

    torch.save(
        obj=dst_params,
        f=args.dst_params)

    logging.info('Convert mx-model {} into gl-model {}'.format(args.src_model, args.dst_model))


if __name__ == '__main__':
    main()

