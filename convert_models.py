"""
    Script for converting models between frameworks (MXNet, Gluon, PyTroch, Chainer, Keras, TensorFlow).
"""

import argparse
import logging
import re
import numpy as np
from common.logger_utils import initialize_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Convert models (Gluon/PyTorch/Chainer/MXNet/Keras/TF/TF2)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--src-fwk",
        type=str,
        required=True,
        help="source model framework name")
    parser.add_argument(
        "--dst-fwk",
        type=str,
        required=True,
        help="destination model framework name")
    parser.add_argument(
        "--src-model",
        type=str,
        required=True,
        help="source model name")
    parser.add_argument(
        "--dst-model",
        type=str,
        required=True,
        help="destination model name")
    parser.add_argument(
        "--src-params",
        type=str,
        default="",
        help="source model parameter file path")
    parser.add_argument(
        "--dst-params",
        type=str,
        default="",
        help="destination model parameter file path")
    parser.add_argument(
        "--load-ignore-extra",
        action="store_true",
        help="ignore extra layers in the source PyTroch model")
    parser.add_argument(
        "--remove-module",
        action="store_true",
        help="enable if stored PyTorch model has module")

    parser.add_argument(
        "--src-num-classes",
        type=int,
        default=1000,
        help="number of classes for source model")
    parser.add_argument(
        "--src-in-channels",
        type=int,
        default=3,
        help="number of input channels for source model")
    parser.add_argument(
        "--dst-num-classes",
        type=int,
        default=1000,
        help="number of classes for destination model")
    parser.add_argument(
        "--dst-in-channels",
        type=int,
        default=3,
        help="number of input channels for destination model")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="directory of saved models and log-files")
    parser.add_argument(
        "--logging-file-name",
        type=str,
        default="train.log",
        help="filename of training log")
    args = parser.parse_args()
    return args


def prepare_src_model(src_fwk,
                      src_model,
                      src_params_file_path,
                      dst_fwk,
                      ctx,
                      use_cuda,
                      load_ignore_extra=False,
                      remove_module=False,
                      num_classes=None,
                      in_channels=None):

    ext_src_param_keys = None
    ext_src_param_keys2 = None

    if src_fwk == "gluon":
        from gluon.utils import prepare_model as prepare_model_gl
        src_net = prepare_model_gl(
            model_name=src_model,
            use_pretrained=False,
            pretrained_model_file_path=src_params_file_path,
            dtype=np.float32,
            tune_layers="",
            classes=(num_classes if num_classes > 0 else None),
            in_channels=in_channels,
            ctx=ctx)
        src_params = src_net._collect_params_with_prefix()
        src_param_keys = list(src_params.keys())

        if src_model in ["oth_resnet50_v1", "oth_resnet101_v1", "oth_resnet152_v1", "oth_resnet50_v1b",
                         "oth_resnet101_v1b", "oth_resnet152_v1b"]:
            src_param_keys = [key for key in src_param_keys if
                              not (key.startswith("features.") and key.endswith(".bias"))]

        if src_model in ["oth_resnet50_v1", "oth_resnet101_v1", "oth_resnet152_v1", "oth_resnet50_v1b",
                         "oth_resnet101_v1b", "oth_resnet152_v1b"]:
            src_param_keys = [key for key in src_param_keys if
                              not (key.startswith("features.") and key.endswith(".bias"))]

        if src_model.startswith("wrn20_10_1bit") or src_model.startswith("wrn20_10_32bit"):
            src_param_keys = [key for key in src_param_keys if
                              not (key.startswith("features.") and
                                   (key.endswith(".bn.gamma") or key.endswith(".bn.beta")))]

        if dst_fwk == "chainer":
            src_param_keys_ = src_param_keys.copy()
            src_param_keys = [key for key in src_param_keys_ if (not key.endswith(".running_mean")) and
                              (not key.endswith(".running_var"))]
            ext_src_param_keys = [key for key in src_param_keys_ if (key.endswith(".running_mean")) or
                                  (key.endswith(".running_var"))]
            if src_model in ["condensenet74_c4_g4", "condensenet74_c8_g8"]:
                src_param_keys_ = src_param_keys.copy()
                src_param_keys = [key for key in src_param_keys_ if (not key.endswith(".index"))]
                ext_src_param_keys2 = [key for key in src_param_keys_ if (key.endswith(".index"))]
            elif src_model.startswith("xdensenet"):
                src_param_keys_ = src_param_keys.copy()
                src_param_keys = [key for key in src_param_keys_ if (not key.endswith(".mask"))]
                ext_src_param_keys2 = [key for key in src_param_keys_ if (key.endswith(".mask"))]

    elif src_fwk == "pytorch":
        from pytorch.utils import prepare_model as prepare_model_pt
        src_net = prepare_model_pt(
            model_name=src_model,
            use_pretrained=False,
            pretrained_model_file_path=src_params_file_path,
            use_cuda=use_cuda,
            use_data_parallel=False,
            load_ignore_extra=load_ignore_extra,
            num_classes=(num_classes if num_classes > 0 else None),
            in_channels=in_channels,
            remove_module=remove_module)
        src_params = src_net.state_dict()
        src_param_keys = list(src_params.keys())
        if dst_fwk != "pytorch":
            src_param_keys = [key for key in src_param_keys if not key.endswith("num_batches_tracked")]
        if src_model in ["oth_shufflenetv2_wd2"]:
            src_param_keys = [key for key in src_param_keys if not key.startswith("network.0.")]
        if src_model.startswith("oth_dla"):
            src1 = list(filter(re.compile("\.project").search, src_param_keys))
            src1n = [key for key in src_param_keys if key not in src1]
            src2 = []
            for i in range(2, 6):
                src1_i = list(filter(re.compile("level{}".format(i)).search, src1))
                if len(src1_i) == 0:
                    continue
                max_len = max([len(k) for k in src1_i])
                pattern_i = [k for k in src1_i if len(k) == max_len][0][:-21]
                src2_i = list(filter(re.compile(pattern_i).search, src1))
                src2 += src2_i
            src_param_keys = src2 + src1n

    elif src_fwk == "mxnet":
        import mxnet as mx
        src_sym, src_arg_params, src_aux_params = mx.model.load_checkpoint(
            prefix=src_params_file_path,
            epoch=0)
        src_params = {}
        src_params.update(src_arg_params)
        src_params.update(src_aux_params)
        src_param_keys = list(src_params.keys())
    elif src_fwk == "tensorflow":
        # import tensorflow as tf
        # from tensorflow_.utils import prepare_model as prepare_model_tf
        # src_net = prepare_model_tf(
        #     model_name=src_model,
        #     classes=num_classes,
        #     use_pretrained=False,
        #     pretrained_model_file_path=src_params_file_path)
        # src_param_keys = [v.name for v in tf.global_variables()]
        # src_params = {v.name: v for v in tf.global_variables()}

        src_net = None
        src_params = dict(np.load(src_params_file_path))
        src_param_keys = list(src_params.keys())
    else:
        raise ValueError("Unsupported src fwk: {}".format(src_fwk))

    return src_params, src_param_keys, ext_src_param_keys, ext_src_param_keys2


def prepare_dst_model(dst_fwk,
                      dst_model,
                      src_fwk,
                      ctx,
                      use_cuda,
                      num_classes=None,
                      in_channels=None):
    if dst_fwk == "gluon":
        from gluon.utils import prepare_model as prepare_model_gl
        dst_net = prepare_model_gl(
            model_name=dst_model,
            use_pretrained=False,
            pretrained_model_file_path="",
            dtype=np.float32,
            tune_layers="",
            classes=(num_classes if num_classes > 0 else None),
            in_channels=in_channels,
            ctx=ctx)
        dst_params = dst_net._collect_params_with_prefix()
        dst_param_keys = list(dst_params.keys())
    elif dst_fwk == "pytorch":
        from pytorch.utils import prepare_model as prepare_model_pt
        dst_net = prepare_model_pt(
            model_name=dst_model,
            use_pretrained=False,
            pretrained_model_file_path="",
            use_cuda=use_cuda,
            use_data_parallel=False,
            num_classes=(num_classes if num_classes > 0 else None),
            in_channels=in_channels)
        dst_params = dst_net.state_dict()
        dst_param_keys = list(dst_params.keys())
        if src_fwk != "pytorch":
            dst_param_keys = [key for key in dst_param_keys if not key.endswith("num_batches_tracked")]
    elif dst_fwk == "chainer":
        from chainer_.utils import prepare_model as prepare_model_ch
        dst_net = prepare_model_ch(
            model_name=dst_model,
            use_pretrained=False,
            pretrained_model_file_path="")
        dst_params = {i[0]: i[1] for i in dst_net.namedparams()}
        dst_param_keys = list(dst_params.keys())
    elif dst_fwk == "keras":
        from keras_.utils import prepare_model as prepare_model_ke
        dst_net = prepare_model_ke(
            model_name=dst_model,
            use_pretrained=False,
            pretrained_model_file_path="")
        # dst_param_keys = list(dst_net._arg_names) + list(dst_net._aux_names)
        dst_param_keys = [v.name for v in dst_net.weights]
        dst_params = {}
        for layer in dst_net.layers:
            if layer.name:
                for weight in layer.weights:
                    if weight.name:
                        dst_params.setdefault(weight.name, []).append(weight)
                        dst_params[weight.name] = (layer, weight)
    elif dst_fwk == "tensorflow":
        import tensorflow as tf
        from tensorflow_.utils import prepare_model as prepare_model_tf
        dst_net = prepare_model_tf(
            model_name=dst_model,
            use_pretrained=False,
            pretrained_model_file_path="")
        dst_param_keys = [v.name for v in tf.global_variables()]
        dst_params = {v.name: v for v in tf.global_variables()}
    elif dst_fwk == "tf2":
        import tensorflow as tf
        from tensorflow2.utils import prepare_model as prepare_model_tf2
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        dst_net = prepare_model_tf2(
            model_name=dst_model,
            use_pretrained=False,
            pretrained_model_file_path="")
        batch_size = 1
        input_shape = ((batch_size, 3, dst_net.in_size[0], dst_net.in_size[1]) if
                       dst_net.data_format == "channels_first" else
                       (batch_size, dst_net.in_size[0], dst_net.in_size[1], 3))
        dst_net(tf.random.normal(input_shape))
        dst_param_keys = [v.name for v in dst_net.weights]
        dst_params = {v.name: v for v in dst_net.weights}
    else:
        raise ValueError("Unsupported dst fwk: {}".format(dst_fwk))

    return dst_params, dst_param_keys, dst_net


def convert_mx2gl(dst_net,
                  dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys,
                  src_model,
                  ctx):
    if src_model in ["crunet56", "crunet116"]:
        src_param_keys.sort()
        src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        src_param_keys = [re.sub("^conv", "features.", key) for key in src_param_keys]
        src_param_keys = [re.sub("^fc6", "output.1.", key) for key in src_param_keys]
        src_param_keys = [re.sub('_c1x1-a', '.body.conv1.', key) for key in src_param_keys]
        src_param_keys = [re.sub('_c3x3-b', '.body.conv2A.', key) for key in src_param_keys]
        src_param_keys = [re.sub('_c1x1-b', '.body.conv2B.', key) for key in src_param_keys]
        src_param_keys = [re.sub('_c1x1-c', '.body.conv3.', key) for key in src_param_keys]
        src_param_keys = [re.sub('_x__x_1x1_bases\[dim3\]_weight$', '_x__1.body.conv1.convT.weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__x_3x3_bases\[dim21\]_weight$', '_x__1.body.conv2.convT.weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__\(1\)_1x1_bases\[dim3\]_weight$', '_x__1.body.conv1.convQ.weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__\(1\)_3x3_bases\[dim21\]_weight$', '_x__1.body.conv2.convQ.weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__\(2\)_1x1_bases\[dim3\]_weight$', '_x__7.body.conv1.convQ.weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__\(2\)_3x3_bases\[dim21\]_weight$', '_x__7.body.conv2.convQ.weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__\(3\)_1x1_bases\[dim3\]_weight$', '_x__14.body.conv1.convQ.weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__\(3\)_3x3_bases\[dim21\]_weight$', '_x__14.body.conv2.convQ.weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_c1x1-w\(s\/2\)', '.input_convZ.', key) for key in src_param_keys]
        src_param_keys = [re.sub('_c1x1-w_weight$', '.input_convZ.conv.weight', key) for key in src_param_keys]
        src_param_keys = [re.sub('_c1x1-w\(s\/1\)', '.input_conv.', key) for key in src_param_keys]
        src_param_keys = [re.sub('_c1x1-w\(s\/key\)', '.identity_conv.', key) for key in src_param_keys]
        src_param_keys = [re.sub('__conv_weight$', '.conv.weight', key) for key in src_param_keys]
        src_param_keys = [re.sub('__bn__bn_beta$', '.bn.beta', key) for key in src_param_keys]
        src_param_keys = [re.sub('__bn__bn_gamma$', '.bn.gamma', key) for key in src_param_keys]
        src_param_keys = [re.sub('__bn__bn_moving_mean$', '.bn.running_mean', key) for key in src_param_keys]
        src_param_keys = [re.sub('__bn__bn_moving_var$', '.bn.running_var', key) for key in src_param_keys]
        src_param_keys = [re.sub('1_x_1__relu-sp__bn_', '1_x_1.conv.bnA.', key) for key in src_param_keys]

        src_param_keys.sort()
        src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        dst_param_keys.sort()
        dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        src_param_keys = [re.sub("^features\.", "conv", key) for key in src_param_keys]
        src_param_keys = [re.sub('^output\.1\.', 'fc6', key) for key in src_param_keys]
        src_param_keys = [re.sub('_x__1\.body\.conv1\.convT\.weight$', '_x__x_1x1_bases[dim3]_weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__1\.body\.conv2\.convT\.weight$', '_x__x_3x3_bases[dim21]_weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__1\.body\.conv1\.convQ\.weight$', '_x__(1)_1x1_bases[dim3]_weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__1\.body\.conv2\.convQ\.weight$', '_x__(1)_3x3_bases[dim21]_weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__7\.body\.conv1\.convQ\.weight$', '_x__(2)_1x1_bases[dim3]_weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__7\.body\.conv2\.convQ\.weight$', '_x__(2)_3x3_bases[dim21]_weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__14\.body\.conv1\.convQ\.weight$', '_x__(3)_1x1_bases[dim3]_weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('_x__14\.body\.conv2\.convQ\.weight$', '_x__(3)_3x3_bases[dim21]_weight', key)
                          for key in src_param_keys]
        src_param_keys = [re.sub('\.body\.conv1\.', '_c1x1-a', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.body\.conv2A\.', '_c3x3-b', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.body\.conv2B\.', '_c1x1-b', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.body\.conv3\.', '_c1x1-c', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.input_convZ\.conv\.weight$', '_c1x1-w_weight', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.input_convZ\.', '_c1x1-w(s/2)', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.input_conv\.', '_c1x1-w(s/1)', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.identity_conv\.', '_c1x1-w(s/key)', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.conv\.weight$', '__conv_weight', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.bn\.beta$', '__bn__bn_beta', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.bn\.gamma$', '__bn__bn_gamma', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.bn\.running_mean$', '__bn__bn_moving_mean', key) for key in src_param_keys]
        src_param_keys = [re.sub('\.bn\.running_var$', '__bn__bn_moving_var', key) for key in src_param_keys]
        src_param_keys = [re.sub('1_x_1\.conv\.bnA\.', '1_x_1__relu-sp__bn_', key) for key in src_param_keys]

        dst_i = 0
        for src_i, src_key in enumerate(src_param_keys):
            dst_key = dst_param_keys[dst_i]
            for tt in range(10):
                if (dst_key.split('.')[-1].split('_')[-1] == src_key.split('_')[-1]) and\
                        (dst_params[dst_key].shape == src_params[src_key].shape):
                    break
                assert (dst_key.split('.')[-1].split('_')[-1] == "weight")
                dst_i += 1
                dst_key = dst_param_keys[dst_i]
            dst_i += 1
            assert (dst_key.split('.')[-1].split('_')[-1] == src_key.split('_')[-1])
            assert (dst_params[dst_key].shape == src_params[src_key].shape), \
                "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                    src_key, dst_key, src_params[src_key].shape, dst_params[dst_key].shape)
            dst_params[dst_key]._load_init(src_params[src_key], ctx)

        for param in dst_net.collect_params().values():
            if param._data is not None:
                continue
            print("param={}".format(param))
            param.initialize(ctx=ctx)

        dst_net.save_parameters(dst_params_file_path)

        return

    elif src_model in ["igcv3_w1"]:
        src_param_keys = [key.replace("seq-", "features.") for key in src_param_keys]
        src_param_keys = [key.replace("fc_", "output.1.") for key in src_param_keys]
        src_param_keys = [key.replace('-batchnorm_beta', '.bn.beta') for key in src_param_keys]
        src_param_keys = [key.replace('-batchnorm_gamma', '.bn.gamma') for key in src_param_keys]
        src_param_keys = [key.replace('-batchnorm_moving_mean', '.bn.running_mean') for key in src_param_keys]
        src_param_keys = [key.replace('-batchnorm_moving_var', '.bn.running_var') for key in src_param_keys]
        src_param_keys = [key.replace('-conv2d_weight', '.conv.weight') for key in src_param_keys]
        src_param_keys = [key.replace('first-3x3-conv', 'features.A') for key in src_param_keys]
        src_param_keys = [key.replace('last-1x1-conv', 'features.B') for key in src_param_keys]
        src_param_keys = [key.replace('-exp', '.conv1') for key in src_param_keys]
        src_param_keys = [key.replace('-depthwise', '.conv2') for key in src_param_keys]
        src_param_keys = [key.replace('-linear', '.conv3') for key in src_param_keys]
        src_param_keys = [key.replace("-block", ".block") for key in src_param_keys]

        dst_param_keys = [key.replace('features.0.', 'features.A.') for key in dst_param_keys]
        dst_param_keys = [key.replace('features.6.', 'features.B.') for key in dst_param_keys]

        src_param_keys.sort()
        src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        dst_param_keys.sort()
        dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        src_param_keys = [key.replace('.bn.beta', '-batchnorm_beta') for key in src_param_keys]
        src_param_keys = [key.replace('.bn.gamma', '-batchnorm_gamma') for key in src_param_keys]
        src_param_keys = [key.replace('.bn.running_mean', '-batchnorm_moving_mean') for key in src_param_keys]
        src_param_keys = [key.replace('.bn.running_var', '-batchnorm_moving_var') for key in src_param_keys]
        src_param_keys = [key.replace('.conv.weight', '-conv2d_weight') for key in src_param_keys]
        src_param_keys = [key.replace('features.A', 'first-3x3-conv') for key in src_param_keys]
        src_param_keys = [key.replace('features.B', 'last-1x1-conv') for key in src_param_keys]
        src_param_keys = [key.replace('.conv1', '-exp') for key in src_param_keys]
        src_param_keys = [key.replace('.conv2', '-depthwise', ) for key in src_param_keys]
        src_param_keys = [key.replace('.conv3', '-linear') for key in src_param_keys]
        src_param_keys = [key.replace("features.", "seq-") for key in src_param_keys]
        src_param_keys = [key.replace("output.1.", "fc_") for key in src_param_keys]
        src_param_keys = [key.replace(".block", "-block") for key in src_param_keys]

        dst_param_keys = [key.replace('features.A.', 'features.0.') for key in dst_param_keys]
        dst_param_keys = [key.replace('features.B.', 'features.6.') for key in dst_param_keys]

    elif src_model in ["preresnet269b"]:

        dst_net.features[1][0].body.conv1.bn.initialize(ctx=ctx, verbose=True, force_reinit=True)
        dst1 = list(filter(re.compile("^features.1.0.body.conv1.bn.").search, dst_param_keys))
        dst_param_keys = [key for key in dst_param_keys if key not in dst1]

        src_param_keys.sort()
        src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        src_param_keys = [re.sub('^classifier_', "output.", key) for key in src_param_keys]
        src_param_keys = [re.sub('^res', "features.", key) for key in src_param_keys]
        src_param_keys = [re.sub('_conv1_weight$', '_conv1_aweight', key) for key in src_param_keys]
        src_param_keys = [re.sub('_conv2_weight$', '_conv2_aweight', key) for key in src_param_keys]
        src_param_keys = [re.sub('_conv3_weight$', '_conv3_aweight', key) for key in src_param_keys]

        src_param_keys.sort()
        src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        dst_param_keys.sort()
        dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                             x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

        src_param_keys = [re.sub("^output\.", "classifier_", key) for key in src_param_keys]
        src_param_keys = [re.sub("^features\.", "res", key) for key in src_param_keys]
        src_param_keys = [re.sub('_conv1_aweight$', '_conv1_weight', key) for key in src_param_keys]
        src_param_keys = [re.sub('_conv2_aweight$', '_conv2_weight', key) for key in src_param_keys]
        src_param_keys = [re.sub('_conv3_aweight$', '_conv3_weight', key) for key in src_param_keys]

    for src_i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        assert (dst_key.split('.')[-1].split('_')[-1] == src_key.split('_')[-1]), \
            "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                src_key, dst_key, src_params[src_key].shape, dst_params[dst_key].shape)
        assert (dst_params[dst_key].shape == src_params[src_key].shape), \
            "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                src_key, dst_key, src_params[src_key].shape, dst_params[dst_key].shape)
        dst_params[dst_key]._load_init(src_params[src_key], ctx)

    for param in dst_net.collect_params().values():
        if param._data is not None:
            continue
        print("param={}".format(param))
        param.initialize(ctx=ctx)

    dst_net.save_parameters(dst_params_file_path)


def convert_gl2ch(dst_net,
                  dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys,
                  ext_src_param_keys,
                  ext_src_param_keys2,
                  src_model):

    if src_model.startswith("diares") or src_model.startswith("diapreres"):
        src1 = list(filter(re.compile("^features\.[0-9]*\.\d*[1-9]\d*\.attention").search, src_param_keys))
        src1n = [key for key in src_param_keys if key not in src1]
        src_param_keys = src1n
        assert (len(src_param_keys) == len(dst_param_keys))

    dst_param_keys = [key.replace("/W", "/weight") for key in dst_param_keys]
    dst_param_keys = [key.replace("/post_activ/", "/stageN/post_activ/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/features/body/", "/features/zbody/") for key in dst_param_keys]
    dst_param_keys = [key.replace("features/final_postactiv/", "features/stageN/final_postactiv/") for key in dst_param_keys]
    dst_param_keys = [key.replace("features/final_block/", "features/stageN/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/final_block/", "/zfinal_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("features/final_conv/", "features/stageN/final_conv/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stem1_unit/", "/stage0/stem1_unit/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stem2_unit/", "/stage0/stem2_unit/") for key in dst_param_keys]
    if not src_model.startswith("ibppose_coco"):
        dst_param_keys = [key.replace("/hg/", "/stage1_hg/") for key in dst_param_keys]

    if src_model.startswith("centernet"):
        dst_param_keys = [key.replace("/unit", "/a_unit") for key in dst_param_keys]
        dst_param_keys = [key.replace("/reg_block/", "/z_reg_block/") for key in dst_param_keys]

    src_param_keys.sort()
    src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys.sort()
    dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys = [key.replace("/weight", "/W") for key in dst_param_keys]
    dst_param_keys = [key.replace("/zfinal_block/", "/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stageN/post_activ/", "/post_activ/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stageN/final_postactiv/", "/final_postactiv/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stageN/final_block/", "/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/features/zbody/", "/features/body/") for key in dst_param_keys]
    dst_param_keys = [key.replace("features/stageN/final_conv/", "features/final_conv/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stage0/stem1_unit/", "/stem1_unit/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stage0/stem2_unit/", "/stem2_unit/") for key in dst_param_keys]
    if not src_model.startswith("ibppose_coco"):
        dst_param_keys = [key.replace("/stage1_hg/", "/hg/") for key in dst_param_keys]

    if src_model.startswith("centernet"):
        dst_param_keys = [key.replace("/a_unit", "/unit") for key in dst_param_keys]
        dst_param_keys = [key.replace("/z_reg_block/", "/reg_block/") for key in dst_param_keys]

    if src_model.startswith("wrn20_10_1bit") or src_model.startswith("wrn20_10_32bit"):
        ext2_src_param_keys = [key.replace('.conv.weight', '.bn.beta') for key in src_param_keys if
                               key.endswith(".conv.weight")]
        ext2_src_param_keys.append("features.4.bn.beta")
        ext2_dst_param_keys = [key.replace("/conv/W", "/bn/beta") for key in dst_param_keys if key.endswith("/conv/W")]
        ext2_dst_param_keys.append("/features/post_activ/bn/beta")
        ext3_src_param_keys = {".".join(v.split(".")[:-1]): i for i, v in enumerate(ext2_src_param_keys)}
        ext3_dst_param_keys = list(map(lambda x: x.split("/")[1:-1], ext2_dst_param_keys))
    else:
        ext2_src_param_keys = [key for key in src_param_keys if key.endswith(".beta")]
        ext2_dst_param_keys = [key for key in dst_param_keys if key.endswith("/beta")]
        ext3_src_param_keys = {".".join(v.split(".")[:-1]): i for i, v in enumerate(ext2_src_param_keys)}
        ext3_dst_param_keys = list(map(lambda x: x.split("/")[1:-1], ext2_dst_param_keys))

    for i, src_key in enumerate(ext_src_param_keys):
        src_key1 = src_key.split(".")[-1]
        src_key2 = ".".join(src_key.split(".")[:-1])
        dst_ind = ext3_src_param_keys[src_key2]
        dst_path = ext3_dst_param_keys[dst_ind]
        obj = dst_net
        for j, sub_path in enumerate(dst_path):
            obj = getattr(obj, sub_path)
        if src_key1 == 'running_mean':
            assert (obj.avg_mean.shape == src_params[src_key].shape), \
                "src_key={}, dst_path={}, src_shape={}, obj.avg_mean.shape={}".format(
                    src_key, dst_path, src_params[src_key].shape, obj.avg_mean.shape)
            obj.avg_mean = src_params[src_key]._data[0].asnumpy()
        elif src_key1 == 'running_var':
            assert (obj.avg_var.shape == src_params[src_key].shape)
            obj.avg_var = src_params[src_key]._data[0].asnumpy()

    if src_model in ["condensenet74_c4_g4", "condensenet74_c8_g8"]:
        assert (dst_net.output.fc.index.shape == src_params["output.1.index"].shape)
        dst_net.output.fc.index = src_params["output.1.index"]._data[0].asnumpy().astype(np.int32)
        ext_src_param_keys2.remove("output.1.index")

        ext2_src_param_keys = [key for key in src_param_keys if key.endswith(".conv1.conv.weight")]
        ext2_dst_param_keys = [key for key in dst_param_keys if key.endswith("/conv1/conv/W")]
        ext3_src_param_keys = {".".join(v.split(".")[:-2]): i for i, v in enumerate(ext2_src_param_keys)}
        ext3_dst_param_keys = list(map(lambda x: x.split("/")[1:-2], ext2_dst_param_keys))

        for i, src_key in enumerate(ext_src_param_keys2):
            src_key2 = ".".join(src_key.split(".")[:-1])
            dst_ind = ext3_src_param_keys[src_key2]
            dst_path = ext3_dst_param_keys[dst_ind]
            obj = dst_net
            for j, sub_path in enumerate(dst_path):
                obj = getattr(obj, sub_path)
            assert (obj.index.shape == src_params[src_key].shape), \
                "src_key={}, dst_path={}, src_shape={}, obj.index.shape={}".format(
                    src_key, dst_path, src_params[src_key].shape, obj.index.shape)
            obj.index = src_params[src_key]._data[0].asnumpy().astype(np.int32)
    elif src_model.startswith("xdensenet"):
        ext2_src_param_keys = [key for key in src_param_keys if key.endswith(".conv1.conv.weight")] +\
                              [key for key in src_param_keys if key.endswith(".conv2.conv.weight")]
        ext2_dst_param_keys = [key for key in dst_param_keys if key.endswith("/conv1/conv/W")] +\
                              [key for key in dst_param_keys if key.endswith("/conv2/conv/W")]
        ext3_src_param_keys = {".".join(v.split(".")[:-1]): i for i, v in enumerate(ext2_src_param_keys)}
        ext3_dst_param_keys = list(map(lambda x: x.split("/")[1:-1], ext2_dst_param_keys))

        for i, src_key in enumerate(ext_src_param_keys2):
            src_key2 = ".".join(src_key.split(".")[:-1])
            dst_ind = ext3_src_param_keys[src_key2]
            dst_path = ext3_dst_param_keys[dst_ind]
            obj = dst_net
            for j, sub_path in enumerate(dst_path):
                obj = getattr(obj, sub_path)
            assert (obj.mask.shape == src_params[src_key].shape), \
                "src_key={}, dst_path={}, src_shape={}, obj.index.shape={}".format(
                    src_key, dst_path, src_params[src_key].shape, obj.mask.shape)
            obj.mask = src_params[src_key]._data[0].asnumpy()

    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        assert (dst_params[dst_key].array.shape == src_params[src_key].shape), \
            "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                src_key, dst_key, src_params[src_key].shape, dst_params[dst_key].array.shape)
        dst_params[dst_key].array = src_params[src_key]._data[0].asnumpy()
        # print("src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
        #         src_key, dst_key, src_params[src_key].shape, dst_params[dst_key].array.shape))

    from chainer.serializers import save_npz
    save_npz(
        file=dst_params_file_path,
        obj=dst_net)


def convert_gl2gl(dst_net,
                  dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys,
                  finetune,
                  src_model,
                  ctx):
    if src_model.startswith("oth_icnet_resnet50_citys"):
        src1 = list(filter(re.compile("^conv_sub1").search, src_param_keys))
        src1n = [key for key in src_param_keys if key not in src1]
        src_param_keys = src1 + src1n

        src2 = list(filter(re.compile("^head").search, src_param_keys))
        src2n = [key for key in src_param_keys if key not in src2]
        src_param_keys = src2n + src2

    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        if dst_params[dst_key].shape != src_params[src_key].shape:
            logging.warning(
                "dst_param.shape != src_param.shape, src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                    src_key, dst_key, src_params[src_key].shape, dst_params[dst_key].shape))
            if finetune:
                continue
            else:
                raise ValueError
        if dst_key.split('.')[-1] != src_key.split('.')[-1]:
            logging.warning(
                "dst_key.suff != src_key.suff, src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                    src_key, dst_key, src_params[src_key].shape, dst_params[dst_key].shape))
        dst_params[dst_key]._load_init(src_params[src_key]._data[0], ctx)
    dst_net.save_parameters(dst_params_file_path)


def convert_gl2ke(dst_net,
                  dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys):
    import mxnet as mx

    dst_param_keys = [key.replace("/post_activ/", "/stageN/post_activ/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/final_block/", "/stageN/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stem1_unit/", "/stage0/stem1_unit/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stem2_unit/", "/stage0/stem2_unit/") for key in dst_param_keys]

    src_param_keys.sort()
    src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys.sort()
    dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys = [key.replace("/stageN/post_activ/", "/post_activ/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stageN/final_block/", "/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stage0/stem1_unit/", "/stem1_unit/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stage0/stem2_unit/", "/stem2_unit/") for key in dst_param_keys]

    dst_param_keys_orig = dst_param_keys.copy()
    dst_param_keys = [s[:(s.find("convgroup") + 9)] + "/" + s.split("/")[-1] if s.find("convgroup") >= 0 else s
                      for s in dst_param_keys]
    dst_param_keys_uniq, dst_param_keys_index = np.unique(dst_param_keys, return_index=True)
    dst_param_keys = list(dst_param_keys_uniq[dst_param_keys_index.argsort()])
    # dst_param_keys = list(np.unique(dst_param_keys))

    assert (len(src_param_keys) == len(dst_param_keys))

    def process_width(src_key, dst_key, src_weight):
        dst_layer = dst_params[dst_key][0]
        dst_weight = dst_params[dst_key][1]
        if (dst_layer.__class__.__name__ in ["Conv2D"]) and dst_key.endswith("kernel1") and\
                (dst_layer.data_format == "channels_last"):
            src_weight = np.transpose(src_weight, (2, 3, 1, 0))
        if (dst_layer.__class__.__name__ in ["DepthwiseConv2D"]) and dst_key.endswith("kernel1") and\
                (dst_layer.data_format == "channels_last"):
            src_weight = np.transpose(src_weight, (2, 3, 0, 1))
        if (dst_layer.__class__.__name__ in ["Dense"]) and dst_key.endswith("kernel1"):
            src_weight = np.transpose(src_weight, (1, 0))
        assert (dst_weight._keras_shape == src_weight.shape), \
            "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                src_key, dst_key, src_weight.shape, dst_weight._keras_shape)
        # print("src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
        #     src_key, dst_key, src_weight.shape, dst_weight._keras_shape))
        dst_weight.bind(mx.nd.array(src_weight))

    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        if dst_key.find("convgroup") >= 0:
            dst_key_stem = dst_key[:(dst_key.find("convgroup") + 9)]
            dst_keys = [s for s in dst_param_keys_orig if s.startswith(dst_key_stem)]
            if src_key.endswith("weight"):
                dst_keys = [s for s in dst_keys if s.endswith("kernel1")]
            elif src_key.endswith("bias"):
                dst_keys = [s for s in dst_keys if s.endswith("bias1")]
            groups = len(dst_keys)
            src_weight0 = src_params[src_key]._data[0]
            src_weight0_list = mx.nd.split(src_weight0, axis=0, num_outputs=groups)
            for gi in range(groups):
                src_weight_gi = src_weight0_list[gi].asnumpy()
                dst_key_gi = dst_keys[gi]
                process_width(src_key, dst_key_gi, src_weight_gi)
        else:
            src_weight = src_params[src_key]._data[0].asnumpy()
            process_width(src_key, dst_key, src_weight)

    dst_net.save_weights(dst_params_file_path)


def convert_gl2tf(dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys):
    import mxnet as mx

    dst_param_keys = [key.replace("/kernel:", "/weight:") for key in dst_param_keys]
    dst_param_keys = [key.replace("/dw_kernel:", "/weight_dw:") for key in dst_param_keys]
    dst_param_keys = [key.replace("/post_activ/", "/stageN/post_activ/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/final_block/", "/stageN/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stem1_unit/", "/stage0/stem1_unit/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stem2_unit/", "/stage0/stem2_unit/") for key in dst_param_keys]

    src_param_keys.sort()
    src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys.sort()
    dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys = [key.replace("/weight:", "/kernel:") for key in dst_param_keys]
    dst_param_keys = [key.replace("/weight_dw:", "/dw_kernel:") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stageN/post_activ/", "/post_activ/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stageN/final_block/", "/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stage0/stem1_unit/", "/stem1_unit/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stage0/stem2_unit/", "/stem2_unit/") for key in dst_param_keys]

    dst_param_keys_orig = dst_param_keys.copy()
    dst_param_keys = [s[:(s.find("convgroup") + 9)] + "/" + s.split("/")[-1] if s.find("convgroup") >= 0 else s
                      for s in dst_param_keys]
    dst_param_keys_uniq, dst_param_keys_index = np.unique(dst_param_keys, return_index=True)
    dst_param_keys = list(dst_param_keys_uniq[dst_param_keys_index.argsort()])

    assert (len(src_param_keys) == len(dst_param_keys))

    import tensorflow as tf
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        def process_width(src_key, dst_key, src_weight):
            if len(src_weight.shape) == 4:
                if dst_key.split("/")[-1][:-2] == "dw_kernel":
                    src_weight = np.transpose(src_weight, axes=(2, 3, 0, 1))
                else:
                    src_weight = np.transpose(src_weight, axes=(2, 3, 1, 0))
            elif len(src_weight.shape) == 2:
                src_weight = np.transpose(src_weight, axes=(1, 0))
            assert (tuple(dst_params[dst_key].get_shape().as_list()) == src_weight.shape)
            sess.run(dst_params[dst_key].assign(src_weight))
            # print(dst_params[dst_key].eval(sess))

        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            if dst_key.find("convgroup") >= 0:
                dst_key_stem = dst_key[:(dst_key.find("convgroup") + 9)]
                dst_keys = [s for s in dst_param_keys_orig if s.startswith(dst_key_stem)]
                if src_key.endswith("weight"):
                    dst_keys = [s for s in dst_keys if s.endswith("kernel:0")]
                elif src_key.endswith("bias"):
                    dst_keys = [s for s in dst_keys if s.endswith("bias:0")]
                groups = len(dst_keys)
                src_weight0 = src_params[src_key]._data[0]
                src_weight0_list = mx.nd.split(src_weight0, axis=0, num_outputs=groups)
                for gi in range(groups):
                    src_weight_gi = src_weight0_list[gi].asnumpy()
                    dst_key_gi = dst_keys[gi]
                    process_width(src_key, dst_key_gi, src_weight_gi)
            else:
                src_weight = src_params[src_key]._data[0].asnumpy()
                process_width(src_key, dst_key, src_weight)
        # saver = tf.train.Saver()
        # saver.save(
        #     sess=sess,
        #     save_path=dst_params_file_path)
        from tensorflow_.utils import save_model_params
        save_model_params(
            sess=sess,
            file_path=dst_params_file_path)


def convert_gl2tf2(dst_net,
                   dst_params_file_path,
                   dst_params,
                   dst_param_keys,
                   src_params,
                   src_param_keys,
                   src_model):
    if src_model.startswith("hrnet"):
        src_param_keys = [key.replace(".transition.", ".atransition.") for key in src_param_keys]

    src_param_keys.sort()
    src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    if src_model.startswith("hrnet"):
        src_param_keys = [key.replace(".atransition.", ".transition.") for key in src_param_keys]

    dst_param_keys = [key.replace("/kernel:", "/weight:") for key in dst_param_keys]
    dst_param_keys = [key.replace("/depthwise_kernel:", "/weight_depthwise:") for key in dst_param_keys]
    dst_param_keys = [key.replace("/post_activ/", "/stageN/post_activ/") for key in dst_param_keys]

    if (not src_model.startswith("pspnet_")) and (not src_model.startswith("deeplabv3_")) and\
            (not src_model.startswith("simplepose_")) and (not src_model.startswith("alphapose_")) and\
            (not src_model.startswith("lwopenpose")):
        dst_param_keys = [key.replace("/final_block/", "/stageN/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/final_block/", "/zfinal_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stem1_unit/", "/stage0/stem1_unit/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stem2_unit/", "/stage0/stem2_unit/") for key in dst_param_keys]
    if src_model.startswith("hrnet"):
        dst_param_keys = [key.replace("/transition/", "/atransition/") for key in dst_param_keys]
    if src_model.startswith("hardnet"):
        # dst_param_keys = [key.replace('/dw_conv/', '/z_dw_conv/') for key in dst_param_keys]
        dst_param_keys = [key.replace("features/down", "features/z_down") for key in dst_param_keys]

    if src_model.startswith("centernet"):
        dst_param_keys = [key.replace("/unit", "/a_unit") for key in dst_param_keys]
        dst_param_keys = [key.replace("/reg_block/", "/z_reg_block/") for key in dst_param_keys]

    dst_param_keys.sort()
    dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys = [key.replace("/weight:", "/kernel:") for key in dst_param_keys]
    dst_param_keys = [key.replace("/weight_depthwise:", "/depthwise_kernel:") for key in dst_param_keys]
    dst_param_keys = [key.replace("/zfinal_block/", "/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stageN/post_activ/", "/post_activ/") for key in dst_param_keys]
    if (not src_model.startswith("pspnet_")) and (not src_model.startswith("deeplabv3_")) and\
            (not src_model.startswith("simplepose_")) and (not src_model.startswith("alphapose_")) and\
            (not src_model.startswith("lwopenpose")):
        dst_param_keys = [key.replace("/stageN/final_block/", "/final_block/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stage0/stem1_unit/", "/stem1_unit/") for key in dst_param_keys]
    dst_param_keys = [key.replace("/stage0/stem2_unit/", "/stem2_unit/") for key in dst_param_keys]
    if src_model.startswith("hrnet"):
        dst_param_keys = [key.replace("/atransition/", "/transition/") for key in dst_param_keys]
    if src_model.startswith("hardnet"):
        # dst_param_keys = [key.replace('/z_dw_conv/', '/dw_conv/') for key in dst_param_keys]
        dst_param_keys = [key.replace("features/z_down", "features/down") for key in dst_param_keys]

    if src_model.startswith("centernet"):
        dst_param_keys = [key.replace("/a_unit", "/unit") for key in dst_param_keys]
        dst_param_keys = [key.replace("/z_reg_block/", "/reg_block/") for key in dst_param_keys]

    dst_param_keys_orig = dst_param_keys.copy()
    dst_param_keys = [s[:(s.find("convgroup") + 9)] + "/" + s.split("/")[-1] if s.find("convgroup") >= 0 else s
                      for s in dst_param_keys]
    dst_param_keys_uniq, dst_param_keys_index = np.unique(dst_param_keys, return_index=True)
    dst_param_keys = list(dst_param_keys_uniq[dst_param_keys_index.argsort()])

    assert (len(src_param_keys) == len(dst_param_keys))

    def process_width(src_key, dst_key, src_weight):
        if len(src_weight.shape) == 4:
            if dst_key.split("/")[-1][:-2] == "depthwise_kernel":
                src_weight = np.transpose(src_weight, axes=(2, 3, 0, 1))
            else:
                src_weight = np.transpose(src_weight, axes=(2, 3, 1, 0))
        elif len(src_weight.shape) == 2:
            src_weight = np.transpose(src_weight, axes=(1, 0))
        dst_weight = dst_params[dst_key]
        assert (tuple(dst_weight.shape) == src_weight.shape), \
            "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                src_key, dst_key, src_weight.shape, tuple(dst_weight.shape))
        dst_weight.assign(src_weight)

    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        # print("src_key={},\tsrc_key2={},\tdst_key={}".format(src_key, src_params[src_key].name, dst_key))
        if dst_key.find("convgroup") >= 0:
            import mxnet as mx
            dst_key_stem = dst_key[:(dst_key.find("convgroup") + 9)]
            dst_keys = [s for s in dst_param_keys_orig if s.startswith(dst_key_stem)]
            if src_key.endswith("weight"):
                dst_keys = [s for s in dst_keys if s.endswith("kernel:0")]
            elif src_key.endswith("bias"):
                dst_keys = [s for s in dst_keys if s.endswith("bias:0")]
            groups = len(dst_keys)
            src_weight0 = src_params[src_key]._data[0]
            src_weight0_list = mx.nd.split(src_weight0, axis=0, num_outputs=groups)
            for gi in range(groups):
                src_weight_gi = src_weight0_list[gi].asnumpy()
                dst_key_gi = dst_keys[gi]
                process_width(src_key, dst_key_gi, src_weight_gi)
        else:
            src_weight = src_params[src_key]._data[0].asnumpy()
            process_width(src_key, dst_key, src_weight)

    dst_net.save_weights(dst_params_file_path)


def convert_pt2pt(dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys,
                  src_model,
                  dst_model):
    import torch
    if src_model.startswith("oth_proxyless"):
        src1 = src_param_keys[5]
        del src_param_keys[5]
        src_param_keys.insert(0, src1)
        src2 = src_param_keys[-3]
        del src_param_keys[-3]
        src_param_keys.insert(-7, src2)

    elif src_model == "oth_bisenet":
        src1 = list(filter(re.compile("^cp.conv_avg").search, src_param_keys))
        src1n = [key for key in src_param_keys if key not in src1]
        src2 = list(filter(re.compile("^cp.arm32").search, src1n))
        src2n = [key for key in src1n if key not in src2]
        src3 = list(filter(re.compile("^cp.conv_head32").search, src2n))
        src3n = [key for key in src2n if key not in src3]
        src4 = list(filter(re.compile("^cp.arm16").search, src3n))
        src4n = [key for key in src3n if key not in src4]
        src5 = list(filter(re.compile("^cp.conv_head16").search, src4n))
        src5n = [key for key in src4n if key not in src5]
        src6 = list(filter(re.compile("^ffm").search, src5n))
        src6n = [key for key in src5n if key not in src6]

        src_param_keys = src6n + src1 + src2 + src3 + src4 + src5 + src6

        dst1 = list(filter(re.compile("^pool").search, dst_param_keys))
        dst1n = [key for key in dst_param_keys if key not in dst1]

        dst_param_keys = dst1n + dst1

    elif src_model.startswith("oth_dla"):
        src1 = list(filter(re.compile("\.project").search, src_param_keys))
        src1n = [key for key in src_param_keys if key not in src1]
        src_param_keys = src1 + src1n
        dst1 = list(filter(re.compile("\.project_conv").search, dst_param_keys))
        dst1n = [key for key in dst_param_keys if key not in dst1]
        dst_param_keys = dst1 + dst1n

    elif dst_model == "ntsnet":
        src1 = list(filter(re.compile("^proposal_net").search, src_param_keys))
        src1n = [key for key in src_param_keys if key not in src1]
        src_param_keys = src1 + src1n
        dst1 = list(filter(re.compile("^navigator_unit\.branch\d+\.down").search, dst_param_keys))
        dst1n = [key for key in dst_param_keys if key not in dst1]
        dst2 = list(filter(re.compile("^navigator_unit\.branch\d+\.tidy").search, dst1n))
        dst2n = [key for key in dst1n if key not in dst2]
        dst_param_keys = dst1 + dst2 + dst2n

    elif dst_model == "fishnet150":
        src1 = list(filter(re.compile("^(conv|fish\.fish\.[0-2])").search, src_param_keys))
        src1n = [key for key in src_param_keys if key not in src1]
        src2 = list(filter(re.compile("^fish\.fish\.6\.1").search, src1n))
        src2n = [key for key in src1n if key not in src2]
        src3 = list(filter(re.compile("^fish\.fish\.5\.1").search, src2n))
        src3n = [key for key in src2n if key not in src3]
        src4 = list(filter(re.compile("^fish\.fish\.4\.1").search, src3n))
        src4n = [key for key in src3n if key not in src4]
        src5 = list(filter(re.compile("^fish\.fish\.3\.[0-1]").search, src4n))
        src5n = [key for key in src4n if key not in src5]
        src6 = list(filter(re.compile("^fish\.fish\.3\.3").search, src5n))
        src6n = [key for key in src5n if key not in src6]
        src7 = list(filter(re.compile("^fish\.fish\.[3-6]").search, src6n))
        src7n = [key for key in src6n if key not in src7]
        src8 = list(filter(re.compile("^fish\.fish\.9\.1").search, src7n))
        src8n = [key for key in src7n if key not in src8]
        src9 = list(filter(re.compile("^fish\.fish\.8\.1").search, src8n))
        src9n = [key for key in src8n if key not in src9]
        src10 = list(filter(re.compile("^fish\.fish\.7\.1").search, src9n))
        src10n = [key for key in src9n if key not in src10]
        src_param_keys = src1 + src2 + src3 + src4 + src5 + src6 + src7 + src8 + src9 + src10 + src10n

    elif dst_model == "bam_resnet50":
        src_bams = list(filter(re.compile("^bam").search, src_param_keys))
        src_param_keys = [key for key in src_param_keys if key not in src_bams]
        src_param_keys = src_param_keys + src_bams

        dst_bams = list(filter(re.compile("^features.stage[0-9].unit1.bam.").search, dst_param_keys))
        dst_param_keys = [key for key in dst_param_keys if key not in dst_bams]
        dst_param_keys = dst_param_keys + dst_bams

    elif dst_model.startswith("sinet"):
        src1 = list(filter(re.compile("\.vertical.weight").search, src_param_keys))
        src1n = [key for key in src_param_keys if key not in src1]
        src_param_keys = src1n + src1
        src2 = list(filter(re.compile("\.horizontal.weight").search, src_param_keys))
        src2n = [key for key in src_param_keys if key not in src2]
        src_param_keys = src2n + src2
        src3 = list(filter(re.compile("\.B_v\.").search, src_param_keys))
        src3n = [key for key in src_param_keys if key not in src3]
        src_param_keys = src3n + src3
        src4 = list(filter(re.compile("\.B_h\.").search, src_param_keys))
        src4n = [key for key in src_param_keys if key not in src4]
        src_param_keys = src4n + src4
        src5 = list(filter(re.compile("bn_4\.").search, src_param_keys))
        src5n = [key for key in src_param_keys if key not in src5]
        src_param_keys = src5n + src5
        src6 = list(filter(re.compile("bn_3\.").search, src_param_keys))
        src6n = [key for key in src_param_keys if key not in src6]
        src_param_keys = src6n + src6

        dst1 = list(filter(re.compile("\.v_conv.conv\.").search, dst_param_keys))
        dst1n = [key for key in dst_param_keys if key not in dst1]
        dst_param_keys = dst1n + dst1
        dst2 = list(filter(re.compile("\.h_conv.conv\.").search, dst_param_keys))
        dst2n = [key for key in dst_param_keys if key not in dst2]
        dst_param_keys = dst2n + dst2
        dst3 = list(filter(re.compile("\.v_conv.bn\.").search, dst_param_keys))
        dst3n = [key for key in dst_param_keys if key not in dst3]
        dst_param_keys = dst3n + dst3
        dst4 = list(filter(re.compile("\.h_conv.bn\.").search, dst_param_keys))
        dst4n = [key for key in dst_param_keys if key not in dst4]
        dst_param_keys = dst4n + dst4
        dst5 = list(filter(re.compile("decoder.decode1.bn\.").search, dst_param_keys))
        dst5n = [key for key in dst_param_keys if key not in dst5]
        dst_param_keys = dst5n + dst5
        dst6 = list(filter(re.compile("decoder.decode2.bn\.").search, dst_param_keys))
        dst6n = [key for key in dst_param_keys if key not in dst6]
        dst_param_keys = dst6n + dst6

    elif src_model.startswith("oth_ibppose"):
        def sort_hg(src2):
            src2b1 = list(filter(re.compile("^hourglass.[0-9].hg.0.1.").search, src2))
            src2b2 = list(filter(re.compile("^hourglass.[0-9].hg.1.1.").search, src2))
            src2b3 = list(filter(re.compile("^hourglass.[0-9].hg.2.1.").search, src2))
            src2b4 = list(filter(re.compile("^hourglass.[0-9].hg.3.1.").search, src2))

            src2b5 = list(filter(re.compile("^hourglass.[0-9].hg.3.2.").search, src2))
            src2b6 = list(filter(re.compile("^hourglass.[0-9].hg.3.3.").search, src2))
            src2b7 = list(filter(re.compile("^hourglass.[0-9].hg.2.2.").search, src2))
            src2b8 = list(filter(re.compile("^hourglass.[0-9].hg.2.3.").search, src2))
            src2b9 = list(filter(re.compile("^hourglass.[0-9].hg.1.2.").search, src2))
            src2b10 = list(filter(re.compile("^hourglass.[0-9].hg.1.3.").search, src2))
            src2b11 = list(filter(re.compile("^hourglass.[0-9].hg.0.2.").search, src2))
            src2b12 = list(filter(re.compile("^hourglass.[0-9].hg.0.3.").search, src2))

            src2b13 = list(filter(re.compile("^hourglass.[0-9].hg.0.0.").search, src2))
            src2b14 = list(filter(re.compile("^hourglass.[0-9].hg.1.0.").search, src2))
            src2b15 = list(filter(re.compile("^hourglass.[0-9].hg.2.0.").search, src2))
            src2b16 = list(filter(re.compile("^hourglass.[0-9].hg.3.0.").search, src2))
            src2b17 = list(filter(re.compile("^hourglass.[0-9].hg.3.4.").search, src2))
            return src2b1 + src2b2 + src2b3 + src2b4 +\
                   src2b11 + src2b12 + src2b9 + src2b10 + src2b7 + src2b8 + src2b5 + src2b6 +\
                   src2b13 + src2b14 + src2b15 + src2b16 + src2b17

        src1 = list(filter(re.compile("^pre.").search, src_param_keys))
        src1n = [key for key in src_param_keys if key not in src1]
        src_param_keys = src1n + src1
        src2 = list(filter(re.compile("^hourglass.").search, src_param_keys))
        src2n = [key for key in src_param_keys if key not in src2]
        src2b1 = sort_hg(list(filter(re.compile("^hourglass.0.hg.").search, src2)))
        src2b2 = sort_hg(list(filter(re.compile("^hourglass.1.hg.").search, src2)))
        src2b3 = sort_hg(list(filter(re.compile("^hourglass.2.hg.").search, src2)))
        src2b4 = sort_hg(list(filter(re.compile("^hourglass.3.hg.").search, src2)))
        src_param_keys = src2n + src2b1 + src2b2 + src2b3 + src2b4
        src3 = list(filter(re.compile("^features.[0-9].before_regress").search, src_param_keys))
        src3n = [key for key in src_param_keys if key not in src3]
        src3b = list(filter(re.compile("^features.[0-9].before_regress.0.").search, src3))
        src_param_keys = src3n + src3b
        src4 = list(filter(re.compile("^outs.[0-9].").search, src_param_keys))
        src4n = [key for key in src_param_keys if key not in src4]
        src4b = list(filter(re.compile("^outs.[0-9].0.").search, src4))
        src_param_keys = src4n + src4b
        src5 = list(filter(re.compile("^merge_features.[0-9].").search, src_param_keys))
        src5n = [key for key in src_param_keys if key not in src5]
        src5b = list(filter(re.compile("^merge_features.[0-9].0.").search, src5))
        src_param_keys = src5n + src5b
        src6 = list(filter(re.compile("^merge_preds.[0-9].").search, src_param_keys))
        src6n = [key for key in src_param_keys if key not in src6]
        src6b = list(filter(re.compile("^merge_preds.[0-9].0.").search, src6))
        src_param_keys = src6n + src6b

        dst1 = list(filter(re.compile("^backbone.").search, dst_param_keys))
        dst1n = [key for key in dst_param_keys if key not in dst1]
        dst_param_keys = dst1n + dst1
        dst2 = list(filter(re.compile("^decoder.pass[1-9].hg.").search, dst_param_keys))
        dst2n = [key for key in dst_param_keys if key not in dst2]
        dst_param_keys = dst2n + dst2
        dst3 = list(filter(re.compile("^decoder.pass[1-9].pre_block.").search, dst_param_keys))
        dst3n = [key for key in dst_param_keys if key not in dst3]
        dst_param_keys = dst3n + dst3
        dst4 = list(filter(re.compile("^decoder.pass[1-9].post_block.").search, dst_param_keys))
        dst4n = [key for key in dst_param_keys if key not in dst4]
        dst_param_keys = dst4n + dst4
        dst5 = list(filter(re.compile("^decoder.pass[1-9].pre_merge_block.").search, dst_param_keys))
        dst5n = [key for key in dst_param_keys if key not in dst5]
        dst_param_keys = dst5n + dst5
        dst6 = list(filter(re.compile("^decoder.pass[1-9].post_merge_block.").search, dst_param_keys))
        dst6n = [key for key in dst_param_keys if key not in dst6]
        dst_param_keys = dst6n + dst6

    assert (len(src_param_keys) == len(dst_param_keys))

    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        if (src_model == "oth_shufflenetv2_wd2" and dst_model == "shufflenetv2_wd2") and \
                (src_key == "network.8.weight"):
            dst_params[dst_key] = torch.from_numpy(src_params[src_key].numpy()[:, :, 0, 0])
        else:
            # print("src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
            #         src_key, dst_key, tuple(src_params[src_key].size()), tuple(dst_params[dst_key].size())))
            assert (tuple(dst_params[dst_key].size()) == tuple(src_params[src_key].size())), \
                "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                    src_key, dst_key, tuple(src_params[src_key].size()), tuple(dst_params[dst_key].size()))
            assert (dst_key.split('.')[-1] == src_key.split('.')[-1])
            dst_params[dst_key] = torch.from_numpy(src_params[src_key].numpy())
    torch.save(
        obj=dst_params,
        f=dst_params_file_path)


def convert_gl2pt(dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys):
    import torch
    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        assert (tuple(dst_params[dst_key].size()) == src_params[src_key].shape)
        dst_params[dst_key] = torch.from_numpy(src_params[src_key]._data[0].asnumpy())
    torch.save(
        obj=dst_params,
        f=dst_params_file_path)


def convert_pt2gl(dst_net,
                  dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys,
                  ctx):
    import mxnet as mx

    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        assert (dst_params[dst_key].shape == tuple(src_params[src_key].size())), \
            "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                src_key, dst_key, tuple(src_params[src_key].size()), dst_params[dst_key].shape)
        dst_params[dst_key]._load_init(mx.nd.array(src_params[src_key].numpy(), ctx), ctx)
    dst_net.save_parameters(dst_params_file_path)


def convert_tf2tf(dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys):
    import re

    src_param_keys = [key.replace("/W:", "/kernel:") for key in src_param_keys]
    src_param_keys = [key.replace("/b:", "/bias:") for key in src_param_keys]
    src_param_keys = [key.replace("linear/", "output/") for key in src_param_keys]
    src_param_keys = [key.replace("stage", "features/stage") for key in src_param_keys]
    src_param_keys = [re.sub("^conv1/", "features/init_block/conv/", key) for key in src_param_keys]
    src_param_keys = [re.sub("^conv5/", "features/final_block/conv/", key) for key in src_param_keys]
    src_param_keys = [key.replace('/dconv_bn/', '/dconv/bn/') for key in src_param_keys]
    src_param_keys = [key.replace('/shortcut_dconv_bn/', '/shortcut_dconv/bn/') for key in src_param_keys]

    src_param_keys.sort()
    src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys.sort()
    dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    src_param_keys = [key.replace("/kernel:", "/W:") for key in src_param_keys]
    src_param_keys = [key.replace("/bias:", "/b:") for key in src_param_keys]
    src_param_keys = [key.replace("output/", "linear/") for key in src_param_keys]
    src_param_keys = [key.replace("features/stage", "stage") for key in src_param_keys]
    src_param_keys = [key.replace("features/init_block/conv/", 'conv1/') for key in src_param_keys]
    src_param_keys = [key.replace("features/final_block/conv/", 'conv5/') for key in src_param_keys]
    src_param_keys = [key.replace('/dconv/bn/', '/dconv_bn/') for key in src_param_keys]
    src_param_keys = [key.replace('/shortcut_dconv/bn/', '/shortcut_dconv_bn/') for key in src_param_keys]

    assert (len(src_param_keys) == len(dst_param_keys))

    import tensorflow as tf
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
            assert (src_params[src_key].shape == tuple(dst_params[dst_key].get_shape().as_list()))
            sess.run(dst_params[dst_key].assign(src_params[src_key]))

        from tensorflow_.utils import save_model_params
        save_model_params(
            sess=sess,
            file_path=dst_params_file_path)


def convert_tf2gl(dst_net,
                  dst_params_file_path,
                  dst_params,
                  dst_param_keys,
                  src_params,
                  src_param_keys,
                  ctx):
    import mxnet as mx

    src_param_keys = [key.replace("/kernel:", "/weight:") for key in src_param_keys]
    src_param_keys = [key.replace("/dw_kernel:", "/weight_dw:") for key in src_param_keys]
    src_param_keys = [key.replace("/post_activ/", "/stageN/post_activ/") for key in src_param_keys]
    src_param_keys = [key.replace("/final_block/", "/stageN/final_block/") for key in src_param_keys]
    src_param_keys = [key.replace("/stem1_unit/", "/stage0/stem1_unit/") for key in src_param_keys]
    src_param_keys = [key.replace("/stem2_unit/", "/stage0/stem2_unit/") for key in src_param_keys]

    src_param_keys.sort()
    src_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    dst_param_keys.sort()
    dst_param_keys.sort(key=lambda var: ["{:10}".format(int(x)) if
                                         x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)])

    src_param_keys = [key.replace("/weight:", "/kernel:") for key in src_param_keys]
    src_param_keys = [key.replace("/weight_dw:", "/dw_kernel:") for key in src_param_keys]
    src_param_keys = [key.replace("/stageN/post_activ/", "/post_activ/") for key in src_param_keys]
    src_param_keys = [key.replace("/stageN/final_block/", "/final_block/") for key in src_param_keys]
    src_param_keys = [key.replace("/stage0/stem1_unit/", "/stem1_unit/") for key in src_param_keys]
    src_param_keys = [key.replace("/stage0/stem2_unit/", "/stem2_unit/") for key in src_param_keys]

    assert (len(src_param_keys) == len(dst_param_keys))

    for i, (src_key, dst_key) in enumerate(zip(src_param_keys, dst_param_keys)):
        src_weight = src_params[src_key]
        if len(src_weight.shape) == 4:
            if src_key.split("/")[-1][:-2] == "dw_kernel":
                dst_weight = np.transpose(src_weight, axes=(2, 3, 0, 1))
            else:
                dst_weight = np.transpose(src_weight, axes=(3, 2, 0, 1))
        elif len(src_weight.shape) == 2:
            dst_weight = np.transpose(src_weight, axes=(1, 0))
        else:
            dst_weight = src_weight
        assert (dst_weight.shape == dst_params[dst_key].shape), \
            "src_key={}, dst_key={}, src_shape={}, dst_shape={}".format(
                src_key, dst_key, dst_weight.shape, dst_params[dst_key].shape)
        dst_params[dst_key]._load_init(mx.nd.array(dst_weight, ctx), ctx)

    dst_net.save_parameters(dst_params_file_path)


def _init_ctx(args):
    ctx = None
    if args.src_fwk in ("gluon", "mxnet", "keras") or args.dst_fwk in ("gluon", "mxnet", "keras"):
        import mxnet as mx
        ctx = mx.cpu()
    return ctx


def _prepare_src_model(args, ctx, use_cuda):
    return prepare_src_model(
        src_fwk=args.src_fwk,
        src_model=args.src_model,
        src_params_file_path=args.src_params,
        dst_fwk=args.dst_fwk,
        ctx=ctx,
        use_cuda=use_cuda,
        load_ignore_extra=args.load_ignore_extra,
        remove_module=args.remove_module,
        num_classes=args.src_num_classes,
        in_channels=args.src_in_channels)


def _prepare_dst_model(args, ctx, use_cuda):
    return prepare_dst_model(
        dst_fwk=args.dst_fwk,
        dst_model=args.dst_model,
        src_fwk=args.src_fwk,
        ctx=ctx,
        use_cuda=use_cuda,
        num_classes=args.dst_num_classes,
        in_channels=args.dst_in_channels)


def update_and_initialize_logging(args):
    """
    Update arguments ans initialize logging.

    Parameters
    ----------
    args : ArgumentParser
        Main script arguments.
    """
    packages = []
    pip_packages = []
    if (args.src_fwk == "gluon") or (args.dst_fwk == "gluon"):
        packages += ["mxnet, numpy"]
        pip_packages += ["mxnet-cu101"]
    if (args.src_fwk == "pytorch") or (args.dst_fwk == "pytorch"):
        packages += ["torch", "torchvision"]
    if (args.src_fwk == "chainer") or (args.dst_fwk == "chainer"):
        packages += ["chainer"]
        pip_packages += ["cupy-cuda101", "chainer"]
    if (args.src_fwk == "keras") or (args.dst_fwk == "keras"):
        packages += ["keras"]
        pip_packages += ["keras", "keras-mxnet", "mxnet-cu101"]
    if (args.src_fwk == "tensorflow") or (args.dst_fwk == "tensorflow"):
        packages += ["tensorflow-gpu"]
        pip_packages += ["tensorflow", "tensorflow-gpu", "tensorpack"]
    if (args.src_fwk == "tf2") or (args.dst_fwk == "tf2"):
        packages += ["tensorflow-gpu"]
        pip_packages += ["tensorflow", "tensorflow-gpu"]

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=packages,
        log_pip_packages=pip_packages)


def main():
    args = parse_args()

    ctx = None
    use_cuda = False

    if args.dst_fwk == "tf2":
        dst_params, dst_param_keys, dst_net = _prepare_dst_model(args, ctx, use_cuda)

    update_and_initialize_logging(args=args)

    ctx = _init_ctx(args)
    src_params, src_param_keys, ext_src_param_keys, ext_src_param_keys2 = _prepare_src_model(args, ctx, use_cuda)
    if args.dst_fwk != "tf2":
        dst_params, dst_param_keys, dst_net = _prepare_dst_model(args, ctx, use_cuda)

    if ((args.dst_fwk in ["keras", "tensorflow", "tf2"]) and any([s.find("convgroup") >= 0 for s in dst_param_keys]))\
            or ((args.src_fwk == "mxnet") and (args.src_model in ["crunet56", "crunet116", "preresnet269b"])):
        assert (len(src_param_keys) <= len(dst_param_keys))
    elif ((args.dst_fwk == "chainer") and
          (args.src_model.startswith("diaresnet") or args.src_model.startswith("diapreresnet"))) or\
            args.src_model.startswith("oth_ibppose"):
        assert (len(src_param_keys) >= len(dst_param_keys))
    else:
        assert (len(src_param_keys) == len(dst_param_keys))

    if args.src_fwk == "gluon" and args.dst_fwk == "gluon":
        convert_gl2gl(
            dst_net=dst_net,
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys,
            finetune=((args.src_num_classes != args.dst_num_classes) or (args.src_in_channels != args.dst_in_channels)),
            src_model=args.src_model,
            ctx=ctx)
    elif args.src_fwk == "pytorch" and args.dst_fwk == "pytorch":
        convert_pt2pt(
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys,
            src_model=args.src_model,
            dst_model=args.dst_model)
    elif args.src_fwk == "gluon" and args.dst_fwk == "pytorch":
        convert_gl2pt(
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys)
    elif args.src_fwk == "gluon" and args.dst_fwk == "chainer":
        convert_gl2ch(
            dst_net=dst_net,
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys,
            ext_src_param_keys=ext_src_param_keys,
            ext_src_param_keys2=ext_src_param_keys2,
            src_model=args.src_model)
    elif args.src_fwk == "gluon" and args.dst_fwk == "keras":
        convert_gl2ke(
            dst_net=dst_net,
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys)
    elif args.src_fwk == "gluon" and args.dst_fwk == "tensorflow":
        convert_gl2tf(
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys)
    elif args.src_fwk == "gluon" and args.dst_fwk == "tf2":
        convert_gl2tf2(
            dst_net=dst_net,
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys,
            src_model=args.src_model)
    elif args.src_fwk == "pytorch" and args.dst_fwk == "gluon":
        convert_pt2gl(
            dst_net=dst_net,
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys,
            ctx=ctx)
    elif args.src_fwk == "mxnet" and args.dst_fwk == "gluon":
        convert_mx2gl(
            dst_net=dst_net,
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys,
            src_model=args.src_model,
            ctx=ctx)
    elif args.src_fwk == "tensorflow" and args.dst_fwk == "tensorflow":
        convert_tf2tf(
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys)
    elif args.src_fwk == "tensorflow" and args.dst_fwk == "gluon":
        convert_tf2gl(
            dst_net=dst_net,
            dst_params_file_path=args.dst_params,
            dst_params=dst_params,
            dst_param_keys=dst_param_keys,
            src_params=src_params,
            src_param_keys=src_param_keys,
            ctx=ctx)
    else:
        raise NotImplementedError

    logging.info("Convert {}-model {} into {}-model {}".format(
        args.src_fwk, args.src_model, args.dst_fwk, args.dst_model))


if __name__ == '__main__':
    main()
