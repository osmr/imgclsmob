import os
import re
import logging
import numpy as np
import mxnet as mx
from .gluoncv2.model_provider import get_model


def prepare_mx_context(num_gpus,
                       batch_size):
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
    batch_size *= max(1, num_gpus)
    return ctx, batch_size


def prepare_model(model_name,
                  use_pretrained,
                  pretrained_model_file_path,
                  dtype,
                  tune_layers,
                  classes=None,
                  in_channels=None,
                  do_hybridize=True,
                  ctx=mx.cpu()):
    kwargs = {'ctx': ctx,
              'pretrained': use_pretrained}
    if classes is not None:
        kwargs["classes"] = classes
    if in_channels is not None:
        kwargs["in_channels"] = in_channels

    net = get_model(model_name, **kwargs)

    if pretrained_model_file_path:
        assert (os.path.isfile(pretrained_model_file_path))
        logging.info('Loading model: {}'.format(pretrained_model_file_path))
        net.load_parameters(
            filename=pretrained_model_file_path,
            ctx=ctx)

    net.cast(dtype)

    if do_hybridize:
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

    if tune_layers:
        tune_layers_pattern = re.compile(tune_layers)
        for k, v in net._collect_params_with_prefix().items():
            if tune_layers_pattern.match(k):
                logging.info('Fine-tune parameter: {}'.format(k))
            else:
                v.grad_req = 'null'
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    return net


def calc_net_weight_count(net):
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
        # if np.prod(param.shape) > 0:
        #     print("name={}, d_weight_count={}".format(param.name, np.prod(param.shape)))
    return weight_count


def validate(acc_top1,
             acc_top5,
             net,
             val_data,
             batch_fn,
             data_source_needs_reset,
             dtype,
             ctx):
    if data_source_needs_reset:
        val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for batch in val_data:
        data_list, labels_list = batch_fn(batch, ctx)
        outputs_list = [net(X.astype(dtype, copy=False)) for X in data_list]
        acc_top1.update(labels_list, outputs_list)
        acc_top5.update(labels_list, outputs_list)
    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return 1.0 - top1, 1.0 - top5
