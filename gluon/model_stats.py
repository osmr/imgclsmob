import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from .gluoncv2.models.common import ReLU6

__all__ = ['measure_model']


def calc_block_num_params(block):
    weight_count = 0
    for param in block.params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def measure_model(model,
                  in_channels,
                  in_size):
    """
    Calculate model statistics.

    Parameters:
    ----------
    model : HybridBlock
        Tested model.
    in_channels : int
        Number of input channels.
    in_size : tuple of two ints
        Spatial size of the expected input image.
    """
    global num_flops
    global num_macs
    global num_params
    num_flops = 0
    num_macs = 0
    num_params = 0

    def call_hook(block, x, y):
        assert (x[0].shape[0] == 1)
        if isinstance(block, nn.Dense):
            in_units = block._in_units
            out_units = block._units
            extra_num_macs = in_units * out_units
            if block.bias is None:
                extra_num_flops = (2 * in_units - 1) * out_units
            else:
                extra_num_flops = 2 * in_units * out_units
        elif isinstance(block, nn.Activation):
            if block._act_type == "relu":
                extra_num_flops = x[0].size
                extra_num_macs = 0
            elif block._act_type == "sigmoid":
                extra_num_flops = 4 * x[0].size
                extra_num_macs = 0
            else:
                raise TypeError('Unknown activation type: {}'.format(block._act_type))
        elif isinstance(block, ReLU6):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, nn.Conv2D):
            x_h = x[0].shape[2]
            x_w = x[0].shape[3]
            kernel_size = block._kwargs["kernel"]
            strides = block._kwargs["stride"]
            dilation = block._kwargs["dilate"]
            padding = block._kwargs["pad"]
            groups = block._kwargs["num_group"]
            in_channels = block._in_channels
            out_channels = block._channels
            y_h = (x_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // strides[0] + 1
            y_w = (x_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // strides[1] + 1
            assert (y_h == y.shape[2])
            assert (y_w == y.shape[3])
            kernel_total_size = kernel_size[0] * kernel_size[1]
            y_size = y_h * y_w
            extra_num_macs = kernel_total_size * y_size * in_channels * out_channels // groups
            if block.bias is None:
                extra_num_flops = (2 * kernel_total_size - 1) * y_size * in_channels * out_channels // groups
            else:
                extra_num_flops = 2 * kernel_total_size * y_size * in_channels * out_channels // groups
        elif isinstance(block, nn.BatchNorm):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif type(block) in [nn.MaxPool2D, nn.AvgPool2D]:
            x_h = x[0].shape[2]
            x_w = x[0].shape[3]
            pool_size = block._kwargs["kernel"]
            strides = block._kwargs["stride"]
            padding = block._kwargs["pad"]
            y_h = (x_h + 2 * padding[0] - pool_size[0]) // strides[0] + 1
            y_w = (x_w + 2 * padding[1] - pool_size[1]) // strides[1] + 1
            assert (y_h == y.shape[2])
            assert (y_w == y.shape[3])
            extra_num_flops = (x_h * x_w) * (y_h * y_w) * (pool_size[0] * pool_size[1])
            extra_num_macs = 0
        elif type(block) in [nn.Flatten]:
            extra_num_flops = 0
            extra_num_macs = 0
        else:
            raise TypeError('Unknown layer type: {}'.format(type(block)))

        global num_flops
        global num_macs
        global num_params
        num_flops += extra_num_flops
        num_macs += extra_num_macs
        num_params += calc_block_num_params(block)

    def register_forward_hooks(a_block):
        if len(a_block._children) > 0:
            children_handles = []
            for child_block in a_block._children.values():
                child_handles = register_forward_hooks(child_block)
                children_handles += child_handles
            return children_handles
        else:
            handle = a_block.register_forward_hook(call_hook)
            return [handle]

    hook_handles = register_forward_hooks(model)

    ctx = mx.cpu()
    x = mx.nd.zeros((1, in_channels, in_size[0], in_size[1]), ctx=ctx)
    model(x)

    [h.detach() for h in hook_handles]

    return num_flops, num_macs, num_params
