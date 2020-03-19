"""
    Routines for model statistics calculation.
"""

import logging
import numpy as np
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import Identity, PixelShuffle2D
from .gluoncv2.models.common import ReLU6, ChannelShuffle, ChannelShuffle2, PReLU2, HSigmoid, HSwish,\
    InterpolationBlock, HeatmapMaxDetBlock
from .gluoncv2.models.fishnet import ChannelSqueeze
from .gluoncv2.models.irevnet import IRevDownscale, IRevSplitBlock, IRevMergeBlock
from .gluoncv2.models.rir_cifar import RiRFinalBlock
from .gluoncv2.models.proxylessnas import ProxylessUnit
from .gluoncv2.models.lwopenpose_cmupan import LwopDecoderFinalBlock
from .gluoncv2.models.centernet import CenterNetHeatmapMaxDet

__all__ = ['measure_model']


def calc_block_num_params2(net):
    """
    Calculate number of trainable parameters in the block (not iterative).

    Parameters
    ----------
    net : Block
        Model/block.

    Returns
    -------
    int
        Number of parameters.
    """
    net_params = net.collect_params()
    weight_count = 0
    for param in net_params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def calc_block_num_params(block):
    """
    Calculate number of trainable parameters in the block (iterative).

    Parameters
    ----------
    block : Block
        Model/block.

    Returns
    -------
    int
        Number of parameters.
    """
    weight_count = 0
    for param in block.params.values():
        if (param.shape is None) or (not param._differentiable):
            continue
        weight_count += np.prod(param.shape)
    return weight_count


def measure_model(model,
                  in_channels,
                  in_size,
                  ctx=mx.cpu()):
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    global num_flops
    global num_macs
    global num_params
    global names
    num_flops = 0
    num_macs = 0
    num_params = 0
    names = {}

    def call_hook(block, x, y):
        if not (isinstance(block, IRevSplitBlock) or isinstance(block, IRevMergeBlock) or
                isinstance(block, RiRFinalBlock)):
            assert (len(x) == 1)
        assert (len(block._children) == 0)
        if isinstance(block, nn.Dense):
            batch = x[0].shape[0]
            in_units = block._in_units
            out_units = block._units
            extra_num_macs = in_units * out_units
            if block.bias is None:
                extra_num_flops = (2 * in_units - 1) * out_units
            else:
                extra_num_flops = 2 * in_units * out_units
            extra_num_flops *= batch
            extra_num_macs *= batch
        elif isinstance(block, nn.Activation):
            if block._act_type == "relu":
                extra_num_flops = x[0].size
                extra_num_macs = 0
            elif block._act_type == "sigmoid":
                extra_num_flops = 4 * x[0].size
                extra_num_macs = 0
            else:
                raise TypeError("Unknown activation type: {}".format(block._act_type))
        elif isinstance(block, nn.ELU):
            extra_num_flops = 3 * x[0].size
            extra_num_macs = 0
        elif isinstance(block, nn.LeakyReLU):
            extra_num_flops = 2 * x[0].size
            extra_num_macs = 0
        elif isinstance(block, ReLU6):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, PReLU2):
            extra_num_flops = 3 * x[0].size
            extra_num_macs = 0
        elif isinstance(block, nn.Swish):
            extra_num_flops = 5 * x[0].size
            extra_num_macs = 0
        elif isinstance(block, HSigmoid):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, HSwish):
            extra_num_flops = 2 * x[0].size
            extra_num_macs = 0
        elif type(block) in [nn.Conv2DTranspose]:
            extra_num_flops = 4 * x[0].size
            extra_num_macs = 0
        elif isinstance(block, nn.Conv2D):
            batch = x[0].shape[0]
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
            assert (out_channels == y.shape[1])
            assert (y_h == y.shape[2])
            assert (y_w == y.shape[3])
            kernel_total_size = kernel_size[0] * kernel_size[1]
            y_size = y_h * y_w
            extra_num_macs = kernel_total_size * in_channels * y_size * out_channels // groups
            if block.bias is None:
                extra_num_flops = (2 * kernel_total_size * y_size - 1) * in_channels * out_channels // groups
            else:
                extra_num_flops = 2 * kernel_total_size * in_channels * y_size * out_channels // groups
            extra_num_flops *= batch
            extra_num_macs *= batch
        elif isinstance(block, nn.BatchNorm):
            extra_num_flops = 4 * x[0].size
            extra_num_macs = 0
        elif isinstance(block, nn.InstanceNorm):
            extra_num_flops = 4 * x[0].size
            extra_num_macs = 0
        elif type(block) in [nn.MaxPool2D, nn.AvgPool2D, nn.GlobalAvgPool2D, nn.GlobalMaxPool2D]:
            batch = x[0].shape[0]
            assert (x[0].shape[1] == y.shape[1])
            pool_size = block._kwargs["kernel"]
            y_h = y.shape[2]
            y_w = y.shape[3]
            channels = x[0].shape[1]
            y_size = y_h * y_w
            pool_total_size = pool_size[0] * pool_size[1]
            extra_num_flops = channels * y_size * pool_total_size
            extra_num_macs = 0
            extra_num_flops *= batch
            extra_num_macs *= batch
        elif isinstance(block, nn.Dropout):
            extra_num_flops = 0
            extra_num_macs = 0
        elif type(block) in [nn.Flatten]:
            extra_num_flops = 0
            extra_num_macs = 0
        elif isinstance(block, nn.HybridSequential):
            assert (len(block._children) == 0)
            extra_num_flops = 0
            extra_num_macs = 0
        elif type(block) in [ChannelShuffle, ChannelShuffle2]:
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, Identity):
            extra_num_flops = 0
            extra_num_macs = 0
        elif isinstance(block, PixelShuffle2D):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, ChannelSqueeze):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, IRevDownscale):
            extra_num_flops = 5 * x[0].size
            extra_num_macs = 0
        elif isinstance(block, IRevSplitBlock):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, IRevMergeBlock):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, RiRFinalBlock):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif isinstance(block, ProxylessUnit):
            extra_num_flops = x[0].size
            extra_num_macs = 0
        elif type(block) in [InterpolationBlock, HeatmapMaxDetBlock, CenterNetHeatmapMaxDet]:
            extra_num_flops, extra_num_macs = block.calc_flops(x[0])
        elif isinstance(block, LwopDecoderFinalBlock):
            if not block.calc_3d_features:
                extra_num_flops = 0
                extra_num_macs = 0
            else:
                raise TypeError("LwopDecoderFinalBlock!")
        else:
            raise TypeError("Unknown layer type: {}".format(type(block)))

        global num_flops
        global num_macs
        global num_params
        global names
        num_flops += extra_num_flops
        num_macs += extra_num_macs
        if block.name not in names:
            names[block.name] = 1
            num_params += calc_block_num_params(block)

    def register_forward_hooks(a_block):
        if len(a_block._children) > 0:
            assert (calc_block_num_params(a_block) == 0)
            children_handles = []
            for child_block in a_block._children.values():
                child_handles = register_forward_hooks(child_block)
                children_handles += child_handles
            return children_handles
        else:
            handle = a_block.register_forward_hook(call_hook)
            return [handle]

    hook_handles = register_forward_hooks(model)

    x = mx.nd.zeros((1, in_channels, in_size[0], in_size[1]), ctx=ctx)
    model(x)

    num_params1 = calc_block_num_params2(model)
    if num_params != num_params1:
        logging.warning(
            "Calculated numbers of parameters are different: standard method: {},\tper-leaf method: {}".format(
                num_params1, num_params))

    [h.detach() for h in hook_handles]

    return num_flops, num_macs, num_params1
