"""
    Routines for model statistics calculation.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from .pytorchcv.models.common import ChannelShuffle, ChannelShuffle2, Identity, Flatten, Swish, HSigmoid, HSwish,\
    InterpolationBlock, HeatmapMaxDetBlock
from .pytorchcv.models.fishnet import ChannelSqueeze
from .pytorchcv.models.irevnet import IRevDownscale, IRevSplitBlock, IRevMergeBlock
from .pytorchcv.models.rir_cifar import RiRFinalBlock
from .pytorchcv.models.proxylessnas import ProxylessUnit
from .pytorchcv.models.lwopenpose_cmupan import LwopDecoderFinalBlock
from .pytorchcv.models.centernet import CenterNetHeatmapMaxDet

__all__ = ['measure_model']


def calc_block_num_params2(net):
    """
    Calculate number of trainable parameters in the block (not iterative).

    Parameters
    ----------
    net : Module
        Model/block.

    Returns
    -------
    int
        Number of parameters.
    """
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def calc_block_num_params(module):
    """
    Calculate number of trainable parameters in the block (iterative).

    Parameters
    ----------
    module : Module
        Model/block.

    Returns
    -------
    int
        Number of parameters.
    """
    assert isinstance(module, nn.Module)
    net_params = filter(lambda p: isinstance(p[1], nn.parameter.Parameter) and p[1].requires_grad,
                        module._parameters.items())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param[1].size())
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
    # global names
    num_flops = 0
    num_macs = 0
    num_params = 0
    # names = {}

    def call_hook(module, x, y):
        if not (isinstance(module, IRevSplitBlock) or isinstance(module, IRevMergeBlock) or
                isinstance(module, RiRFinalBlock)):
            assert (len(x) == 1)
        assert (len(module._modules) == 0)
        if isinstance(module, nn.Linear):
            batch = x[0].shape[0]
            in_units = module.in_features
            out_units = module.out_features
            extra_num_macs = in_units * out_units
            if module.bias is None:
                extra_num_flops = (2 * in_units - 1) * out_units
            else:
                extra_num_flops = 2 * in_units * out_units
            extra_num_flops *= batch
            extra_num_macs *= batch
        elif isinstance(module, nn.ReLU):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.ELU):
            extra_num_flops = 3 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.Sigmoid):
            extra_num_flops = 4 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.LeakyReLU):
            extra_num_flops = 2 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.ReLU6):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.PReLU):
            extra_num_flops = 3 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, Swish):
            extra_num_flops = 5 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, HSigmoid):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, HSwish):
            extra_num_flops = 2 * x[0].numel()
            extra_num_macs = 0
        elif type(module) in [nn.ConvTranspose2d]:
            extra_num_flops = 4 * x[0].numel()
            extra_num_macs = 0
        elif type(module) in [nn.Conv2d]:
            batch = x[0].shape[0]
            x_h = x[0].shape[2]
            x_w = x[0].shape[3]
            kernel_size = module.kernel_size
            stride = module.stride
            dilation = module.dilation
            padding = module.padding
            groups = module.groups
            in_channels = module.in_channels
            out_channels = module.out_channels
            y_h = (x_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0] + 1
            y_w = (x_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1] + 1
            assert (out_channels == y.shape[1])
            assert (y_h == y.shape[2])
            assert (y_w == y.shape[3])
            kernel_total_size = kernel_size[0] * kernel_size[1]
            y_size = y_h * y_w
            extra_num_macs = kernel_total_size * in_channels * y_size * out_channels // groups
            if module.bias is None:
                extra_num_flops = (2 * kernel_total_size * y_size - 1) * in_channels * out_channels // groups
            else:
                extra_num_flops = 2 * kernel_total_size * in_channels * y_size * out_channels // groups
            extra_num_flops *= batch
            extra_num_macs *= batch
        elif isinstance(module, nn.BatchNorm2d):
            extra_num_flops = 4 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.InstanceNorm2d):
            extra_num_flops = 4 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.BatchNorm1d):
            extra_num_flops = 4 * x[0].numel()
            extra_num_macs = 0
        elif type(module) in [nn.MaxPool2d, nn.AvgPool2d]:
            assert (x[0].shape[1] == y.shape[1])
            batch = x[0].shape[0]
            kernel_size = module.kernel_size if isinstance(module.kernel_size, tuple) else\
                (module.kernel_size, module.kernel_size)
            y_h = y.shape[2]
            y_w = y.shape[3]
            channels = x[0].shape[1]
            y_size = y_h * y_w
            pool_total_size = kernel_size[0] * kernel_size[1]
            extra_num_flops = channels * y_size * pool_total_size
            extra_num_macs = 0
            extra_num_flops *= batch
            extra_num_macs *= batch
        elif type(module) in [nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d]:
            assert (x[0].shape[1] == y.shape[1])
            batch = x[0].shape[0]
            x_h = x[0].shape[2]
            x_w = x[0].shape[3]
            y_h = y.shape[2]
            y_w = y.shape[3]
            channels = x[0].shape[1]
            y_size = y_h * y_w
            pool_total_size = x_h * x_w
            extra_num_flops = channels * y_size * pool_total_size
            extra_num_macs = 0
            extra_num_flops *= batch
            extra_num_macs *= batch
        elif isinstance(module, nn.Dropout):
            extra_num_flops = 0
            extra_num_macs = 0
        elif isinstance(module, nn.Sequential):
            assert (len(module._modules) == 0)
            extra_num_flops = 0
            extra_num_macs = 0
        elif type(module) in [ChannelShuffle, ChannelShuffle2]:
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.ZeroPad2d):
            extra_num_flops = 0
            extra_num_macs = 0
        elif isinstance(module, Identity):
            extra_num_flops = 0
            extra_num_macs = 0
        elif isinstance(module, nn.PixelShuffle):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, Flatten):
            extra_num_flops = 0
            extra_num_macs = 0
        elif isinstance(module, nn.Upsample):
            extra_num_flops = 4 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, ChannelSqueeze):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, IRevDownscale):
            extra_num_flops = 5 * x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, IRevSplitBlock):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, IRevMergeBlock):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, RiRFinalBlock):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, ProxylessUnit):
            extra_num_flops = x[0].numel()
            extra_num_macs = 0
        elif isinstance(module, nn.Softmax2d):
            extra_num_flops = 4 * x[0].numel()
            extra_num_macs = 0
        elif type(module) in [InterpolationBlock, HeatmapMaxDetBlock, CenterNetHeatmapMaxDet]:
            extra_num_flops, extra_num_macs = module.calc_flops(x[0])
        elif isinstance(module, LwopDecoderFinalBlock):
            if not module.calc_3d_features:
                extra_num_flops = 0
                extra_num_macs = 0
            else:
                raise TypeError("LwopDecoderFinalBlock!")
        else:
            raise TypeError("Unknown layer type: {}".format(type(module)))

        global num_flops
        global num_macs
        global num_params
        # global names
        num_flops += extra_num_flops
        num_macs += extra_num_macs
        # if module.name not in names:
        #     names[module.name] = 1
        #     num_params += calc_block_num_params(module)
        num_params += calc_block_num_params(module)

    def register_forward_hooks(a_module):
        if len(a_module._modules) > 0:
            assert (calc_block_num_params(a_module) == 0)
            children_handles = []
            for child_module in a_module._modules.values():
                child_handles = register_forward_hooks(child_module)
                children_handles += child_handles
            return children_handles
        else:
            handle = a_module.register_forward_hook(call_hook)
            return [handle]

    hook_handles = register_forward_hooks(model)

    x = Variable(torch.zeros(1, in_channels, in_size[0], in_size[1]))
    model.eval()
    model(x)

    num_params1 = calc_block_num_params2(model)
    if num_params != num_params1:
        logging.warning(
            "Calculated numbers of parameters are different: standard method: {},\tper-leaf method: {}".format(
                num_params1, num_params))

    [h.remove() for h in hook_handles]

    return num_flops, num_macs, num_params1
