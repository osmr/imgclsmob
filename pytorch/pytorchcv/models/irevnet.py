"""
    i-RevNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'i-RevNet: Deep Invertible Networks,' https://arxiv.org/abs/1802.07088.
"""

__all__ = ['IRevNet', 'irevnet301', 'IRevDownscale', 'IRevSplitBlock', 'IRevMergeBlock']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv3x3, pre_conv3x3_block, DualPathSequential


class IRevDualPathSequential(DualPathSequential):
    """
    An invertible sequential container for modules with dual inputs/outputs.
    Modules will be executed in the order they are added.

    Parameters:
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first modules with single input/output.
    last_ordinals : int, default 0
        Number of the final modules with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a module.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal module.
    last_noninvertible : int, default 0
        Number of the final modules skipped during inverse.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda module, x1, x2: module(x1, x2)),
                 dual_path_scheme_ordinal=(lambda module, x1, x2: (module(x1), x2)),
                 last_noninvertible=0):
        super(IRevDualPathSequential, self).__init__(
            return_two=return_two,
            first_ordinals=first_ordinals,
            last_ordinals=last_ordinals,
            dual_path_scheme=dual_path_scheme,
            dual_path_scheme_ordinal=dual_path_scheme_ordinal)
        self.last_noninvertible = last_noninvertible

    def inverse(self, x1, x2=None):
        length = len(self._modules.values())
        for i, module in enumerate(reversed(self._modules.values())):
            if i < self.last_noninvertible:
                pass
            elif (i < self.last_ordinals) or (i >= length - self.first_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(module.inverse, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(module.inverse, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class IRevDownscale(nn.Module):
    """
    i-RevNet specific downscale (so-called psi-block).

    Parameters:
    ----------
    scale : int
        Scale (downscale) value.
    """
    def __init__(self, scale):
        super(IRevDownscale, self).__init__()
        self.scale = scale

    def forward(self, x):
        batch, x_channels, x_height, x_width = x.size()
        y_channels = x_channels * self.scale * self.scale
        assert (x_height % self.scale == 0)
        y_height = x_height // self.scale

        y = x.permute(0, 2, 3, 1)
        d2_split_seq = y.split(split_size=self.scale, dim=2)
        d2_split_seq = [t.contiguous().view(batch, y_height, y_channels) for t in d2_split_seq]
        y = torch.stack(d2_split_seq, dim=1)
        y = y.permute(0, 3, 2, 1)
        return y.contiguous()

    def inverse(self, y):
        scale_sqr = self.scale * self.scale
        batch, y_channels, y_height, y_width = y.size()
        assert (y_channels % scale_sqr == 0)
        x_channels = y_channels // scale_sqr
        x_height = y_height * self.scale
        x_width = y_width * self.scale

        x = y.permute(0, 2, 3, 1)
        x = x.contiguous().view(batch, y_height, y_width, scale_sqr, x_channels)
        d3_split_seq = x.split(split_size=self.scale, dim=3)
        d3_split_seq = [t.contiguous().view(batch, y_height, x_width, x_channels) for t in d3_split_seq]
        x = torch.stack(d3_split_seq, dim=0)
        x = x.transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch, x_height, x_width, x_channels)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class IRevInjectivePad(nn.Module):
    """
    i-RevNet channel zero padding block.

    Parameters:
    ----------
    padding : int
        Size of the padding.
    """
    def __init__(self, padding):
        super(IRevInjectivePad, self).__init__()
        self.padding = padding
        self.pad = nn.ZeroPad2d(padding=(0, 0, 0, padding))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.padding, :, :]


class IRevSplitBlock(nn.Module):
    """
    iRevNet split block.
    """
    def __init__(self):
        super(IRevSplitBlock, self).__init__()

    def forward(self, x, _):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        return x1, x2

    def inverse(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return x, None


class IRevMergeBlock(nn.Module):
    """
    iRevNet merge block.
    """
    def __init__(self):
        super(IRevMergeBlock, self).__init__()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        return x, x

    def inverse(self, x, _):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        return x1, x2


class IRevBottleneck(nn.Module):
    """
    iRevNet bottleneck block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    preactivate : bool
        Whether use pre-activation for the first convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 preactivate):
        super(IRevBottleneck, self).__init__()
        mid_channels = out_channels // 4

        if preactivate:
            self.conv1 = pre_conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride)
        else:
            self.conv1 = conv3x3(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.conv3 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class IRevUnit(nn.Module):
    """
    iRevNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    preactivate : bool
        Whether use pre-activation for the first convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 preactivate):
        super(IRevUnit, self).__init__()
        if not preactivate:
            in_channels = in_channels // 2

        padding = 2 * (out_channels - in_channels)
        self.do_padding = (padding != 0) and (stride == 1)
        self.do_downscale = (stride != 1)

        if self.do_padding:
            self.pad = IRevInjectivePad(padding)
        self.bottleneck = IRevBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            preactivate=preactivate)
        if self.do_downscale:
            self.psi = IRevDownscale(stride)

    def forward(self, x1, x2):
        if self.do_padding:
            x = torch.cat((x1, x2), dim=1)
            x = self.pad(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
        fx2 = self.bottleneck(x2)
        if self.do_downscale:
            x1 = self.psi(x1)
            x2 = self.psi(x2)
        y1 = fx2 + x1
        return x2, y1

    def inverse(self, x2, y1):
        if self.do_downscale:
            x2 = self.psi.inverse(x2)
        fx2 = - self.bottleneck(x2)
        x1 = fx2 + y1
        if self.do_downscale:
            x1 = self.psi.inverse(x1)
        if self.do_padding:
            x = torch.cat((x1, x2), dim=1)
            x = self.pad.inverse(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
        return x1, x2


class IRevPostActivation(nn.Module):
    """
    iRevNet specific post-activation block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(IRevPostActivation, self).__init__()
        self.bn = nn.BatchNorm2d(
            num_features=in_channels,
            momentum=0.9)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class IRevNet(nn.Module):
    """
    i-RevNet model from 'i-RevNet: Deep Invertible Networks,' https://arxiv.org/abs/1802.07088.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(IRevNet, self).__init__()
        assert (in_channels > 0)
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = IRevDualPathSequential(
            first_ordinals=1,
            last_ordinals=2,
            last_noninvertible=2)
        self.features.add_module("init_block", IRevDownscale(scale=2))
        in_channels = init_block_channels
        self.features.add_module("init_split", IRevSplitBlock())
        for i, channels_per_stage in enumerate(channels):
            stage = IRevDualPathSequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) else 1
                preactivate = not ((i == 0) and (j == 0))
                stage.add_module("unit{}".format(j + 1), IRevUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    preactivate=preactivate))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        in_channels = final_block_channels
        self.features.add_module("final_merge", IRevMergeBlock())
        self.features.add_module("final_postactiv", IRevPostActivation(in_channels=in_channels))
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x, return_out_bij=False):
        x, out_bij = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        if return_out_bij:
            return x, out_bij
        else:
            return x

    def inverse(self, out_bij):
        x, _ = self.features.inverse(out_bij)
        return x


def get_irevnet(blocks,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create i-RevNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if blocks == 301:
        layers = [6, 16, 72, 6]
    else:
        raise ValueError("Unsupported i-RevNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 1 == blocks)

    channels_per_layers = [24, 96, 384, 1536]
    init_block_channels = 12
    final_block_channels = 3072

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = IRevNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def irevnet301(**kwargs):
    """
    i-RevNet-301 model from 'i-RevNet: Deep Invertible Networks,' https://arxiv.org/abs/1802.07088.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_irevnet(blocks=301, model_name="irevnet301", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        irevnet301,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != irevnet301 or weight_count == 125120356)

        x = torch.randn(2, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (2, 1000))

        y, out_bij = net(x, return_out_bij=True)
        x_ = net.inverse(out_bij)
        assert (tuple(x_.size()) == (2, 3, 224, 224))

        import numpy as np
        assert (np.max(np.abs(x.detach().numpy() - x_.detach().numpy())) < 1e-4)


if __name__ == "__main__":
    _test()
