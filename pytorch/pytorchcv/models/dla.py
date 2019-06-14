"""
    DLA for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.
"""

__all__ = ['DLA', 'dla34', 'dla46c', 'dla46xc', 'dla60', 'dla60x', 'dla60xc', 'dla102', 'dla102x', 'dla102x2', 'dla169']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1, conv1x1_block, conv3x3_block, conv7x7_block
from .resnet import ResBlock, ResBottleneck
from .resnext import ResNeXtBottleneck


class DLABottleneck(ResBottleneck):
    """
    DLA bottleneck block for residual path in residual block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck_factor=2):
        super(DLABottleneck, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bottleneck_factor=bottleneck_factor)


class DLABottleneckX(ResNeXtBottleneck):
    """
    DLA ResNeXt-like bottleneck block for residual path in residual block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int, default 32
        Number of groups.
    bottleneck_width: int, default 8
        Width of bottleneck block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 cardinality=32,
                 bottleneck_width=8):
        super(DLABottleneckX, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width)


class DLAResBlock(nn.Module):
    """
    DLA residual block with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    body_class : nn.Module, default ResBlock
        Residual block body class.
    return_down : bool, default False
        Whether return downsample result.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 body_class=ResBlock,
                 return_down=False):
        super(DLAResBlock, self).__init__()
        self.return_down = return_down
        self.downsample = (stride > 1)
        self.project = (in_channels != out_channels)

        self.body = body_class(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        self.activ = nn.ReLU(inplace=True)
        if self.downsample:
            self.downsample_pool = nn.MaxPool2d(
                kernel_size=stride,
                stride=stride)
        if self.project:
            self.project_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None)

    def forward(self, x):
        down = self.downsample_pool(x) if self.downsample else x
        identity = self.project_conv(down) if self.project else down
        if identity is None:
            identity = x
        x = self.body(x)
        x += identity
        x = self.activ(x)
        if self.return_down:
            return x, down
        else:
            return x


class DLARoot(nn.Module):
    """
    DLA root block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    residual : bool
        Whether use residual connection.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 residual):
        super(DLARoot, self).__init__()
        self.residual = residual

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x2, x1, extra):
        last_branch = x2
        x = torch.cat((x2, x1) + tuple(extra), dim=1)
        x = self.conv(x)
        if self.residual:
            x += last_branch
        x = self.activ(x)
        return x


class DLATree(nn.Module):
    """
    DLA tree unit. It's like iterative stage.

    Parameters:
    ----------
    levels : int
        Number of levels in the stage.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    res_body_class : nn.Module
        Residual block body class.
    stride : int or tuple/list of 2 int
        Strides of the convolution in a residual block.
    root_residual : bool
        Whether use residual connection in the root.
    root_dim : int
        Number of input channels in the root block.
    first_tree : bool, default False
        Is this tree stage the first stage in the net.
    input_level : bool, default True
        Is this tree unit the first unit in the stage.
    return_down : bool, default False
        Whether return downsample result.
    """
    def __init__(self,
                 levels,
                 in_channels,
                 out_channels,
                 res_body_class,
                 stride,
                 root_residual,
                 root_dim=0,
                 first_tree=False,
                 input_level=True,
                 return_down=False):
        super(DLATree, self).__init__()
        self.return_down = return_down
        self.add_down = (input_level and not first_tree)
        self.root_level = (levels == 1)

        if root_dim == 0:
            root_dim = 2 * out_channels
        if self.add_down:
            root_dim += in_channels

        if self.root_level:
            self.tree1 = DLAResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                body_class=res_body_class,
                return_down=True)
            self.tree2 = DLAResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                body_class=res_body_class,
                return_down=False)
        else:
            self.tree1 = DLATree(
                levels=levels - 1,
                in_channels=in_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                stride=stride,
                root_residual=root_residual,
                root_dim=0,
                input_level=False,
                return_down=True)
            self.tree2 = DLATree(
                levels=levels - 1,
                in_channels=out_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                stride=1,
                root_residual=root_residual,
                root_dim=root_dim + out_channels,
                input_level=False,
                return_down=False)
        if self.root_level:
            self.root = DLARoot(
                in_channels=root_dim,
                out_channels=out_channels,
                residual=root_residual)

    def forward(self, x, extra=None):
        extra = [] if extra is None else extra
        x1, down = self.tree1(x)
        if self.add_down:
            extra.append(down)
        if self.root_level:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, extra)
        else:
            extra.append(x1)
            x = self.tree2(x1, extra)
        if self.return_down:
            return x, down
        else:
            return x


class DLAInitBlock(nn.Module):
    """
    DLA specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DLAInitBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = conv7x7_block(
            in_channels=in_channels,
            out_channels=mid_channels)
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels)
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DLA(nn.Module):
    """
    DLA model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    levels : int
        Number of levels in each stage.
    channels : list of int
        Number of output channels for each stage.
    init_block_channels : int
        Number of output channels for the initial unit.
    res_body_class : nn.Module
        Residual block body class.
    residual_root : bool
        Whether use residual connection in the root blocks.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 levels,
                 channels,
                 init_block_channels,
                 res_body_class,
                 residual_root,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(DLA, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", DLAInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels

        for i in range(len(levels)):
            levels_i = levels[i]
            out_channels = channels[i]
            first_tree = (i == 0)
            self.features.add_module("stage{}".format(i + 1), DLATree(
                levels=levels_i,
                in_channels=in_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                stride=2,
                root_residual=residual_root,
                first_tree=first_tree))
            in_channels = out_channels

        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = conv1x1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        x = x.view(x.size(0), -1)
        return x


def get_dla(levels,
            channels,
            res_body_class,
            residual_root=False,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".torch", "models"),
            **kwargs):
    """
    Create DLA model with specific parameters.

    Parameters:
    ----------
    levels : int
        Number of levels in each stage.
    channels : list of int
        Number of output channels for each stage.
    res_body_class : nn.Module
        Residual block body class.
    residual_root : bool, default False
        Whether use residual connection in the root blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32

    net = DLA(
        levels=levels,
        channels=channels,
        init_block_channels=init_block_channels,
        res_body_class=res_body_class,
        residual_root=residual_root,
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


def dla34(**kwargs):
    """
    DLA-34 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 2, 1], channels=[64, 128, 256, 512], res_body_class=ResBlock, model_name="dla34",
                   **kwargs)


def dla46c(**kwargs):
    """
    DLA-46-C model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 2, 1], channels=[64, 64, 128, 256], res_body_class=DLABottleneck, model_name="dla46c",
                   **kwargs)


def dla46xc(**kwargs):
    """
    DLA-X-46-C model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 2, 1], channels=[64, 64, 128, 256], res_body_class=DLABottleneckX,
                   model_name="dla46xc", **kwargs)


def dla60(**kwargs):
    """
    DLA-60 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 3, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneck,
                   model_name="dla60", **kwargs)


def dla60x(**kwargs):
    """
    DLA-X-60 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 3, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneckX,
                   model_name="dla60x", **kwargs)


def dla60xc(**kwargs):
    """
    DLA-X-60-C model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 3, 1], channels=[64, 64, 128, 256], res_body_class=DLABottleneckX,
                   model_name="dla60xc", **kwargs)


def dla102(**kwargs):
    """
    DLA-102 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 3, 4, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneck,
                   residual_root=True, model_name="dla102", **kwargs)


def dla102x(**kwargs):
    """
    DLA-X-102 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 3, 4, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneckX,
                   residual_root=True, model_name="dla102x", **kwargs)


def dla102x2(**kwargs):
    """
    DLA-X2-102 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    class DLABottleneckX64(DLABottleneckX):
        def __init__(self, in_channels, out_channels, stride):
            super(DLABottleneckX64, self).__init__(in_channels, out_channels, stride, cardinality=64)

    return get_dla(levels=[1, 3, 4, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneckX64,
                   residual_root=True, model_name="dla102x2", **kwargs)


def dla169(**kwargs):
    """
    DLA-169 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[2, 3, 5, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneck,
                   residual_root=True, model_name="dla169", **kwargs)


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
        dla34,
        dla46c,
        dla46xc,
        dla60,
        dla60x,
        dla60xc,
        dla102,
        dla102x,
        dla102x2,
        dla169,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dla34 or weight_count == 15742104)
        assert (model != dla46c or weight_count == 1301400)
        assert (model != dla46xc or weight_count == 1068440)
        assert (model != dla60 or weight_count == 22036632)
        assert (model != dla60x or weight_count == 17352344)
        assert (model != dla60xc or weight_count == 1319832)
        assert (model != dla102 or weight_count == 33268888)
        assert (model != dla102x or weight_count == 26309272)
        assert (model != dla102x2 or weight_count == 41282200)
        assert (model != dla169 or weight_count == 53389720)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
