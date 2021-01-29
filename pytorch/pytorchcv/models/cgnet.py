"""
    CGNet for image segmentation, implemented in PyTorch.
    Original paper: 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.
"""

__all__ = ['CGNet', 'cgnet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import NormActivation, conv1x1, conv1x1_block, conv3x3_block, depthwise_conv3x3, SEBlock, Concurrent,\
    DualPathSequential, InterpolationBlock


class CGBlock(nn.Module):
    """
    CGNet block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilation : int
        Dilation value.
    se_reduction : int
        SE-block reduction value.
    down : bool
        Whether to downsample.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilation,
                 se_reduction,
                 down,
                 bn_eps):
        super(CGBlock, self).__init__()
        self.down = down
        if self.down:
            mid1_channels = out_channels
            mid2_channels = 2 * out_channels
        else:
            mid1_channels = out_channels // 2
            mid2_channels = out_channels

        if self.down:
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2,
                bn_eps=bn_eps,
                activation=(lambda: nn.PReLU(out_channels)))
        else:
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid1_channels,
                bn_eps=bn_eps,
                activation=(lambda: nn.PReLU(mid1_channels)))

        self.branches = Concurrent()
        self.branches.add_module("branches1", depthwise_conv3x3(channels=mid1_channels))
        self.branches.add_module("branches2", depthwise_conv3x3(
            channels=mid1_channels,
            padding=dilation,
            dilation=dilation))

        self.norm_activ = NormActivation(
            in_channels=mid2_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(mid2_channels)))

        if self.down:
            self.conv2 = conv1x1(
                in_channels=mid2_channels,
                out_channels=out_channels)

        self.se = SEBlock(
            channels=out_channels,
            reduction=se_reduction,
            use_conv=False)

    def forward(self, x):
        if not self.down:
            identity = x
        x = self.conv1(x)
        x = self.branches(x)
        x = self.norm_activ(x)
        if self.down:
            x = self.conv2(x)
        x = self.se(x)
        if not self.down:
            x += identity
        return x


class CGUnit(nn.Module):
    """
    CGNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    layers : int
        Number of layers.
    dilation : int
        Dilation value.
    se_reduction : int
        SE-block reduction value.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 layers,
                 dilation,
                 se_reduction,
                 bn_eps):
        super(CGUnit, self).__init__()
        mid_channels = out_channels // 2

        self.down = CGBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            dilation=dilation,
            se_reduction=se_reduction,
            down=True,
            bn_eps=bn_eps)
        self.blocks = nn.Sequential()
        for i in range(layers - 1):
            self.blocks.add_module("block{}".format(i + 1), CGBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                dilation=dilation,
                se_reduction=se_reduction,
                down=False,
                bn_eps=bn_eps))

    def forward(self, x):
        x = self.down(x)
        y = self.blocks(x)
        x = torch.cat((y, x), dim=1)  # NB: This differs from the original implementation.
        return x


class CGStage(nn.Module):
    """
    CGNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    layers : int
        Number of layers in the unit.
    dilation : int
        Dilation for blocks.
    se_reduction : int
        SE-block reduction value for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 layers,
                 dilation,
                 se_reduction,
                 bn_eps):
        super(CGStage, self).__init__()
        self.use_x = (x_channels > 0)
        self.use_unit = (layers > 0)

        if self.use_x:
            self.x_down = nn.AvgPool2d(
                kernel_size=3,
                stride=2,
                padding=1)

        if self.use_unit:
            self.unit = CGUnit(
                in_channels=y_in_channels,
                out_channels=(y_out_channels - x_channels),
                layers=layers,
                dilation=dilation,
                se_reduction=se_reduction,
                bn_eps=bn_eps)

        self.norm_activ = NormActivation(
            in_channels=y_out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(y_out_channels)))

    def forward(self, y, x=None):
        if self.use_unit:
            y = self.unit(y)
        if self.use_x:
            x = self.x_down(x)
            y = torch.cat((y, x), dim=1)
        y = self.norm_activ(y)
        return y, x


class CGInitBlock(nn.Module):
    """
    CGNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_eps):
        super(CGInitBlock, self).__init__()
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))
        self.conv2 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CGNet(nn.Module):
    """
    CGNet model from 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.

    Parameters:
    ----------
    layers : list of int
        Number of layers for each unit.
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of int
        Dilations for each unit.
    se_reductions : list of int
        SE-block reduction value for each unit.
    cut_x : list of int
        Whether to concatenate with x-branch for each unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    aux : bool, default False
        Whether to output an auxiliary result.
    fixed_size : bool, default False
        Whether to expect fixed spatial size of input image.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (1024, 2048)
        Spatial size of the expected input image.
    num_classes : int, default 19
        Number of segmentation classes.
    """
    def __init__(self,
                 layers,
                 channels,
                 init_block_channels,
                 dilations,
                 se_reductions,
                 cut_x,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(CGNet, self).__init__()
        assert (aux is not None)
        assert (fixed_size is not None)
        assert ((in_size[0] % 8 == 0) and (in_size[1] % 8 == 0))
        self.in_size = in_size
        self.num_classes = num_classes
        self.fixed_size = fixed_size

        self.features = DualPathSequential(
            return_two=False,
            first_ordinals=1,
            last_ordinals=0)
        self.features.add_module("init_block", CGInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps))
        y_in_channels = init_block_channels

        for i, (layers_i, y_out_channels) in enumerate(zip(layers, channels)):
            self.features.add_module("stage{}".format(i + 1), CGStage(
                x_channels=in_channels if cut_x[i] == 1 else 0,
                y_in_channels=y_in_channels,
                y_out_channels=y_out_channels,
                layers=layers_i,
                dilation=dilations[i],
                se_reduction=se_reductions[i],
                bn_eps=bn_eps))
            y_in_channels = y_out_channels

        self.classifier = conv1x1(
            in_channels=y_in_channels,
            out_channels=num_classes)

        self.up = InterpolationBlock(
            scale_factor=8,
            align_corners=False)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        in_size = self.in_size if self.fixed_size else x.shape[2:]
        y = self.features(x, x)
        y = self.classifier(y)
        y = self.up(y, size=in_size)
        return y


def get_cgnet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create CGNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32
    layers = [0, 3, 21]
    channels = [35, 131, 256]
    dilations = [0, 2, 4]
    se_reductions = [0, 8, 16]
    cut_x = [1, 1, 0]
    bn_eps = 1e-3

    net = CGNet(
        layers=layers,
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
        se_reductions=se_reductions,
        cut_x=cut_x,
        bn_eps=bn_eps,
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


def cgnet_cityscapes(num_classes=19, **kwargs):
    """
    CGNet model for Cityscapes from 'CGNet: A Light-weight Context Guided Network for Semantic Segmentation,'
    https://arxiv.org/abs/1811.08201.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_cgnet(num_classes=num_classes, model_name="cgnet_cityscapes", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        cgnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != cgnet_cityscapes or weight_count == 496306)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
