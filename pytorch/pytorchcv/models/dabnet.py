"""
    DABNet for image segmentation, implemented in PyTorch.
    Original paper: 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.
"""

__all__ = ['DABNet', 'dabnet_cityscapes']

import os
import torch
import torch.nn as nn
from .common import conv1x1, conv3x3, conv3x3_block, ConvBlock, NormActivation, Concurrent, InterpolationBlock,\
    DualPathSequential


class DwaConvBlock(nn.Module):
    """
    Depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    kernel_size : int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    def __init__(self,
                 channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True))):
        super(DwaConvBlock, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=(padding, 0),
            dilation=(dilation, 1),
            groups=channels,
            bias=bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=activation)
        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=(0, padding),
            dilation=(1, dilation),
            groups=channels,
            bias=bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def dwa_conv3x3_block(channels,
                      stride=1,
                      padding=1,
                      dilation=1,
                      bias=False,
                      use_bn=True,
                      bn_eps=1e-5,
                      activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 version of the depthwise asymmetric separable convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int, default 1
        Strides of the convolution.
    padding : int, default 1
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return DwaConvBlock(
        channels=channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class DABBlock(nn.Module):
    """
    DABNet specific base block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    dilation : int
        Dilation value for a dilated branch in the unit.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 channels,
                 dilation,
                 bn_eps):
        super(DABBlock, self).__init__()
        mid_channels = channels // 2

        self.norm_activ1 = NormActivation(
            in_channels=channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(channels)))
        self.conv1 = conv3x3_block(
            in_channels=channels,
            out_channels=mid_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(mid_channels)))

        self.branches = Concurrent(stack=True)
        self.branches.add_module("branches1", dwa_conv3x3_block(
            channels=mid_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(mid_channels))))
        self.branches.add_module("branches2", dwa_conv3x3_block(
            channels=mid_channels,
            padding=dilation,
            dilation=dilation,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(mid_channels))))

        self.norm_activ2 = NormActivation(
            in_channels=mid_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(mid_channels)))
        self.conv2 = conv1x1(
            in_channels=mid_channels,
            out_channels=channels)

    def forward(self, x):
        identity = x

        x = self.norm_activ1(x)
        x = self.conv1(x)

        x = self.branches(x)
        x = x.sum(dim=1)

        x = self.norm_activ2(x)
        x = self.conv2(x)

        x = x + identity
        return x


class DownBlock(nn.Module):
    """
    DABNet specific downsample block for the main branch.

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
        super(DownBlock, self).__init__()
        self.expand = (in_channels < out_channels)
        mid_channels = out_channels - in_channels if self.expand else out_channels

        self.conv = conv3x3(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2)
        if self.expand:
            self.pool = nn.MaxPool2d(
                kernel_size=2,
                stride=2)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(out_channels)))

    def forward(self, x):
        y = self.conv(x)

        if self.expand:
            z = self.pool(x)
            y = torch.cat((y, z), dim=1)

        y = self.norm_activ(y)
        return y


class DABUnit(nn.Module):
    """
    DABNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dilations : list of int
        Dilations for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dilations,
                 bn_eps):
        super(DABUnit, self).__init__()
        mid_channels = out_channels // 2

        self.down = DownBlock(
            in_channels=in_channels,
            out_channels=mid_channels,
            bn_eps=bn_eps)
        self.blocks = nn.Sequential()
        for i, dilation in enumerate(dilations):
            self.blocks.add_module("block{}".format(i + 1), DABBlock(
                channels=mid_channels,
                dilation=dilation,
                bn_eps=bn_eps))

    def forward(self, x):
        x = self.down(x)
        y = self.blocks(x)
        x = torch.cat((y, x), dim=1)
        return x


class DABStage(nn.Module):
    """
    DABNet stage.

    Parameters:
    ----------
    x_channels : int
        Number of input/output channels for x.
    y_in_channels : int
        Number of input channels for y.
    y_out_channels : int
        Number of output channels for y.
    dilations : list of int
        Dilations for blocks.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 x_channels,
                 y_in_channels,
                 y_out_channels,
                 dilations,
                 bn_eps):
        super(DABStage, self).__init__()
        self.use_unit = (len(dilations) > 0)

        self.x_down = nn.AvgPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        if self.use_unit:
            self.unit = DABUnit(
                in_channels=y_in_channels,
                out_channels=(y_out_channels - x_channels),
                dilations=dilations,
                bn_eps=bn_eps)

        self.norm_activ = NormActivation(
            in_channels=y_out_channels,
            bn_eps=bn_eps,
            activation=(lambda: nn.PReLU(y_out_channels)))

    def forward(self, y, x):
        x = self.x_down(x)
        if self.use_unit:
            y = self.unit(y)
        y = torch.cat((y, x), dim=1)
        y = self.norm_activ(y)
        return y, x


class DABInitBlock(nn.Module):
    """
    DABNet specific initial block.

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
        super(DABInitBlock, self).__init__()
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


class DABNet(nn.Module):
    """
    DABNet model from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit (for y-branch).
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of list of int
        Dilations for blocks.
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
                 channels,
                 init_block_channels,
                 dilations,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(DABNet, self).__init__()
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
        self.features.add_module("init_block", DABInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps))
        y_in_channels = init_block_channels

        for i, (y_out_channels, dilations_i) in enumerate(zip(channels, dilations)):
            self.features.add_module("stage{}".format(i + 1), DABStage(
                x_channels=in_channels,
                y_in_channels=y_in_channels,
                y_out_channels=y_out_channels,
                dilations=dilations_i,
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


def get_dabnet(model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create DABNet model with specific parameters.

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
    channels = [35, 131, 259]
    dilations = [[], [2, 2, 2], [4, 4, 8, 8, 16, 16]]
    bn_eps = 1e-3

    net = DABNet(
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
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


def dabnet_cityscapes(num_classes=19, **kwargs):
    """
    DABNet model for Cityscapes from 'DABNet: Depth-wise Asymmetric Bottleneck for Real-time Semantic Segmentation,'
    https://arxiv.org/abs/1907.11357.

    Parameters:
    ----------
    num_classes : int, default 19
        Number of segmentation classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dabnet(num_classes=num_classes, model_name="dabnet_cityscapes", **kwargs)


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
        dabnet_cityscapes,
    ]

    for model in models:

        net = model(pretrained=pretrained, in_size=in_size, fixed_size=fixed_size)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dabnet_cityscapes or weight_count == 756643)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
