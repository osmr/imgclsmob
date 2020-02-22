"""
    PRNet for AFLW2000-3D, implemented in PyTorch.
    Original paper: 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network,'
    https://arxiv.org/abs/1803.07835.
"""

__all__ = ['PRNet', 'prnet']

import os
import torch.nn as nn
from .common import ConvBlock, DeconvBlock, conv1x1, conv1x1_block, NormActivation


def conv4x4_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=(1, 2, 1, 2),
                  dilation=1,
                  groups=1,
                  bias=False,
                  use_bn=True,
                  bn_eps=1e-5,
                  activation=(lambda: nn.ReLU(inplace=True))):
    """
    4x4 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default (1, 2, 1, 2)
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


def deconv4x4_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=3,
                    ext_padding=(2, 1, 2, 1),
                    out_padding=0,
                    dilation=1,
                    groups=1,
                    bias=False,
                    use_bn=True,
                    bn_eps=1e-5,
                    activation=(lambda: nn.ReLU(inplace=True))):
    """
    4x4 version of the standard deconvolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default (2, 1, 2, 1)
        Padding value for deconvolution layer.
    ext_padding : tuple/list of 4 int, default None
        Extra padding value for deconvolution layer.
    out_padding : int or tuple/list of 2 int
        Output padding value for deconvolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    """
    return DeconvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=4,
        stride=stride,
        padding=padding,
        ext_padding=ext_padding,
        out_padding=out_padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
        use_bn=use_bn,
        bn_eps=bn_eps,
        activation=activation)


class PRResBottleneck(nn.Module):
    """
    PRNet specific bottleneck block for residual path in residual unit unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for the second convolution layer in bottleneck.
    bn_eps : float
        Small float added to variance in Batch norm.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding,
                 bn_eps,
                 bottleneck_factor=2):
        super(PRResBottleneck, self).__init__()
        mid_channels = out_channels // bottleneck_factor

        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bn_eps=bn_eps)
        self.conv2 = conv4x4_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride,
            padding=padding,
            bn_eps=bn_eps)
        self.conv3 = conv1x1(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PRResUnit(nn.Module):
    """
    PRNet specific ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for the second convolution layer in bottleneck.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 padding,
                 bn_eps,
                 stride):
        super(PRResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        self.body = PRResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            stride=stride,
            padding=padding)
        self.norm_activ = NormActivation(
            in_channels=out_channels,
            bn_eps=bn_eps)

    def forward(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        x = self.norm_activ(x)
        return x


class PROutputBlock(nn.Module):
    """
    PRNet specific output block.

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
        super(PROutputBlock, self).__init__()
        self.conv1 = deconv4x4_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bn_eps=bn_eps)
        self.conv2 = deconv4x4_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps)
        self.conv3 = deconv4x4_block(
            in_channels=out_channels,
            out_channels=out_channels,
            bn_eps=bn_eps,
            activation=nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class PRNet(nn.Module):
    """
    PRNet model from 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network,'
    https://arxiv.org/abs/1803.07835.

    Parameters:
    ----------
    channels : list of list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (256, 256)
        Spatial size of the expected input image.
    num_classes : int, default 3
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(256, 256),
                 num_classes=3):
        super(PRNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv4x4_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            bn_eps=bn_eps))
        in_channels = init_block_channels

        encoder = nn.Sequential()
        for i, channels_per_stage in enumerate(channels[0]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) else 1
                padding = (1, 2, 1, 2) if (stride == 1) else 1
                stage.add_module("unit{}".format(j + 1), PRResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding,
                    bn_eps=bn_eps))
                in_channels = out_channels
            encoder.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("encoder", encoder)

        decoder = nn.Sequential()
        for i, channels_per_stage in enumerate(channels[1]):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                padding = 3 if (stride == 1) else 1
                ext_padding = (2, 1, 2, 1) if (stride == 1) else None
                stage.add_module("unit{}".format(j + 1), deconv4x4_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=padding,
                    ext_padding=ext_padding,
                    bn_eps=bn_eps))
                in_channels = out_channels
            decoder.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("decoder", decoder)

        self.output = PROutputBlock(
            in_channels=in_channels,
            out_channels=num_classes,
            bn_eps=bn_eps)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_prnet(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".torch", "models"),
              **kwargs):
    """
    Create PRNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 16
    enc_channels = [[32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]
    dec_channels = [[512], [256, 256, 256], [128, 128, 128], [64, 64, 64], [32, 32], [16, 16]]
    channels = [enc_channels, dec_channels]

    net = PRNet(
        channels=channels,
        init_block_channels=init_block_channels,
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


def prnet(**kwargs):
    """
    PRNet model for AFLW2000-3D from 'Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression
    Network,' https://arxiv.org/abs/1803.07835.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_prnet(model_name="prnet", bn_eps=1e-3, **kwargs)


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
        prnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != prnet or weight_count == 13353618)

        x = torch.randn(1, 3, 256, 256)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 3, 256, 256))


if __name__ == "__main__":
    _test()
