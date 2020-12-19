"""
    Jasper for ASR, implemented in PyTorch.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['Jasper', 'jasper3x5']

import os
import torch.nn as nn


class ConvBlock1d(nn.Module):
    """
    Standard 1D convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    stride : int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int
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
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 dropout_rate=0.0):
        super(ConvBlock1d, self).__init__()
        self.activate = (activation is not None)
        self.use_dropout = (dropout_rate != 0.0)
        self.use_bn = use_bn

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = nn.ReLU(inplace=True)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


def conv1d1x1_block(in_channels,
                    out_channels,
                    stride=1,
                    padding=0,
                    **kwargs):
    """
    1x1 version of the standard 1D convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int, or tuple/list of 2 int, or tuple/list of 4 int, default 0
        Padding value for convolution layer.
    """
    return ConvBlock1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        **kwargs)


class JasperUnit(nn.Module):
    """
    Jasper unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    repeat : int
        Count of body convolution blocks.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 dropout_rate,
                 repeat):
        super(JasperUnit, self).__init__()
        self.identity_conv = conv1d1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout_rate=0.0,
            activation=None)

        self.body = nn.Sequential()
        for i in range(repeat):
            activation = (lambda: nn.ReLU(inplace=True)) if i < repeat - 1 else None
            dropout_rate_i = dropout_rate if i < repeat - 1 else 0.0
            self.body.add_module("block{}".format(i + 1), ConvBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size // 2),
                dropout_rate=dropout_rate_i,
                activation=activation))
            in_channels = out_channels

        self.activ = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        identity = self.identity_conv(x)
        x = self.body(x)
        x = x + identity
        x = self.activ(x)
        x = self.dropout(x)
        return x


class Jasper(nn.Module):
    """
    Jasper model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    dropout_rates : list float
        Parameter of Dropout layer for each unit.
    repeat : int
        Count of body convolution blocks.
    in_channels : int, default 120
        Number of input channels (audio features).
    num_classes : int, default 11
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 dropout_rates,
                 repeat,
                 in_channels=120,
                 num_classes=11):
        super(Jasper, self).__init__()
        self.in_size = None
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ConvBlock1d(
                in_channels=in_channels,
                out_channels=init_block_channels,
                kernel_size=11,
                stride=2,
                padding=5,
                dropout_rate=0.2))
        in_channels = init_block_channels
        for i, out_channels in enumerate(channels):
            dropout_rate = dropout_rates[i]
            self.features.add_module("unit{}".format(i + 1), JasperUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=11,
                dropout_rate=dropout_rate,
                repeat=repeat))
            in_channels = out_channels
        self.features.add_module("final_block", conv1d1x1_block(
            in_channels=in_channels,
            out_channels=final_block_channels,
            dropout_rate=0.4))
        in_channels = final_block_channels

        self.output = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            bias=True)

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


def get_jasper(version,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create Jasper model with specific parameters.

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
    if version == "3x5":
        channels = [256, 256, 512]
        dropout_rates = [0.2, 0.2, 0.2]
        repeat = 5
    else:
        raise ValueError("Unsupported Jasper version: {}".format(version))

    init_block_channels = 256
    final_block_channels = 512

    net = Jasper(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        dropout_rates=dropout_rates,
        repeat=repeat,
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


def jasper3x5(**kwargs):
    """
    Jasper 3x5 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version="3x5", model_name="jasper3x5", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import numpy as np
    import torch

    pretrained = False
    audio_features = 120
    num_classes = 11

    models = [
        jasper3x5,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            num_classes=num_classes,
            pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasper3x5 or weight_count == 21066763)

        batch = 1
        seq_len = np.random.randint(60, 150)
        x = torch.randn(batch, audio_features, seq_len)
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, num_classes, seq_len // 2)) or \
               (tuple(y.size()) == (batch, num_classes, seq_len // 2 + 1))


if __name__ == "__main__":
    _test()
