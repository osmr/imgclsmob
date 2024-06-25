"""
    DIA-ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
"""

__all__ = ['DIAResNet', 'diaresnet10', 'diaresnet12', 'diaresnet14', 'diaresnetbc14b', 'diaresnet16', 'diaresnet18',
           'diaresnet26', 'diaresnetbc26b', 'diaresnet34', 'diaresnetbc38b', 'diaresnet50', 'diaresnet50b',
           'diaresnet101', 'diaresnet101b', 'diaresnet152', 'diaresnet152b', 'diaresnet200', 'diaresnet200b',
           'DIAAttention', 'DIAResUnit']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
from .common import conv1x1_block, DualPathSequential
from .resnet import ResBlock, ResBottleneck, ResInitBlock


class FirstLSTMAmp(nn.Module):
    """
    First LSTM amplifier branch.

    Parameters
    ----------
    in_features : int
        Number of input channels.
    out_features : int
        Number of output channels.
    """
    def __init__(self,
                 in_features,
                 out_features):
        super(FirstLSTMAmp, self).__init__()
        mid_features = in_features // 4

        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=mid_features)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            in_features=mid_features,
            out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class DIALSTMCell(nn.Module):
    """
    DIA-LSTM cell.

    Parameters
    ----------
    in_x_features : int
        Number of x input channels.
    in_h_features : int
        Number of h input channels.
    num_layers : int
        Number of amplifiers.
    dropout_rate : float, default 0.1
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_x_features,
                 in_h_features,
                 num_layers,
                 dropout_rate=0.1):
        super(DIALSTMCell, self).__init__()
        self.num_layers = num_layers
        out_features = 4 * in_h_features

        self.x_amps = nn.Sequential()
        self.h_amps = nn.Sequential()
        for i in range(num_layers):
            amp_class = FirstLSTMAmp if i == 0 else nn.Linear
            self.x_amps.add_module("amp{}".format(i + 1), amp_class(
                in_features=in_x_features,
                out_features=out_features))
            self.h_amps.add_module("amp{}".format(i + 1), amp_class(
                in_features=in_h_features,
                out_features=out_features))
            in_x_features = in_h_features
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, h, c):
        hy = []
        cy = []
        for i in range(self.num_layers):
            hx_i = h[i]
            cx_i = c[i]
            gates = self.x_amps[i](x) + self.h_amps[i](hx_i)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(chunks=4, dim=1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate)
            cy_i = (f_gate * cx_i) + (i_gate * c_gate)
            hy_i = o_gate * torch.sigmoid(cy_i)
            cy.append(cy_i)
            hy.append(hy_i)
            x = self.dropout(hy_i)
        return hy, cy


class DIAAttention(nn.Module):
    """
    DIA-Net attention module.

    Parameters
    ----------
    in_x_features : int
        Number of x input channels.
    in_h_features : int
        Number of h input channels.
    num_layers : int, default 1
        Number of amplifiers.
    """
    def __init__(self,
                 in_x_features,
                 in_h_features,
                 num_layers=1):
        super(DIAAttention, self).__init__()
        self.num_layers = num_layers

        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.lstm = DIALSTMCell(
            in_x_features=in_x_features,
            in_h_features=in_h_features,
            num_layers=num_layers)

    def forward(self, x, hc=None):
        w = self.pool(x)
        w = w.view(w.size(0), -1)
        if hc is None:
            h = [torch.zeros_like(w)] * self.num_layers
            c = [torch.zeros_like(w)] * self.num_layers
        else:
            h, c = hc
        h, c = self.lstm(w, h, c)
        w = h[-1].unsqueeze(dim=-1).unsqueeze(dim=-1)
        x = x * w
        return x, (h, c)


class DIAResUnit(nn.Module):
    """
    DIA-ResNet unit with residual connection.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple(int, int)
        Strides of the convolution.
    padding : int or tuple(int, int), default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple(int, int), default 1
        Dilation value for the second convolution layer in bottleneck.
    bottleneck : bool, default True
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default False
        Whether to use stride in the first or the second convolution layer of the block.
    attention : nn.Module, default None
        Attention module.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 padding=1,
                 dilation=1,
                 bottleneck=True,
                 conv1_stride=False,
                 attention=None):
        super(DIAResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        if bottleneck:
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                padding=padding,
                dilation=dilation,
                conv1_stride=conv1_stride)
        else:
            self.body = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                activation=None)
        self.activ = nn.ReLU(inplace=True)
        self.attention = attention

    def forward(self, x, hc=None):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x, hc = self.attention(x, hc)
        x = x + identity
        x = self.activ(x)
        return x, hc


class DIAResNet(nn.Module):
    """
    DIA-ResNet model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    channels : list(list(int))
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 in_channels: int = 3,
                 in_size: tuple[int, int] = (224, 224),
                 num_classes: int = 1000):
        super(DIAResNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = DualPathSequential(return_two=False)
            attention = DIAAttention(
                in_x_features=channels_per_stage[0],
                in_h_features=channels_per_stage[0])
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) and (i != 0) else 1
                stage.add_module("unit{}".format(j + 1), DIAResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride,
                    attention=attention))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_diaresnet(blocks,
                  bottleneck=None,
                  conv1_stride=True,
                  width_scale=1.0,
                  model_name: str | None = None,
                  pretrained: bool = False,
                  root: str = os.path.join("~", ".torch", "models"),
                  **kwargs) -> nn.Module:
    """
    Create DIA-ResNet model with specific parameters.

    Parameters
    ----------
    blocks : int
        Number of blocks.
    bottleneck : bool, default None
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    if bottleneck is None:
        bottleneck = (blocks >= 50)

    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14 and not bottleneck:
        layers = [2, 2, 1, 1]
    elif (blocks == 14) and bottleneck:
        layers = [1, 1, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif (blocks == 26) and not bottleneck:
        layers = [3, 3, 3, 3]
    elif (blocks == 26) and bottleneck:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif (blocks == 38) and bottleneck:
        layers = [3, 3, 3, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported DIA-ResNet with number of blocks: {}".format(blocks))

    if bottleneck:
        assert (sum(layers) * 3 + 2 == blocks)
    else:
        assert (sum(layers) * 2 + 2 == blocks)

    init_block_channels = 64
    channels_per_layers = [64, 128, 256, 512]

    if bottleneck:
        bottleneck_factor = 4
        channels_per_layers = [ci * bottleneck_factor for ci in channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = DIAResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
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


def diaresnet10(**kwargs) -> nn.Module:
    """
    DIA-ResNet-10 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=10, model_name="diaresnet10", **kwargs)


def diaresnet12(**kwargs) -> nn.Module:
    """
    DIA-ResNet-12 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=12, model_name="diaresnet12", **kwargs)


def diaresnet14(**kwargs) -> nn.Module:
    """
    DIA-ResNet-14 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=14, model_name="diaresnet14", **kwargs)


def diaresnetbc14b(**kwargs) -> nn.Module:
    """
    DIA-ResNet-BC-14b model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=14, bottleneck=True, conv1_stride=False, model_name="diaresnetbc14b", **kwargs)


def diaresnet16(**kwargs) -> nn.Module:
    """
    DIA-ResNet-16 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=16, model_name="diaresnet16", **kwargs)


def diaresnet18(**kwargs) -> nn.Module:
    """
    DIA-ResNet-18 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=18, model_name="diaresnet18", **kwargs)


def diaresnet26(**kwargs) -> nn.Module:
    """
    DIA-ResNet-26 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=26, bottleneck=False, model_name="diaresnet26", **kwargs)


def diaresnetbc26b(**kwargs) -> nn.Module:
    """
    DIA-ResNet-BC-26b model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=26, bottleneck=True, conv1_stride=False, model_name="diaresnetbc26b", **kwargs)


def diaresnet34(**kwargs) -> nn.Module:
    """
    DIA-ResNet-34 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=34, model_name="diaresnet34", **kwargs)


def diaresnetbc38b(**kwargs) -> nn.Module:
    """
    DIA-ResNet-BC-38b model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=38, bottleneck=True, conv1_stride=False, model_name="diaresnetbc38b", **kwargs)


def diaresnet50(**kwargs) -> nn.Module:
    """
    DIA-ResNet-50 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=50, model_name="diaresnet50", **kwargs)


def diaresnet50b(**kwargs) -> nn.Module:
    """
    DIA-ResNet-50 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=50, conv1_stride=False, model_name="diaresnet50b", **kwargs)


def diaresnet101(**kwargs) -> nn.Module:
    """
    DIA-ResNet-101 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=101, model_name="diaresnet101", **kwargs)


def diaresnet101b(**kwargs) -> nn.Module:
    """
    DIA-ResNet-101 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=101, conv1_stride=False, model_name="diaresnet101b", **kwargs)


def diaresnet152(**kwargs) -> nn.Module:
    """
    DIA-ResNet-152 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=152, model_name="diaresnet152", **kwargs)


def diaresnet152b(**kwargs) -> nn.Module:
    """
    DIA-ResNet-152 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=152, conv1_stride=False, model_name="diaresnet152b", **kwargs)


def diaresnet200(**kwargs) -> nn.Module:
    """
    DIA-ResNet-200 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=200, model_name="diaresnet200", **kwargs)


def diaresnet200b(**kwargs) -> nn.Module:
    """
    DIA-ResNet-200 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.

    Returns
    -------
    nn.Module
        Desired module.
    """
    return get_diaresnet(blocks=200, conv1_stride=False, model_name="diaresnet200b", **kwargs)


def calc_net_weights(net: nn.Module) -> int:
    """
    Calculate network trainable weight count.

    Parameters
    ----------
    net : nn.Module
        Network.

    Returns
    -------
    int
        Calculated number of weights.
    """
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
        diaresnet10,
        diaresnet12,
        diaresnet14,
        diaresnetbc14b,
        diaresnet16,
        diaresnet18,
        diaresnet26,
        diaresnetbc26b,
        diaresnet34,
        diaresnetbc38b,
        diaresnet50,
        diaresnet50b,
        diaresnet101,
        diaresnet101b,
        diaresnet152,
        diaresnet152b,
        diaresnet200,
        diaresnet200b,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weights(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != diaresnet10 or weight_count == 6297352)
        assert (model != diaresnet12 or weight_count == 6371336)
        assert (model != diaresnet14 or weight_count == 6666760)
        assert (model != diaresnetbc14b or weight_count == 24023976)
        assert (model != diaresnet16 or weight_count == 7847432)
        assert (model != diaresnet18 or weight_count == 12568072)
        assert (model != diaresnet26 or weight_count == 18838792)
        assert (model != diaresnetbc26b or weight_count == 29954216)
        assert (model != diaresnet34 or weight_count == 22676232)
        assert (model != diaresnetbc38b or weight_count == 35884456)
        assert (model != diaresnet50 or weight_count == 39516072)
        assert (model != diaresnet50b or weight_count == 39516072)
        assert (model != diaresnet101 or weight_count == 58508200)
        assert (model != diaresnet101b or weight_count == 58508200)
        assert (model != diaresnet152 or weight_count == 74151848)
        assert (model != diaresnet152b or weight_count == 74151848)
        assert (model != diaresnet200 or weight_count == 78632872)
        assert (model != diaresnet200b or weight_count == 78632872)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
