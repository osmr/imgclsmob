"""
    DIA-ResNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
"""

__all__ = ['DIAResNet', 'diaresnet10', 'diaresnet12', 'diaresnet14', 'diaresnetbc14b', 'diaresnet16', 'diaresnet18',
           'diaresnet26', 'diaresnetbc26b', 'diaresnet34', 'diaresnetbc38b', 'diaresnet50', 'diaresnet50b',
           'diaresnet101', 'diaresnet101b', 'diaresnet152', 'diaresnet152b', 'diaresnet200', 'diaresnet200b',
           'DIAAttention', 'DIAResUnit']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, DualPathSequential
from .resnet import ResBlock, ResBottleneck, ResInitBlock


class FirstLSTMAmp(HybridBlock):
    """
    First LSTM amplifier branch.

    Parameters:
    ----------
    in_units : int
        Number of input channels.
    units : int
        Number of output channels.
    """
    def __init__(self,
                 in_units,
                 units,
                 **kwargs):
        super(FirstLSTMAmp, self).__init__(**kwargs)
        mid_units = in_units // 4

        with self.name_scope():
            self.fc1 = nn.Dense(
                units=mid_units,
                in_units=in_units)
            self.activ = nn.Activation("relu")
            self.fc2 = nn.Dense(
                units=units,
                in_units=mid_units)

    def hybrid_forward(self, F, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class DIALSTMCell(HybridBlock):
    """
    DIA-LSTM cell.

    Parameters:
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
                 dropout_rate=0.1,
                 **kwargs):
        super(DIALSTMCell, self).__init__(**kwargs)
        self.num_layers = num_layers
        out_features = 4 * in_h_features

        with self.name_scope():
            self.x_amps = nn.HybridSequential(prefix="")
            self.h_amps = nn.HybridSequential(prefix="")
            for i in range(num_layers):
                amp_class = FirstLSTMAmp if i == 0 else nn.Dense
                self.x_amps.add(amp_class(
                    in_units=in_x_features,
                    units=out_features))
                self.h_amps.add(amp_class(
                    in_units=in_h_features,
                    units=out_features))
                in_x_features = in_h_features
            self.dropout = nn.Dropout(rate=dropout_rate)

    def hybrid_forward(self, F, x, h, c):
        hy = []
        cy = []
        for i in range(self.num_layers):
            hx_i = h[i]
            cx_i = c[i]
            gates = self.x_amps[i](x) + self.h_amps[i](hx_i)
            i_gate, f_gate, c_gate, o_gate = F.split(gates, axis=1, num_outputs=4)
            i_gate = F.sigmoid(i_gate)
            f_gate = F.sigmoid(f_gate)
            c_gate = F.tanh(c_gate)
            o_gate = F.sigmoid(o_gate)
            cy_i = (f_gate * cx_i) + (i_gate * c_gate)
            hy_i = o_gate * F.sigmoid(cy_i)
            cy.append(cy_i)
            hy.append(hy_i)
            x = self.dropout(hy_i)
        return hy, cy


class DIAAttention(HybridBlock):
    """
    DIA-Net attention module.

    Parameters:
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
                 num_layers=1,
                 **kwargs):
        super(DIAAttention, self).__init__(**kwargs)
        self.num_layers = num_layers

        with self.name_scope():
            self.lstm = DIALSTMCell(
                in_x_features=in_x_features,
                in_h_features=in_h_features,
                num_layers=num_layers)

    def hybrid_forward(self, F, x, hc=None):
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = w.flatten()
        if hc is None:
            h = [F.zeros_like(w)] * self.num_layers
            c = [F.zeros_like(w)] * self.num_layers
        else:
            h, c = hc
        h, c = self.lstm(w, h, c)
        w = h[self.num_layers - 1].expand_dims(axis=-1).expand_dims(axis=-1)
        x = F.broadcast_mul(x, w)
        return x, (h, c)


class DIAResUnit(HybridBlock):
    """
    DIA-ResNet unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for the second convolution layer in bottleneck.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for the second convolution layer in bottleneck.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
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
                 strides,
                 padding=1,
                 dilation=1,
                 bn_use_global_stats=False,
                 bottleneck=True,
                 conv1_stride=False,
                 attention=None,
                 **kwargs):
        super(DIAResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            if bottleneck:
                self.body = ResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    padding=padding,
                    dilation=dilation,
                    bn_use_global_stats=bn_use_global_stats,
                    conv1_stride=conv1_stride)
            else:
                self.body = ResBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            self.activ = nn.Activation("relu")
            self.attention = attention

    def hybrid_forward(self, F, x, hc=None):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x, hc = self.attention(x, hc)
        x = x + identity
        x = self.activ(x)
        return x, hc


class DIAResNet(HybridBlock):
    """
    DIA-ResNet model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(DIAResNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = DualPathSequential(
                    return_two=False,
                    prefix="stage{}_".format(i + 1))
                attention = DIAAttention(
                    in_x_features=channels_per_stage[0],
                    in_h_features=channels_per_stage[0])
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(DIAResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck,
                            conv1_stride=conv1_stride,
                            attention=attention))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_diaresnet(blocks,
                  bottleneck=None,
                  conv1_stride=True,
                  width_scale=1.0,
                  model_name=None,
                  pretrained=False,
                  ctx=cpu(),
                  root=os.path.join("~", ".mxnet", "models"),
                  **kwargs):
    """
    Create DIA-ResNet model with specific parameters.

    Parameters:
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
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
        from .model_store import get_model_file
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def diaresnet10(**kwargs):
    """
    DIA-ResNet-10 model from 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=10, model_name="diaresnet10", **kwargs)


def diaresnet12(**kwargs):
    """
    DIA-ResNet-12 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=12, model_name="diaresnet12", **kwargs)


def diaresnet14(**kwargs):
    """
    DIA-ResNet-14 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=14, model_name="diaresnet14", **kwargs)


def diaresnetbc14b(**kwargs):
    """
    DIA-ResNet-BC-14b model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=14, bottleneck=True, conv1_stride=False, model_name="diaresnetbc14b", **kwargs)


def diaresnet16(**kwargs):
    """
    DIA-ResNet-16 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=16, model_name="diaresnet16", **kwargs)


def diaresnet18(**kwargs):
    """
    DIA-ResNet-18 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=18, model_name="diaresnet18", **kwargs)


def diaresnet26(**kwargs):
    """
    DIA-ResNet-26 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=26, bottleneck=False, model_name="diaresnet26", **kwargs)


def diaresnetbc26b(**kwargs):
    """
    DIA-ResNet-BC-26b model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=26, bottleneck=True, conv1_stride=False, model_name="diaresnetbc26b", **kwargs)


def diaresnet34(**kwargs):
    """
    DIA-ResNet-34 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=34, model_name="diaresnet34", **kwargs)


def diaresnetbc38b(**kwargs):
    """
    DIA-ResNet-BC-38b model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model (bottleneck compressed).

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=38, bottleneck=True, conv1_stride=False, model_name="diaresnetbc38b", **kwargs)


def diaresnet50(**kwargs):
    """
    DIA-ResNet-50 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=50, model_name="diaresnet50", **kwargs)


def diaresnet50b(**kwargs):
    """
    DIA-ResNet-50 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=50, conv1_stride=False, model_name="diaresnet50b", **kwargs)


def diaresnet101(**kwargs):
    """
    DIA-ResNet-101 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=101, model_name="diaresnet101", **kwargs)


def diaresnet101b(**kwargs):
    """
    DIA-ResNet-101 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=101, conv1_stride=False, model_name="diaresnet101b", **kwargs)


def diaresnet152(**kwargs):
    """
    DIA-ResNet-152 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=152, model_name="diaresnet152", **kwargs)


def diaresnet152b(**kwargs):
    """
    DIA-ResNet-152 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=152, conv1_stride=False, model_name="diaresnet152b", **kwargs)


def diaresnet200(**kwargs):
    """
    DIA-ResNet-200 model 'DIANet: Dense-and-Implicit Attention Network,' https://arxiv.org/abs/1905.10671.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=200, model_name="diaresnet200", **kwargs)


def diaresnet200b(**kwargs):
    """
    DIA-ResNet-200 model with stride at the second convolution in bottleneck block from 'DIANet: Dense-and-Implicit
    Attention Network,' https://arxiv.org/abs/1905.10671.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_diaresnet(blocks=200, conv1_stride=False, model_name="diaresnet200b", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
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

        x = mx.nd.zeros((14, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (14, 1000))


if __name__ == "__main__":
    _test()
