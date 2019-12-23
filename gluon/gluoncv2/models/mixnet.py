"""
    MixNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.
"""

__all__ = ['MixNet', 'mixnet_s', 'mixnet_m', 'mixnet_l']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import round_channels, get_activation_layer, split, conv1x1_block, conv3x3_block, dwconv3x3_block, SEBlock


class MixConv(HybridBlock):
    """
    Mixed convolution layer from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    axis : int, default 1
        The axis on which to concatenate the outputs.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 axis=1,
                 **kwargs):
        super(MixConv, self).__init__(**kwargs)
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        padding = padding if isinstance(padding, list) else [padding]
        kernel_count = len(kernel_size)
        self.splitted_in_channels = self.split_channels(in_channels, kernel_count)
        splitted_out_channels = self.split_channels(out_channels, kernel_count)
        self.axis = axis

        with self.name_scope():
            for i, kernel_size_i in enumerate(kernel_size):
                in_channels_i = self.splitted_in_channels[i]
                out_channels_i = splitted_out_channels[i]
                padding_i = padding[i]
                self.register_child(
                    nn.Conv2D(
                        channels=out_channels_i,
                        kernel_size=kernel_size_i,
                        strides=strides,
                        padding=padding_i,
                        dilation=dilation,
                        groups=(out_channels_i if out_channels == groups else groups),
                        use_bias=use_bias,
                        in_channels=in_channels_i))

    def hybrid_forward(self, F, x):
        xx = split(x, sizes=self.splitted_in_channels, axis=self.axis)
        out = [conv_i(x_i) for x_i, conv_i in zip(xx, self._children.values())]
        x = F.concat(*out, dim=self.axis)
        return x

    @staticmethod
    def split_channels(channels, kernel_count):
        splitted_channels = [channels // kernel_count] * kernel_count
        splitted_channels[0] += channels - sum(splitted_channels)
        return splitted_channels


class MixConvBlock(HybridBlock):
    """
    Mixed convolution block with Batch normalization and activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of int, or tuple/list of tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str or None, default nn.Activation("relu")
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 use_bn=True,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(MixConvBlock, self).__init__(**kwargs)
        self.activate = (activation is not None)
        self.use_bn = use_bn

        with self.name_scope():
            self.conv = MixConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias)
            if self.use_bn:
                self.bn = nn.BatchNorm(
                    in_channels=out_channels,
                    epsilon=bn_epsilon,
                    use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = get_activation_layer(activation)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def mixconv1x1_block(in_channels,
                     out_channels,
                     kernel_count,
                     strides=1,
                     groups=1,
                     use_bias=False,
                     use_bn=True,
                     bn_epsilon=1e-5,
                     bn_use_global_stats=False,
                     activation=(lambda: nn.Activation("relu")),
                     **kwargs):
    """
    1x1 version of the mixed convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_count : int
        Kernel count.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : function or str, or None, default nn.Activation("relu")
        Activation function or name of activation function.
    """
    return MixConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=([1] * kernel_count),
        strides=strides,
        padding=([0] * kernel_count),
        groups=groups,
        use_bias=use_bias,
        use_bn=use_bn,
        bn_epsilon=bn_epsilon,
        bn_use_global_stats=bn_use_global_stats,
        activation=activation,
        **kwargs)


class MixUnit(HybridBlock):
    """
    MixNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    exp_channels : int
        Number of middle (expanded) channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    exp_kernel_count : int
        Expansion convolution kernel count for each unit.
    conv1_kernel_count : int
        Conv1 kernel count for each unit.
    conv2_kernel_count : int
        Conv2 kernel count for each unit.
    exp_factor : int
        Expansion factor for each unit.
    se_factor : int
        SE reduction factor for each unit.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : str
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 exp_kernel_count,
                 conv1_kernel_count,
                 conv2_kernel_count,
                 exp_factor,
                 se_factor,
                 bn_use_global_stats,
                 activation,
                 **kwargs):
        super(MixUnit, self).__init__(**kwargs)
        assert (exp_factor >= 1)
        assert (se_factor >= 0)
        self.residual = (in_channels == out_channels) and (strides == 1)
        self.use_se = se_factor > 0
        mid_channels = exp_factor * in_channels
        self.use_exp_conv = exp_factor > 1

        with self.name_scope():
            if self.use_exp_conv:
                if exp_kernel_count == 1:
                    self.exp_conv = conv1x1_block(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        bn_use_global_stats=bn_use_global_stats,
                        activation=activation)
                else:
                    self.exp_conv = mixconv1x1_block(
                        in_channels=in_channels,
                        out_channels=mid_channels,
                        kernel_count=exp_kernel_count,
                        bn_use_global_stats=bn_use_global_stats,
                        activation=activation)
            if conv1_kernel_count == 1:
                self.conv1 = dwconv3x3_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            else:
                self.conv1 = MixConvBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    kernel_size=[3 + 2 * i for i in range(conv1_kernel_count)],
                    strides=strides,
                    padding=[1 + i for i in range(conv1_kernel_count)],
                    groups=mid_channels,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            if self.use_se:
                self.se = SEBlock(
                    channels=mid_channels,
                    reduction=(exp_factor * se_factor),
                    round_mid=False,
                    mid_activation=activation)
            if conv2_kernel_count == 1:
                self.conv2 = conv1x1_block(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            else:
                self.conv2 = mixconv1x1_block(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_count=conv2_kernel_count,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        if self.use_exp_conv:
            x = self.exp_conv(x)
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class MixInitBlock(HybridBlock):
    """
    MixNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(MixInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = MixUnit(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                exp_kernel_count=1,
                conv1_kernel_count=1,
                conv2_kernel_count=1,
                exp_factor=1,
                se_factor=0,
                bn_use_global_stats=bn_use_global_stats,
                activation="relu")

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MixNet(HybridBlock):
    """
    MixNet model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    exp_kernel_counts : list of list of int
        Expansion convolution kernel count for each unit.
    conv1_kernel_counts : list of list of int
        Conv1 kernel count for each unit.
    conv2_kernel_counts : list of list of int
        Conv2 kernel count for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    se_factors : list of list of int
        SE reduction factor for each unit.
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
                 final_block_channels,
                 exp_kernel_counts,
                 conv1_kernel_counts,
                 conv2_kernel_counts,
                 exp_factors,
                 se_factors,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(MixNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(MixInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if ((j == 0) and (i != 3)) or\
                                       ((j == len(channels_per_stage) // 2) and (i == 3)) else 1
                        exp_kernel_count = exp_kernel_counts[i][j]
                        conv1_kernel_count = conv1_kernel_counts[i][j]
                        conv2_kernel_count = conv2_kernel_counts[i][j]
                        exp_factor = exp_factors[i][j]
                        se_factor = se_factors[i][j]
                        activation = "relu" if i == 0 else "swish"
                        stage.add(MixUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            exp_kernel_count=exp_kernel_count,
                            conv1_kernel_count=conv1_kernel_count,
                            conv2_kernel_count=conv2_kernel_count,
                            exp_factor=exp_factor,
                            se_factor=se_factor,
                            bn_use_global_stats=bn_use_global_stats,
                            activation=activation))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=final_block_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation))
            in_channels = final_block_channels
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


def get_mixnet(version,
               width_scale,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    """
    Create MixNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('s' or 'm').
    width_scale : float
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

    if version == "s":
        init_block_channels = 16
        channels = [[24, 24], [40, 40, 40, 40], [80, 80, 80], [120, 120, 120, 200, 200, 200]]
        exp_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 1, 1], [2, 2, 2, 1, 1, 1]]
        conv1_kernel_counts = [[1, 1], [3, 2, 2, 2], [3, 2, 2], [3, 4, 4, 5, 4, 4]]
        conv2_kernel_counts = [[2, 2], [1, 2, 2, 2], [2, 2, 2], [2, 2, 2, 1, 2, 2]]
        exp_factors = [[6, 3], [6, 6, 6, 6], [6, 6, 6], [6, 3, 3, 6, 6, 6]]
        se_factors = [[0, 0], [2, 2, 2, 2], [4, 4, 4], [2, 2, 2, 2, 2, 2]]
    elif version == "m":
        init_block_channels = 24
        channels = [[32, 32], [40, 40, 40, 40], [80, 80, 80, 80], [120, 120, 120, 120, 200, 200, 200, 200]]
        exp_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 2, 2, 2], [1, 2, 2, 2, 1, 1, 1, 1]]
        conv1_kernel_counts = [[3, 1], [4, 2, 2, 2], [3, 4, 4, 4], [1, 4, 4, 4, 4, 4, 4, 4]]
        conv2_kernel_counts = [[2, 2], [1, 2, 2, 2], [1, 2, 2, 2], [1, 2, 2, 2, 1, 2, 2, 2]]
        exp_factors = [[6, 3], [6, 6, 6, 6], [6, 6, 6, 6], [6, 3, 3, 3, 6, 6, 6, 6]]
        se_factors = [[0, 0], [2, 2, 2, 2], [4, 4, 4, 4], [2, 2, 2, 2, 2, 2, 2, 2]]
    else:
        raise ValueError("Unsupported MixNet version {}".format(version))

    final_block_channels = 1536

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale)

    net = MixNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        exp_kernel_counts=exp_kernel_counts,
        conv1_kernel_counts=conv1_kernel_counts,
        conv2_kernel_counts=conv2_kernel_counts,
        exp_factors=exp_factors,
        se_factors=se_factors,
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


def mixnet_s(**kwargs):
    """
    MixNet-S model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="s", width_scale=1.0, model_name="mixnet_s", **kwargs)


def mixnet_m(**kwargs):
    """
    MixNet-M model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="m", width_scale=1.0, model_name="mixnet_m", **kwargs)


def mixnet_l(**kwargs):
    """
    MixNet-L model from 'MixConv: Mixed Depthwise Convolutional Kernels,' https://arxiv.org/abs/1907.09595.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mixnet(version="m", width_scale=1.3, model_name="mixnet_l", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        mixnet_s,
        mixnet_m,
        mixnet_l,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mixnet_s or weight_count == 4134606)
        assert (model != mixnet_m or weight_count == 5014382)
        assert (model != mixnet_l or weight_count == 7329252)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
