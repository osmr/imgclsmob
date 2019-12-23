"""
    MobileNetV3 for ImageNet-1K, implemented in Gluon.
    Original paper: 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.
"""

__all__ = ['MobileNetV3', 'mobilenetv3_small_w7d20', 'mobilenetv3_small_wd2', 'mobilenetv3_small_w3d4',
           'mobilenetv3_small_w1', 'mobilenetv3_small_w5d4', 'mobilenetv3_large_w7d20', 'mobilenetv3_large_wd2',
           'mobilenetv3_large_w3d4', 'mobilenetv3_large_w1', 'mobilenetv3_large_w5d4']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block, SEBlock,\
    HSwish


class MobileNetV3Unit(HybridBlock):
    """
    MobileNetV3 unit.

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
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    activation : str
        Activation function or name of activation function.
    use_se : bool
        Whether to use SE-module.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 exp_channels,
                 strides,
                 use_kernel3,
                 activation,
                 use_se,
                 bn_use_global_stats=False,
                 **kwargs):
        super(MobileNetV3Unit, self).__init__(**kwargs)
        assert (exp_channels >= out_channels)
        self.residual = (in_channels == out_channels) and (strides == 1)
        self.use_se = use_se
        self.use_exp_conv = exp_channels != out_channels
        mid_channels = exp_channels

        with self.name_scope():
            if self.use_exp_conv:
                self.exp_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            if use_kernel3:
                self.conv1 = dwconv3x3_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            else:
                self.conv1 = dwconv5x5_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=activation)
            if self.use_se:
                self.se = SEBlock(
                    channels=mid_channels,
                    reduction=4,
                    round_mid=True,
                    out_activation="hsigmoid")
            self.conv2 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
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


class MobileNetV3FinalBlock(HybridBlock):
    """
    MobileNetV3 final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_se : bool
        Whether to use SE-module.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_se,
                 bn_use_global_stats=False,
                 **kwargs):
        super(MobileNetV3FinalBlock, self).__init__(**kwargs)
        self.use_se = use_se

        with self.name_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation="hswish")
            if self.use_se:
                self.se = SEBlock(
                    channels=out_channels,
                    reduction=4,
                    round_mid=True,
                    out_activation="hsigmoid")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_se:
            x = self.se(x)
        return x


class MobileNetV3Classifier(HybridBlock):
    """
    MobileNetV3 classifier.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 dropout_rate,
                 **kwargs):
        super(MobileNetV3Classifier, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)

        with self.name_scope():
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.activ = HSwish()
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_rate)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv2(x)
        return x


class MobileNetV3(HybridBlock):
    """
    MobileNetV3 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    exp_channels : list of list of int
        Number of middle (expanded) channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    use_relu : list of list of int/bool
        Using ReLU activation flag for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
    final_use_se : bool
        Whether to use SE-module in the final block.
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
                 exp_channels,
                 init_block_channels,
                 final_block_channels,
                 classifier_mid_channels,
                 kernels3,
                 use_relu,
                 use_se,
                 first_stride,
                 final_use_se,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(MobileNetV3, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats,
                activation="hswish"))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        exp_channels_ij = exp_channels[i][j]
                        strides = 2 if (j == 0) and ((i != 0) or first_stride) else 1
                        use_kernel3 = kernels3[i][j] == 1
                        activation = "relu" if use_relu[i][j] == 1 else "hswish"
                        use_se_flag = use_se[i][j] == 1
                        stage.add(MobileNetV3Unit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            exp_channels=exp_channels_ij,
                            use_kernel3=use_kernel3,
                            strides=strides,
                            activation=activation,
                            use_se=use_se_flag,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(MobileNetV3FinalBlock(
                in_channels=in_channels,
                out_channels=final_block_channels,
                use_se=final_use_se,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = final_block_channels
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(MobileNetV3Classifier(
                in_channels=in_channels,
                out_channels=classes,
                mid_channels=classifier_mid_channels,
                dropout_rate=0.2))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_mobilenetv3(version,
                    width_scale,
                    model_name=None,
                    pretrained=False,
                    ctx=cpu(),
                    root=os.path.join("~", ".mxnet", "models"),
                    **kwargs):
    """
    Create MobileNetV3 model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('small' or 'large').
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

    if version == "small":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40, 48, 48], [96, 96, 96]]
        exp_channels = [[16], [72, 88], [96, 240, 240, 120, 144], [288, 576, 576]]
        kernels3 = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_relu = [[1], [1, 1], [0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[1], [0, 0], [1, 1, 1, 1, 1], [1, 1, 1]]
        first_stride = True
        final_block_channels = 576
    elif version == "large":
        init_block_channels = 16
        channels = [[16], [24, 24], [40, 40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160]]
        exp_channels = [[16], [64, 72], [72, 120, 120], [240, 200, 184, 184, 480, 672], [672, 960, 960]]
        kernels3 = [[1], [1, 1], [0, 0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0]]
        use_relu = [[1], [1, 1], [1, 1, 1], [0, 0, 0, 0, 0, 0], [0, 0, 0]]
        use_se = [[0], [0, 0], [1, 1, 1], [0, 0, 0, 0, 1, 1], [1, 1, 1]]
        first_stride = False
        final_block_channels = 960
    else:
        raise ValueError("Unsupported MobileNetV3 version {}".format(version))

    final_use_se = False
    classifier_mid_channels = 1280

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale) for cij in ci] for ci in channels]
        exp_channels = [[round_channels(cij * width_scale) for cij in ci] for ci in exp_channels]
        init_block_channels = round_channels(init_block_channels * width_scale)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale)

    net = MobileNetV3(
        channels=channels,
        exp_channels=exp_channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        use_relu=use_relu,
        use_se=use_se,
        first_stride=first_stride,
        final_use_se=final_use_se,
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


def mobilenetv3_small_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.35, model_name="mobilenetv3_small_w7d20", **kwargs)


def mobilenetv3_small_wd2(**kwargs):
    """
    MobileNetV3 Small 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.5, model_name="mobilenetv3_small_wd2", **kwargs)


def mobilenetv3_small_w3d4(**kwargs):
    """
    MobileNetV3 Small 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=0.75, model_name="mobilenetv3_small_w3d4", **kwargs)


def mobilenetv3_small_w1(**kwargs):
    """
    MobileNetV3 Small 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.0, model_name="mobilenetv3_small_w1", **kwargs)


def mobilenetv3_small_w5d4(**kwargs):
    """
    MobileNetV3 Small 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="small", width_scale=1.25, model_name="mobilenetv3_small_w5d4", **kwargs)


def mobilenetv3_large_w7d20(**kwargs):
    """
    MobileNetV3 Small 224/0.35 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.35, model_name="mobilenetv3_small_w7d20", **kwargs)


def mobilenetv3_large_wd2(**kwargs):
    """
    MobileNetV3 Large 224/0.5 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.5, model_name="mobilenetv3_large_wd2", **kwargs)


def mobilenetv3_large_w3d4(**kwargs):
    """
    MobileNetV3 Large 224/0.75 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=0.75, model_name="mobilenetv3_large_w3d4", **kwargs)


def mobilenetv3_large_w1(**kwargs):
    """
    MobileNetV3 Large 224/1.0 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.0, model_name="mobilenetv3_large_w1", **kwargs)


def mobilenetv3_large_w5d4(**kwargs):
    """
    MobileNetV3 Large 224/1.25 model from 'Searching for MobileNetV3,' https://arxiv.org/abs/1905.02244.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_mobilenetv3(version="large", width_scale=1.25, model_name="mobilenetv3_large_w5d4", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        mobilenetv3_small_w7d20,
        mobilenetv3_small_wd2,
        mobilenetv3_small_w3d4,
        mobilenetv3_small_w1,
        mobilenetv3_small_w5d4,
        mobilenetv3_large_w7d20,
        mobilenetv3_large_wd2,
        mobilenetv3_large_w3d4,
        mobilenetv3_large_w1,
        mobilenetv3_large_w5d4,
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
        assert (model != mobilenetv3_small_w7d20 or weight_count == 2159600)
        assert (model != mobilenetv3_small_wd2 or weight_count == 2288976)
        assert (model != mobilenetv3_small_w3d4 or weight_count == 2581312)
        assert (model != mobilenetv3_small_w1 or weight_count == 2945288)
        assert (model != mobilenetv3_small_w5d4 or weight_count == 3643632)
        assert (model != mobilenetv3_large_w7d20 or weight_count == 2943080)
        assert (model != mobilenetv3_large_wd2 or weight_count == 3334896)
        assert (model != mobilenetv3_large_w3d4 or weight_count == 4263496)
        assert (model != mobilenetv3_large_w1 or weight_count == 5481752)
        assert (model != mobilenetv3_large_w5d4 or weight_count == 7459144)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
