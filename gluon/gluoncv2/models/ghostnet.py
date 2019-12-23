"""
    GhostNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.
"""

__all__ = ['GhostNet', 'ghostnet']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import round_channels, conv1x1, conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv5x5_block,\
    dwsconv3x3_block, SEBlock


class GhostHSigmoid(HybridBlock):
    """
    Approximated sigmoid function, specific for GhostNet.
    """
    def __init__(self, **kwargs):
        super(GhostHSigmoid, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0.0, 1.0)


class GhostConvBlock(HybridBlock):
    """
    GhostNet specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    activation : function or str or None, default default nn.Activation('relu')
        Activation function or name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats=False,
                 activation=(lambda: nn.Activation("relu")),
                 **kwargs):
        super(GhostConvBlock, self).__init__(**kwargs)
        main_out_channels = math.ceil(0.5 * out_channels)
        cheap_out_channels = out_channels - main_out_channels

        with self.name_scope():
            self.main_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=main_out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation)
            self.cheap_conv = dwconv3x3_block(
                in_channels=main_out_channels,
                out_channels=cheap_out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation)

    def hybrid_forward(self, F, x):
        x = self.main_conv(x)
        y = self.cheap_conv(x)
        return F.concat(x, y, dim=1)


class GhostExpBlock(HybridBlock):
    """
    GhostNet expansion block for residual path in GhostNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : float
        Expansion factor.
    use_se : bool
        Whether to use SE-module.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_kernel3,
                 exp_factor,
                 use_se,
                 bn_use_global_stats=False,
                 **kwargs):
        super(GhostExpBlock, self).__init__(**kwargs)
        self.use_dw_conv = (strides != 1)
        self.use_se = use_se
        mid_channels = int(math.ceil(exp_factor * in_channels))

        with self.name_scope():
            self.exp_conv = GhostConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            if self.use_dw_conv:
                dw_conv_class = dwconv3x3_block if use_kernel3 else dwconv5x5_block
                self.dw_conv = dw_conv_class(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            if self.use_se:
                self.se = SEBlock(
                    channels=mid_channels,
                    reduction=4,
                    out_activation=GhostHSigmoid())
            self.pw_conv = GhostConvBlock(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.exp_conv(x)
        if self.use_dw_conv:
            x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class GhostUnit(HybridBlock):
    """
    GhostNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    use_kernel3 : bool
        Whether to use 3x3 (instead of 5x5) kernel.
    exp_factor : float
        Expansion factor.
    use_se : bool
        Whether to use SE-module.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 use_kernel3,
                 exp_factor,
                 use_se,
                 bn_use_global_stats=False,
                 **kwargs):
        super(GhostUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = GhostExpBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                use_kernel3=use_kernel3,
                exp_factor=exp_factor,
                use_se=use_se,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = dwsconv3x3_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    pw_activation=None)

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class GhostClassifier(HybridBlock):
    """
    GhostNet classifier.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 **kwargs):
        super(GhostClassifier, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.conv2 = conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                use_bias=True)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class GhostNet(HybridBlock):
    """
    GhostNet model from 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    classifier_mid_channels : int
        Number of middle channels for classifier.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 5x5) kernel for each unit.
    exp_factors : list of list of int
        Expansion factor for each unit.
    use_se : list of list of int/bool
        Using SE-block flag for each unit.
    first_stride : bool
        Whether to use stride for the first stage.
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
                 classifier_mid_channels,
                 kernels3,
                 exp_factors,
                 use_se,
                 first_stride,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(GhostNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and ((i != 0) or first_stride) else 1
                        use_kernel3 = kernels3[i][j] == 1
                        exp_factor = exp_factors[i][j]
                        use_se_flag = use_se[i][j] == 1
                        stage.add(GhostUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            use_kernel3=use_kernel3,
                            exp_factor=exp_factor,
                            use_se=use_se_flag,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=final_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = final_block_channels
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(GhostClassifier(
                in_channels=in_channels,
                out_channels=classes,
                mid_channels=classifier_mid_channels))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_ghostnet(width_scale=1.0,
                 model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", "models"),
                 **kwargs):
    """
    Create GhostNet model with specific parameters.

    Parameters:
    ----------
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
    init_block_channels = 16
    channels = [[16], [24, 24], [40, 40], [80, 80, 80, 80, 112, 112], [160, 160, 160, 160, 160]]
    kernels3 = [[1], [1, 1], [0, 0], [1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0]]
    exp_factors = [[1], [3, 3], [3, 3], [6, 2.5, 2.3, 2.3, 6, 6], [6, 6, 6, 6, 6]]
    use_se = [[0], [0, 0], [1, 1], [0, 0, 0, 0, 1, 1], [1, 0, 1, 0, 1]]
    final_block_channels = 960
    classifier_mid_channels = 1280
    first_stride = False

    if width_scale != 1.0:
        channels = [[round_channels(cij * width_scale, divisor=4) for cij in ci] for ci in channels]
        init_block_channels = round_channels(init_block_channels * width_scale, divisor=4)
        if width_scale > 1.0:
            final_block_channels = round_channels(final_block_channels * width_scale, divisor=4)

    net = GhostNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        classifier_mid_channels=classifier_mid_channels,
        kernels3=kernels3,
        exp_factors=exp_factors,
        use_se=use_se,
        first_stride=first_stride,
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


def ghostnet(**kwargs):
    """
    GhostNet model from 'GhostNet: More Features from Cheap Operations,' https://arxiv.org/abs/1911.11907.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_ghostnet(model_name="ghostnet", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        ghostnet,
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
        assert (model != ghostnet or weight_count == 5180840)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
