"""
    EfficientNet-Edge for ImageNet-1K, implemented in Gluon.
    Original paper: 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.
"""

__all__ = ['EfficientNetEdge', 'efficientnet_edge_small_b', 'efficientnet_edge_medium_b', 'efficientnet_edge_large_b']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import round_channels, conv1x1_block, conv3x3_block, SEBlock
from .efficientnet import EffiInvResUnit, EffiInitBlock


class EffiEdgeResUnit(HybridBlock):
    """
    EfficientNet-Edge edge residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the second convolution layer.
    exp_factor : int
        Factor for expansion of channels.
    se_factor : int
        SE reduction factor for each unit.
    mid_from_in : bool
        Whether to use input channel count for middle channel count calculation.
    use_skip : bool
        Whether to use skip connection.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activation : str
        Name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 exp_factor,
                 se_factor,
                 mid_from_in,
                 use_skip,
                 bn_epsilon,
                 bn_use_global_stats,
                 activation,
                 **kwargs):
        super(EffiEdgeResUnit, self).__init__(**kwargs)
        self.residual = (in_channels == out_channels) and (strides == 1) and use_skip
        self.use_se = se_factor > 0
        mid_channels = in_channels * exp_factor if mid_from_in else out_channels * exp_factor

        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation)
            if self.use_se:
                self.se = SEBlock(
                    channels=mid_channels,
                    reduction=(exp_factor * se_factor),
                    mid_activation=activation)
            self.conv2 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=strides,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        if self.residual:
            identity = x
        x = self.conv1(x)
        if self.use_se:
            x = self.se(x)
        x = self.conv2(x)
        if self.residual:
            x = x + identity
        return x


class EfficientNetEdge(HybridBlock):
    """
    EfficientNet-Edge model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for initial unit.
    final_block_channels : int
        Number of output channels for the final block of the feature extractor.
    kernel_sizes : list of list of int
        Number of kernel sizes for each unit.
    strides_per_stage : list int
        Stride value for the first unit of each stage.
    expansion_factors : list of list of int
        Number of expansion factors for each unit.
    dropout_rate : float, default 0.2
        Fraction of the input units to drop. Must be a number between 0 and 1.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
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
                 kernel_sizes,
                 strides_per_stage,
                 expansion_factors,
                 dropout_rate=0.2,
                 tf_mode=False,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(EfficientNetEdge, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        activation = "relu"

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(EffiInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation,
                tf_mode=tf_mode))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                kernel_sizes_per_stage = kernel_sizes[i]
                expansion_factors_per_stage = expansion_factors[i]
                mid_from_in = (i != 0)
                use_skip = (i != 0)
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        kernel_size = kernel_sizes_per_stage[j]
                        expansion_factor = expansion_factors_per_stage[j]
                        strides = strides_per_stage[i] if (j == 0) else 1
                        if i < 3:
                            stage.add(EffiEdgeResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                exp_factor=expansion_factor,
                                se_factor=0,
                                mid_from_in=mid_from_in,
                                use_skip=use_skip,
                                bn_epsilon=bn_epsilon,
                                bn_use_global_stats=bn_use_global_stats,
                                activation=activation))
                        else:
                            stage.add(EffiInvResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                strides=strides,
                                exp_factor=expansion_factor,
                                se_factor=0,
                                bn_epsilon=bn_epsilon,
                                bn_use_global_stats=bn_use_global_stats,
                                activation=activation,
                                tf_mode=tf_mode))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(conv1x1_block(
                in_channels=in_channels,
                out_channels=final_block_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                activation=activation))
            in_channels = final_block_channels
            self.features.add(nn.GlobalAvgPool2D())

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            if dropout_rate > 0.0:
                self.output.add(nn.Dropout(rate=dropout_rate))
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_efficientnet_edge(version,
                          in_size,
                          tf_mode=False,
                          bn_epsilon=1e-5,
                          model_name=None,
                          pretrained=False,
                          ctx=cpu(),
                          root=os.path.join("~", ".mxnet", "models"),
                          **kwargs):
    """
    Create EfficientNet-Edge model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of EfficientNet ('small', 'medium', 'large').
    in_size : tuple of two ints
        Spatial size of the expected input image.
    tf_mode : bool, default False
        Whether to use TF-like mode.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    dropout_rate = 0.0
    if version == "small":
        assert (in_size == (224, 224))
        depth_factor = 1.0
        width_factor = 1.0
        # dropout_rate = 0.2
    elif version == "medium":
        assert (in_size == (240, 240))
        depth_factor = 1.1
        width_factor = 1.0
        # dropout_rate = 0.2
    elif version == "large":
        assert (in_size == (300, 300))
        depth_factor = 1.4
        width_factor = 1.2
        # dropout_rate = 0.3
    else:
        raise ValueError("Unsupported EfficientNet-Edge version {}".format(version))

    init_block_channels = 32
    layers = [1, 2, 4, 5, 4, 2]
    downsample = [1, 1, 1, 1, 0, 1]
    channels_per_layers = [24, 32, 48, 96, 144, 192]
    expansion_factors_per_layers = [4, 8, 8, 8, 8, 8]
    kernel_sizes_per_layers = [3, 3, 3, 5, 5, 5]
    strides_per_stage = [1, 2, 2, 2, 1, 2]
    final_block_channels = 1280

    layers = [int(math.ceil(li * depth_factor)) for li in layers]
    channels_per_layers = [round_channels(ci * width_factor) for ci in channels_per_layers]

    from functools import reduce
    channels = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                      zip(channels_per_layers, layers, downsample), [])
    kernel_sizes = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                          zip(kernel_sizes_per_layers, layers, downsample), [])
    expansion_factors = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(expansion_factors_per_layers, layers, downsample), [])
    strides_per_stage = reduce(lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
                               zip(strides_per_stage, layers, downsample), [])
    strides_per_stage = [si[0] for si in strides_per_stage]

    init_block_channels = round_channels(init_block_channels * width_factor)

    if width_factor > 1.0:
        assert (int(final_block_channels * width_factor) == round_channels(final_block_channels * width_factor))
        final_block_channels = round_channels(final_block_channels * width_factor)

    net = EfficientNetEdge(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        kernel_sizes=kernel_sizes,
        strides_per_stage=strides_per_stage,
        expansion_factors=expansion_factors,
        dropout_rate=dropout_rate,
        tf_mode=tf_mode,
        bn_epsilon=bn_epsilon,
        in_size=in_size,
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


def efficientnet_edge_small_b(in_size=(224, 224), **kwargs):
    """
    EfficientNet-Edge-Small-b model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters:
    ----------
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet_edge(version="small", in_size=in_size, tf_mode=True, bn_epsilon=1e-3,
                                 model_name="efficientnet_edge_small_b", **kwargs)


def efficientnet_edge_medium_b(in_size=(240, 240), **kwargs):
    """
    EfficientNet-Edge-Medium-b model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters:
    ----------
    in_size : tuple of two ints, default (240, 240)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet_edge(version="medium", in_size=in_size, tf_mode=True, bn_epsilon=1e-3,
                                 model_name="efficientnet_edge_medium_b", **kwargs)


def efficientnet_edge_large_b(in_size=(300, 300), **kwargs):
    """
    EfficientNet-Edge-Large-b model from 'EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,'
    https://arxiv.org/abs/1905.11946.

    Parameters:
    ----------
    in_size : tuple of two ints, default (300, 300)
        Spatial size of the expected input image.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_efficientnet_edge(version="large", in_size=in_size, tf_mode=True, bn_epsilon=1e-3,
                                 model_name="efficientnet_edge_large_b", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        efficientnet_edge_small_b,
        efficientnet_edge_medium_b,
        efficientnet_edge_large_b,
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
        assert (model != efficientnet_edge_small_b or weight_count == 5438392)
        assert (model != efficientnet_edge_medium_b or weight_count == 6899496)
        assert (model != efficientnet_edge_large_b or weight_count == 10589712)

        x = mx.nd.zeros((1, 3, net.in_size[0], net.in_size[1]), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
