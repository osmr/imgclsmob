"""
    InceptionResNetV2 for ImageNet-1K, implemented in Gluon.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionResNetV2', 'inceptionresnetv2']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from .common import conv1x1_block, conv3x3_block
from .inceptionv3 import AvgPoolBranch, Conv1x1Branch, ConvSeqBranch
from .inceptionresnetv1 import InceptionAUnit, InceptionBUnit, InceptionCUnit, ReductionAUnit, ReductionBUnit


class InceptBlock5b(HybridBlock):
    """
    InceptionResNetV2 type Mixed-5b block.

    Parameters:
    ----------
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 bn_epsilon,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptBlock5b, self).__init__(**kwargs)
        in_channels = 192

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=96,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(48, 64),
                kernel_size_list=(1, 5),
                strides_list=(1, 1),
                padding_list=(0, 2),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=(64, 96, 96),
                kernel_size_list=(1, 3, 3),
                strides_list=(1, 1, 1),
                padding_list=(0, 1, 1),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(AvgPoolBranch(
                in_channels=in_channels,
                out_channels=64,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptInitBlock(HybridBlock):
    """
    InceptionResNetV2 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_epsilon,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=32,
                strides=2,
                padding=0,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=32,
                out_channels=32,
                strides=1,
                padding=0,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv3x3_block(
                in_channels=32,
                out_channels=64,
                strides=1,
                padding=1,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats)
            self.pool1 = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=0)
            self.conv4 = conv1x1_block(
                in_channels=64,
                out_channels=80,
                strides=1,
                padding=0,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats)
            self.conv5 = conv3x3_block(
                in_channels=80,
                out_channels=192,
                strides=1,
                padding=0,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats)
            self.pool2 = nn.MaxPool2D(
                pool_size=3,
                strides=2,
                padding=0)
            self.block = InceptBlock5b(bn_epsilon=bn_epsilon, bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.block(x)
        return x


class InceptionResNetV2(HybridBlock):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    bn_epsilon : float, default 1e-5
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 dropout_rate=0.0,
                 bn_epsilon=1e-5,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000,
                 **kwargs):
        super(InceptionResNetV2, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        layers = [10, 21, 11]
        in_channels_list = [320, 1088, 2080]
        normal_out_channels_list = [[32, 32, 32, 32, 48, 64], [192, 128, 160, 192], [192, 192, 224, 256]]
        reduction_out_channels_list = [[384, 256, 256, 384], [256, 384, 256, 288, 256, 288, 320]]

        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(InceptInitBlock(
                in_channels=in_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = in_channels_list[0]
            for i, layers_per_stage in enumerate(layers):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j in range(layers_per_stage):
                        if (j == 0) and (i != 0):
                            unit = reduction_units[i - 1]
                            out_channels_list_per_stage = reduction_out_channels_list[i - 1]
                        else:
                            unit = normal_units[i]
                            out_channels_list_per_stage = normal_out_channels_list[i]
                        if (i == len(layers) - 1) and (j == layers_per_stage - 1):
                            unit_kwargs = {"scale": 1.0, "activate": False}
                        else:
                            unit_kwargs = {}
                        stage.add(unit(
                            in_channels=in_channels,
                            out_channels_list=out_channels_list_per_stage,
                            bn_epsilon=bn_epsilon,
                            bn_use_global_stats=bn_use_global_stats,
                            **unit_kwargs))
                        if (j == 0) and (i != 0):
                            in_channels = in_channels_list[i]
                self.features.add(stage)
            self.features.add(conv1x1_block(
                in_channels=2080,
                out_channels=1536,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.features.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            if dropout_rate > 0.0:
                self.output.add(nn.Dropout(rate=dropout_rate))
            self.output.add(nn.Dense(
                units=classes,
                in_units=1536))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_inceptionresnetv2(model_name=None,
                          pretrained=False,
                          ctx=cpu(),
                          root=os.path.join("~", ".mxnet", "models"),
                          **kwargs):
    """
    Create InceptionResNetV2 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    net = InceptionResNetV2(**kwargs)

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


def inceptionresnetv2(**kwargs):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_inceptionresnetv2(model_name="inceptionresnetv2", bn_epsilon=1e-3, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        inceptionresnetv2,
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
        assert (model != inceptionresnetv2 or weight_count == 55843464)

        x = mx.nd.zeros((1, 3, 299, 299), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
