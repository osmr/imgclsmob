"""
    InceptionResNetV1 for ImageNet-1K, implemented in Gluon.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionResNetV1', 'inceptionresnetv1', 'InceptionAUnit', 'InceptionBUnit', 'InceptionCUnit',
           'ReductionAUnit', 'ReductionBUnit']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import HybridConcurrent
from .common import conv1x1, conv1x1_block, conv3x3_block, BatchNormExtra
from .inceptionv3 import MaxPoolBranch, Conv1x1Branch, ConvSeqBranch


class InceptionAUnit(HybridBlock):
    """
    InceptionResNetV1 type Inception-A unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_epsilon,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptionAUnit, self).__init__(**kwargs)
        self.scale = 0.17

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=out_channels_list[0],
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[1:3],
                kernel_size_list=(1, 3),
                strides_list=(1, 1),
                padding_list=(0, 1),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[3:6],
                kernel_size_list=(1, 3, 3),
                strides_list=(1, 1, 1),
                padding_list=(0, 1, 1),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            conv_in_channels = out_channels_list[0] + out_channels_list[2] + out_channels_list[5]
            self.conv = conv1x1(
                in_channels=conv_in_channels,
                out_channels=in_channels,
                use_bias=True)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        identity = x
        x = self.branches(x)
        x = self.conv(x)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class InceptionBUnit(HybridBlock):
    """
    InceptionResNetV1 type Inception-B unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_epsilon,
                 bn_use_global_stats,
                 **kwargs):
        super(InceptionBUnit, self).__init__(**kwargs)
        self.scale = 0.10

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=out_channels_list[0],
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[1:4],
                kernel_size_list=(1, (1, 7), (7, 1)),
                strides_list=(1, 1, 1),
                padding_list=(0, (0, 3), (3, 0)),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            conv_in_channels = out_channels_list[0] + out_channels_list[3]
            self.conv = conv1x1(
                in_channels=conv_in_channels,
                out_channels=in_channels,
                use_bias=True)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        identity = x
        x = self.branches(x)
        x = self.conv(x)
        x = self.scale * x + identity
        x = self.activ(x)
        return x


class InceptionCUnit(HybridBlock):
    """
    InceptionResNetV1 type Inception-C unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    scale : float, default 0.2
        Scale value for residual branch.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_epsilon,
                 bn_use_global_stats=False,
                 scale=0.2,
                 activate=True,
                 **kwargs):
        super(InceptionCUnit, self).__init__(**kwargs)
        self.activate = activate
        self.scale = scale

        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(Conv1x1Branch(
                in_channels=in_channels,
                out_channels=out_channels_list[0],
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[1:4],
                kernel_size_list=(1, (1, 3), (3, 1)),
                strides_list=(1, 1, 1),
                padding_list=(0, (0, 1), (1, 0)),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            conv_in_channels = out_channels_list[0] + out_channels_list[3]
            self.conv = conv1x1(
                in_channels=conv_in_channels,
                out_channels=in_channels,
                use_bias=True)
            if self.activate:
                self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        identity = x
        x = self.branches(x)
        x = self.conv(x)
        x = self.scale * x + identity
        if self.activate:
            x = self.activ(x)
        return x


class ReductionAUnit(HybridBlock):
    """
    InceptionResNetV1 type Reduction-A unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_epsilon,
                 bn_use_global_stats,
                 **kwargs):
        super(ReductionAUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[0:1],
                kernel_size_list=(3,),
                strides_list=(2,),
                padding_list=(0,),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[1:4],
                kernel_size_list=(1, 3, 3),
                strides_list=(1, 1, 2),
                padding_list=(0, 1, 0),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(MaxPoolBranch())

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class ReductionBUnit(HybridBlock):
    """
    InceptionResNetV1 type Reduction-B unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        List for numbers of output channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 bn_epsilon,
                 bn_use_global_stats,
                 **kwargs):
        super(ReductionBUnit, self).__init__(**kwargs)
        with self.name_scope():
            self.branches = HybridConcurrent(axis=1, prefix="")
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[0:2],
                kernel_size_list=(1, 3),
                strides_list=(1, 2),
                padding_list=(0, 0),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[2:4],
                kernel_size_list=(1, 3),
                strides_list=(1, 2),
                padding_list=(0, 0),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(ConvSeqBranch(
                in_channels=in_channels,
                out_channels_list=out_channels_list[4:7],
                kernel_size_list=(1, 3, 3),
                strides_list=(1, 1, 2),
                padding_list=(0, 1, 0),
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats))
            self.branches.add(MaxPoolBranch())

    def hybrid_forward(self, F, x):
        x = self.branches(x)
        return x


class InceptInitBlock(HybridBlock):
    """
    InceptionResNetV1 specific initial block.

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
            self.pool = nn.MaxPool2D(
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
            self.conv6 = conv3x3_block(
                in_channels=192,
                out_channels=256,
                strides=2,
                padding=0,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class InceptHead(HybridBlock):
    """
    InceptionResNetV1 specific classification block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_epsilon : float
        Small float added to variance in Batch norm.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    dropout_rate : float
        Fraction of the input units to drop. Must be a number between 0 and 1.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 bn_epsilon,
                 bn_use_global_stats,
                 dropout_rate,
                 classes,
                 **kwargs):
        super(InceptHead, self).__init__(**kwargs)
        self.use_dropout = (dropout_rate != 0.0)

        with self.name_scope():
            self.flatten = nn.Flatten()
            if self.use_dropout:
                self.dropout = nn.Dropout(rate=dropout_rate)
            self.fc1 = nn.Dense(
                units=512,
                use_bias=False,
                in_units=in_channels)
            self.bn = BatchNormExtra(
                in_channels=512,
                epsilon=bn_epsilon,
                use_global_stats=bn_use_global_stats)
            self.fc2 = nn.Dense(
                units=classes,
                in_units=512)

    def hybrid_forward(self, F, x):
        x = self.flatten(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x


class InceptionResNetV1(HybridBlock):
    """
    InceptionResNetV1 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
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
        super(InceptionResNetV1, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        layers = [5, 11, 7]
        in_channels_list = [256, 896, 1792]
        normal_out_channels_list = [[32, 32, 32, 32, 32, 32], [128, 128, 128, 128], [192, 192, 192, 192]]
        reduction_out_channels_list = [[384, 192, 192, 256], [256, 384, 256, 256, 256, 256, 256]]

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
            self.features.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))

            self.output = InceptHead(
                in_channels=in_channels,
                bn_epsilon=bn_epsilon,
                bn_use_global_stats=bn_use_global_stats,
                dropout_rate=dropout_rate,
                classes=classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_inceptionresnetv1(model_name=None,
                          pretrained=False,
                          ctx=cpu(),
                          root=os.path.join("~", ".mxnet", "models"),
                          **kwargs):
    """
    Create InceptionResNetV1 model with specific parameters.

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
    net = InceptionResNetV1(**kwargs)

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


def inceptionresnetv1(**kwargs):
    """
    InceptionResNetV1 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
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
    return get_inceptionresnetv1(model_name="inceptionresnetv1", bn_epsilon=1e-3, **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        inceptionresnetv1,
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
        assert (model != inceptionresnetv1 or weight_count == 23995624)

        x = mx.nd.zeros((1, 3, 299, 299), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
