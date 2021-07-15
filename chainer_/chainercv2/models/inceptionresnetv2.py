"""
    InceptionResNetV2 for ImageNet-1K, implemented in Chainer.
    Original paper: 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.
"""

__all__ = ['InceptionResNetV2', 'inceptionresnetv2']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, SimpleSequential, Concurrent
from .inceptionv3 import AvgPoolBranch, Conv1x1Branch, ConvSeqBranch
from .inceptionresnetv1 import InceptionAUnit, InceptionBUnit, InceptionCUnit, ReductionAUnit, ReductionBUnit


class InceptBlock5b(Chain):
    """
    InceptionResNetV2 type Mixed-5b block.

    Parameters:
    ----------
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 bn_eps):
        super(InceptBlock5b, self).__init__()
        in_channels = 192

        with self.init_scope():
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "branch1", Conv1x1Branch(
                    in_channels=in_channels,
                    out_channels=96,
                    bn_eps=bn_eps))
                setattr(self.branches, "branch2", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(48, 64),
                    kernel_size_list=(1, 5),
                    strides_list=(1, 1),
                    padding_list=(0, 2),
                    bn_eps=bn_eps))
                setattr(self.branches, "branch3", ConvSeqBranch(
                    in_channels=in_channels,
                    out_channels_list=(64, 96, 96),
                    kernel_size_list=(1, 3, 3),
                    strides_list=(1, 1, 1),
                    padding_list=(0, 1, 1),
                    bn_eps=bn_eps))
                setattr(self.branches, "branch4", AvgPoolBranch(
                    in_channels=in_channels,
                    out_channels=64,
                    bn_eps=bn_eps,
                    count_include_pad=False))

    def __call__(self, x):
        x = self.branches(x)
        return x


class InceptInitBlock(Chain):
    """
    InceptionResNetV2 specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_eps : float
        Small float added to variance in Batch norm.
    """
    def __init__(self,
                 in_channels,
                 bn_eps):
        super(InceptInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=32,
                stride=2,
                pad=0,
                bn_eps=bn_eps)
            self.conv2 = conv3x3_block(
                in_channels=32,
                out_channels=32,
                stride=1,
                pad=0,
                bn_eps=bn_eps)
            self.conv3 = conv3x3_block(
                in_channels=32,
                out_channels=64,
                stride=1,
                pad=1,
                bn_eps=bn_eps)
            self.pool1 = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=0,
                cover_all=False)
            self.conv4 = conv1x1_block(
                in_channels=64,
                out_channels=80,
                stride=1,
                pad=0,
                bn_eps=bn_eps)
            self.conv5 = conv3x3_block(
                in_channels=80,
                out_channels=192,
                stride=1,
                pad=0,
                bn_eps=bn_eps)
            self.pool2 = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=0,
                cover_all=False)
            self.block = InceptBlock5b(bn_eps=bn_eps)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.block(x)
        return x


class InceptionResNetV2(Chain):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    dropout_rate : float, default 0.0
        Fraction of the input units to drop. Must be a number between 0 and 1.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (299, 299)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 dropout_rate=0.0,
                 bn_eps=1e-5,
                 in_channels=3,
                 in_size=(299, 299),
                 classes=1000):
        super(InceptionResNetV2, self).__init__()
        self.in_size = in_size
        self.classes = classes
        layers = [10, 21, 11]
        in_channels_list = [320, 1088, 2080]
        normal_out_channels_list = [[32, 32, 32, 32, 48, 64], [192, 128, 160, 192], [192, 192, 224, 256]]
        reduction_out_channels_list = [[384, 256, 256, 384], [256, 384, 256, 288, 256, 288, 320]]

        normal_units = [InceptionAUnit, InceptionBUnit, InceptionCUnit]
        reduction_units = [ReductionAUnit, ReductionBUnit]

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", InceptInitBlock(
                    in_channels=in_channels,
                    bn_eps=bn_eps))
                in_channels = in_channels_list[0]
                for i, layers_per_stage in enumerate(layers):
                    stage = SimpleSequential()
                    with stage.init_scope():
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
                            setattr(stage, "unit{}".format(j + 1), unit(
                                in_channels=in_channels,
                                out_channels_list=out_channels_list_per_stage,
                                bn_eps=bn_eps,
                                **unit_kwargs))
                            if (j == 0) and (i != 0):
                                in_channels = in_channels_list[i]
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, 'final_conv', conv1x1_block(
                    in_channels=2080,
                    out_channels=1536,
                    bn_eps=bn_eps))
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=8,
                    stride=1))

            in_channels = 1536
            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                if dropout_rate > 0.0:
                    setattr(self.output, "dropout", partial(
                        F.dropout,
                        ratio=dropout_rate))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_inceptionresnetv2(model_name=None,
                          pretrained=False,
                          root=os.path.join("~", ".chainer", "models"),
                          **kwargs):
    """
    Create InceptionResNetV2 model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    net = InceptionResNetV2(**kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def inceptionresnetv2(**kwargs):
    """
    InceptionResNetV2 model from 'Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning,'
    https://arxiv.org/abs/1602.07261.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_inceptionresnetv2(model_name="inceptionresnetv2", bn_eps=1e-3, **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        inceptionresnetv2,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != inceptionresnetv2 or weight_count == 55843464)

        x = np.zeros((1, 3, 299, 299), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
