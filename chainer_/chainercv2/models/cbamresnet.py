"""
    CBAM-ResNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.
"""

__all__ = ['CbamResNet', 'cbam_resnet18', 'cbam_resnet34', 'cbam_resnet50', 'cbam_resnet101', 'cbam_resnet152']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential, conv1x1_block, conv7x7_block
from .resnet import ResInitBlock, ResBlock, ResBottleneck


class MLP(Chain):
    """
    Multilayer perceptron block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(MLP, self).__init__()
        mid_channels = channels // reduction_ratio

        with self.init_scope():
            self.fc1 = L.Linear(
                in_size=channels,
                out_size=mid_channels)
            self.activ = F.relu
            self.fc2 = L.Linear(
                in_size=mid_channels,
                out_size=channels)

    def __call__(self, x):
        x = F.reshape(x, shape=(x.shape[0], -1))
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class ChannelGate(Chain):
    """
    CBAM channel gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(ChannelGate, self).__init__()
        with self.init_scope():
            self.mlp = MLP(
                channels=channels,
                reduction_ratio=reduction_ratio)

    def __call__(self, x):
        att1 = F.average_pooling_2d(x, ksize=x.shape[2:])
        att1 = self.mlp(att1)
        att2 = F.max_pooling_2d(x, ksize=x.shape[2:])
        att2 = self.mlp(att2)
        att = att1 + att2
        att = F.sigmoid(att)
        att = F.broadcast_to(F.expand_dims(F.expand_dims(att, axis=2), axis=3), x.shape)
        x = x * att
        return x


class SpatialGate(Chain):
    """
    CBAM spatial gate block.
    """
    def __init__(self):
        super(SpatialGate, self).__init__()
        with self.init_scope():
            self.conv = conv7x7_block(
                in_channels=2,
                out_channels=1,
                activation=None)

    def __call__(self, x):
        att1 = F.expand_dims(F.max(x, axis=1), axis=1)
        att2 = F.expand_dims(F.mean(x, axis=1), axis=1)
        att = F.concat((att1, att2), axis=1)
        att = self.conv(att)
        att = F.broadcast_to(F.sigmoid(att), x.shape)
        x = x * att
        return x


class CbamBlock(Chain):
    """
    CBAM attention block for CBAM-ResNet.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16):
        super(CbamBlock, self).__init__()
        with self.init_scope():
            self.ch_gate = ChannelGate(
                channels=channels,
                reduction_ratio=reduction_ratio)
            self.sp_gate = SpatialGate()

    def __call__(self, x):
        x = self.ch_gate(x)
        x = self.sp_gate(x)
        return x


class CbamResUnit(Chain):
    """
    CBAM-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(CbamResUnit, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            if bottleneck:
                self.body = ResBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    conv1_stride=False)
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
            self.cbam = CbamBlock(channels=out_channels)
            self.activ = F.relu

    def __call__(self, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = self.cbam(x)
        x = x + identity
        x = self.activ(x)
        return x


class CbamResNet(Chain):
    """
    CBAM-ResNet model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
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
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(CbamResNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", ResInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), CbamResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                bottleneck=bottleneck))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_resnet(blocks,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".chainer", "models"),
               **kwargs):
    """
    Create CBAM-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    use_se : bool
        Whether to use SE block.
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    if blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Unsupported CBAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CbamResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        **kwargs)

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


def cbam_resnet18(**kwargs):
    """
    CBAM-ResNet-18 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="cbam_resnet18", **kwargs)


def cbam_resnet34(**kwargs):
    """
    CBAM-ResNet-34 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="cbam_resnet34", **kwargs)


def cbam_resnet50(**kwargs):
    """
    CBAM-ResNet-50 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="cbam_resnet50", **kwargs)


def cbam_resnet101(**kwargs):
    """
    CBAM-ResNet-101 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="cbam_resnet101", **kwargs)


def cbam_resnet152(**kwargs):
    """
    CBAM-ResNet-152 model from 'CBAM: Convolutional Block Attention Module,' https://arxiv.org/abs/1807.06521.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="cbam_resnet152", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        # cbam_resnet18,
        # cbam_resnet34,
        cbam_resnet50,
        # cbam_resnet101,
        # cbam_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != cbam_resnet18 or weight_count == 11779392)
        assert (model != cbam_resnet34 or weight_count == 21960468)
        assert (model != cbam_resnet50 or weight_count == 28089624)
        assert (model != cbam_resnet101 or weight_count == 49330172)
        assert (model != cbam_resnet152 or weight_count == 66826848)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
