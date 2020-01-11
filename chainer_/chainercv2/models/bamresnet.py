"""
    BAM-ResNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.
"""

__all__ = ['BamResNet', 'bam_resnet18', 'bam_resnet34', 'bam_resnet50', 'bam_resnet101', 'bam_resnet152']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential, conv1x1, conv1x1_block, conv3x3_block
from .resnet import ResInitBlock, ResUnit


class DenseBlock(Chain):
    """
    Standard dense block with Batch normalization and ReLU activation.

    Parameters:
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DenseBlock, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(
                in_size=in_channels,
                out_size=out_channels)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)
            self.activ = F.relu

    def __call__(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class ChannelGate(Chain):
    """
    BAM channel gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    num_layers : int, default 1
        Number of dense blocks.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 num_layers=1):
        super(ChannelGate, self).__init__()
        mid_channels = channels // reduction_ratio

        with self.init_scope():
            self.init_fc = DenseBlock(
                in_channels=channels,
                out_channels=mid_channels)
            self.main_fcs = SimpleSequential()
            with self.main_fcs.init_scope():
                for i in range(num_layers - 1):
                    setattr(self.main_fcs, "fc{}".format(i + 1), DenseBlock(
                        in_channels=mid_channels,
                        out_channels=mid_channels))
            self.final_fc = L.Linear(
                in_size=mid_channels,
                out_size=channels)

    def __call__(self, x):
        input_shape = x.shape
        x = F.average_pooling_2d(x, ksize=x.shape[2:])
        x = F.reshape(x, shape=(x.shape[0], -1))
        x = self.init_fc(x)
        x = self.main_fcs(x)
        x = self.final_fc(x)
        x = F.broadcast_to(F.expand_dims(F.expand_dims(x, axis=2), axis=3), input_shape)
        return x


class SpatialGate(Chain):
    """
    BAM spatial gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    num_dil_convs : int, default 2
        Number of dilated convolutions.
    dilate : int, default 4
        Dilation/padding value for corresponding convolutions.
    """
    def __init__(self,
                 channels,
                 reduction_ratio=16,
                 num_dil_convs=2,
                 dilate=4):
        super(SpatialGate, self).__init__()
        mid_channels = channels // reduction_ratio

        with self.init_scope():
            self.init_conv = conv1x1_block(
                in_channels=channels,
                out_channels=mid_channels,
                stride=1,
                use_bias=True)
            self.dil_convs = SimpleSequential()
            with self.dil_convs.init_scope():
                for i in range(num_dil_convs):
                    setattr(self.dil_convs, "conv{}".format(i + 1), conv3x3_block(
                        in_channels=mid_channels,
                        out_channels=mid_channels,
                        stride=1,
                        pad=dilate,
                        dilate=dilate,
                        use_bias=True))
            self.final_conv = conv1x1(
                in_channels=mid_channels,
                out_channels=1,
                stride=1,
                use_bias=True)

    def __call__(self, x):
        input_shape = x.shape
        x = self.init_conv(x)
        x = self.dil_convs(x)
        x = self.final_conv(x)
        x = F.broadcast_to(x, input_shape)
        return x


class BamBlock(Chain):
    """
    BAM attention block for BAM-ResNet.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    """
    def __init__(self,
                 channels):
        super(BamBlock, self).__init__()
        with self.init_scope():
            self.ch_att = ChannelGate(channels=channels)
            self.sp_att = SpatialGate(channels=channels)

    def __call__(self, x):
        att = 1 + F.sigmoid(self.ch_att(x) * self.sp_att(x))
        x = x * att
        return x


class BamResUnit(Chain):
    """
    BAM-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 bottleneck):
        super(BamResUnit, self).__init__()
        self.use_bam = (stride != 1)

        with self.init_scope():
            if self.use_bam:
                self.bam = BamBlock(channels=in_channels)
            self.res_unit = ResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bottleneck=bottleneck,
                conv1_stride=False)

    def __call__(self, x):
        if self.use_bam:
            x = self.bam(x)
        x = self.res_unit(x)
        return x


class BamResNet(Chain):
    """
    BAM-ResNet model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

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
        super(BamResNet, self).__init__()
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
                            setattr(stage, "unit{}".format(j + 1), BamResUnit(
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
    Create BAM-ResNet model with specific parameters.

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
        raise ValueError("Unsupported BAM-ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = BamResNet(
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


def bam_resnet18(**kwargs):
    """
    BAM-ResNet-18 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="bam_resnet18", **kwargs)


def bam_resnet34(**kwargs):
    """
    BAM-ResNet-34 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="bam_resnet34", **kwargs)


def bam_resnet50(**kwargs):
    """
    BAM-ResNet-50 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="bam_resnet50", **kwargs)


def bam_resnet101(**kwargs):
    """
    BAM-ResNet-101 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="bam_resnet101", **kwargs)


def bam_resnet152(**kwargs):
    """
    BAM-ResNet-152 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="bam_resnet152", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        bam_resnet18,
        bam_resnet34,
        bam_resnet50,
        bam_resnet101,
        bam_resnet152,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bam_resnet18 or weight_count == 11712503)
        assert (model != bam_resnet34 or weight_count == 21820663)
        assert (model != bam_resnet50 or weight_count == 25915099)
        assert (model != bam_resnet101 or weight_count == 44907227)
        assert (model != bam_resnet152 or weight_count == 60550875)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
