"""
    BAM-ResNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.
"""

__all__ = ['BamResNet', 'bam_resnet18', 'bam_resnet34', 'bam_resnet50', 'bam_resnet101', 'bam_resnet152']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv1x1_block, conv3x3_block
from .resnet import ResInitBlock, ResUnit


class DenseBlock(HybridBlock):
    """
    Standard dense block with Batch normalization and ReLU activation.

    Parameters:
    ----------
    in_channels : int
        Number of input features.
    out_channels : int
        Number of output features.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.fc = nn.Dense(
                units=out_channels,
                in_units=in_channels)
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class ChannelGate(HybridBlock):
    """
    BAM channel gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    num_layers : int, default 1
        Number of dense blocks.
    """
    def __init__(self,
                 channels,
                 bn_use_global_stats,
                 reduction_ratio=16,
                 num_layers=1,
                 **kwargs):
        super(ChannelGate, self).__init__(**kwargs)
        mid_channels = channels // reduction_ratio

        with self.name_scope():
            self.pool = nn.GlobalAvgPool2D()
            self.flatten = nn.Flatten()
            self.init_fc = DenseBlock(
                in_channels=channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.main_fcs = nn.HybridSequential(prefix="")
            for i in range(num_layers - 1):
                self.main_fcs.add(DenseBlock(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    bn_use_global_stats=bn_use_global_stats))
            self.final_fc = nn.Dense(
                units=channels,
                in_units=mid_channels)

    def hybrid_forward(self, F, x):
        input = x
        x = self.pool(x)
        x = self.flatten(x)
        x = self.init_fc(x)
        x = self.main_fcs(x)
        x = self.final_fc(x)
        x = x.expand_dims(2).expand_dims(3).broadcast_like(input)
        return x


class SpatialGate(HybridBlock):
    """
    BAM spatial gate block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    reduction_ratio : int, default 16
        Channel reduction ratio.
    num_dil_convs : int, default 2
        Number of dilated convolutions.
    dilation : int, default 4
        Dilation/padding value for corresponding convolutions.
    """
    def __init__(self,
                 channels,
                 bn_use_global_stats,
                 reduction_ratio=16,
                 num_dil_convs=2,
                 dilation=4,
                 **kwargs):
        super(SpatialGate, self).__init__(**kwargs)
        mid_channels = channels // reduction_ratio

        with self.name_scope():
            self.init_conv = conv1x1_block(
                in_channels=channels,
                out_channels=mid_channels,
                strides=1,
                use_bias=True,
                bn_use_global_stats=bn_use_global_stats)
            self.dil_convs = nn.HybridSequential(prefix="")
            for i in range(num_dil_convs):
                self.dil_convs.add(conv3x3_block(
                    in_channels=mid_channels,
                    out_channels=mid_channels,
                    strides=1,
                    padding=dilation,
                    dilation=dilation,
                    use_bias=True,
                    bn_use_global_stats=bn_use_global_stats))
            self.final_conv = conv1x1(
                in_channels=mid_channels,
                out_channels=1,
                strides=1,
                use_bias=True)

    def hybrid_forward(self, F, x):
        input = x
        x = self.init_conv(x)
        x = self.dil_convs(x)
        x = self.final_conv(x)
        x = x.broadcast_like(input)
        return x


class BamBlock(HybridBlock):
    """
    BAM attention block for BAM-ResNet.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 channels,
                 bn_use_global_stats,
                 **kwargs):
        super(BamBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.ch_att = ChannelGate(
                channels=channels,
                bn_use_global_stats=bn_use_global_stats)
            self.sp_att = SpatialGate(
                channels=channels,
                bn_use_global_stats=bn_use_global_stats)
            self.sigmoid = nn.Activation("sigmoid")

    def hybrid_forward(self, F, x):
        att = 1 + self.sigmoid(self.ch_att(x) * self.sp_att(x))
        x = x * att
        return x


class BamResUnit(HybridBlock):
    """
    BAM-ResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 bottleneck,
                 **kwargs):
        super(BamResUnit, self).__init__(**kwargs)
        self.use_bam = (strides != 1)

        with self.name_scope():
            if self.use_bam:
                self.bam = BamBlock(
                    channels=in_channels,
                    bn_use_global_stats=bn_use_global_stats)
            self.res_unit = ResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                bottleneck=bottleneck,
                conv1_stride=False)

    def hybrid_forward(self, F, x):
        if self.use_bam:
            x = self.bam(x)
        x = self.res_unit(x)
        return x


class BamResNet(HybridBlock):
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
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(BamResNet, self).__init__(**kwargs)
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
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(BamResUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=bottleneck))
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


def get_resnet(blocks,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def bam_resnet18(**kwargs):
    """
    BAM-ResNet-18 model from 'BAM: Bottleneck Attention Module,' https://arxiv.org/abs/1807.06514.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="bam_resnet152", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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
        assert (model != bam_resnet18 or weight_count == 11712503)
        assert (model != bam_resnet34 or weight_count == 21820663)
        assert (model != bam_resnet50 or weight_count == 25915099)
        assert (model != bam_resnet101 or weight_count == 44907227)
        assert (model != bam_resnet152 or weight_count == 60550875)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
