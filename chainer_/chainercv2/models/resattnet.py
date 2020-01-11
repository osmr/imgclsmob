"""
    ResAttNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.
"""

__all__ = ['ResAttNet', 'resattnet56', 'resattnet92', 'resattnet128', 'resattnet164', 'resattnet200', 'resattnet236',
           'resattnet452']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, conv7x7_block, pre_conv1x1_block, pre_conv3x3_block, Hourglass, SimpleSequential


class PreResBottleneck(Chain):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(PreResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        with self.init_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                return_preact=True)
            self.conv2 = pre_conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride)
            self.conv3 = pre_conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x, x_pre_activ = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x, x_pre_activ


class ResBlock(Chain):
    """
    Residual block with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(ResBlock, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        with self.init_scope():
            self.body = PreResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)
            if self.resize_identity:
                self.identity_conv = conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride)

    def __call__(self, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


class InterpolationBlock(Chain):
    """
    Interpolation block.

    Parameters:
    ----------
    size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 size):
        super(InterpolationBlock, self).__init__()
        self.size = size

    def __call__(self, x):
        return F.resize_images(x, output_shape=self.size)


class DoubleSkipBlock(Chain):
    """
    Double skip connection block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DoubleSkipBlock, self).__init__()
        with self.init_scope():
            self.skip1 = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = x + self.skip1(x)
        return x


class ResBlockSequence(Chain):
    """
    Sequence of residual blocks with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of sequence.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length):
        super(ResBlockSequence, self).__init__()
        with self.init_scope():
            self.blocks = SimpleSequential()
            with self.blocks.init_scope():
                for i in range(length):
                    setattr(self.blocks, "block{}".format(i + 1), ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels))

    def __call__(self, x):
        x = self.blocks(x)
        return x


class DownAttBlock(Chain):
    """
    Down sub-block for hourglass of attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of residual blocks list.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length):
        super(DownAttBlock, self).__init__()
        with self.init_scope():
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)
            self.res_blocks = ResBlockSequence(
                in_channels=in_channels,
                out_channels=out_channels,
                length=length)

    def __call__(self, x):
        x = self.pool(x)
        x = self.res_blocks(x)
        return x


class UpAttBlock(Chain):
    """
    Up sub-block for hourglass of attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    length : int
        Length of residual blocks list.
    size : tuple of 2 int
        Spatial size of the output tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 length,
                 size):
        super(UpAttBlock, self).__init__()
        with self.init_scope():
            self.res_blocks = ResBlockSequence(
                in_channels=in_channels,
                out_channels=out_channels,
                length=length)
            self.upsample = InterpolationBlock(size)

    def __call__(self, x):
        x = self.res_blocks(x)
        x = self.upsample(x)
        return x


class MiddleAttBlock(Chain):
    """
    Middle sub-block for attention block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    """
    def __init__(self,
                 channels):
        super(MiddleAttBlock, self).__init__()
        with self.init_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=channels,
                out_channels=channels)
            self.conv2 = pre_conv1x1_block(
                in_channels=channels,
                out_channels=channels)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.sigmoid(x)
        return x


class AttBlock(Chain):
    """
    Attention block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    hourglass_depth : int
        Depth of hourglass block.
    att_scales : list of int
        Attention block specific scales.
    in_size : tuple of 2 int
        Spatial size of the input tensor for the bilinear upsampling operation.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hourglass_depth,
                 att_scales,
                 in_size):
        super(AttBlock, self).__init__()
        assert (len(att_scales) == 3)
        scale_factor = 2
        scale_p, scale_t, scale_r = att_scales

        with self.init_scope():
            self.init_blocks = ResBlockSequence(
                in_channels=in_channels,
                out_channels=out_channels,
                length=scale_p)

            down_seq = SimpleSequential()
            up_seq = SimpleSequential()
            skip_seq = SimpleSequential()
            for i in range(hourglass_depth):
                with down_seq.init_scope():
                    setattr(down_seq, "down{}".format(i + 1), DownAttBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        length=scale_r))
                with up_seq.init_scope():
                    setattr(up_seq, "up{}".format(i + 1), UpAttBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        length=scale_r,
                        size=in_size))
                in_size = tuple([x // scale_factor for x in in_size])
                with skip_seq.init_scope():
                    if i == 0:
                        setattr(skip_seq, "skip1", ResBlockSequence(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            length=scale_t))
                    else:
                        setattr(skip_seq, "skip{}".format(i + 1), DoubleSkipBlock(
                            in_channels=in_channels,
                            out_channels=out_channels))
            self.hg = Hourglass(
                down_seq=down_seq,
                up_seq=up_seq,
                skip_seq=skip_seq,
                return_first_skip=True)

            self.middle_block = MiddleAttBlock(channels=out_channels)
            self.final_block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels)

    def __call__(self, x):
        x = self.init_blocks(x)
        x, y = self.hg(x)
        x = self.middle_block(x)
        x = (1 + x) * y
        x = self.final_block(x)
        return x


class ResAttInitBlock(Chain):
    """
    ResAttNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResAttInitBlock, self).__init__()
        with self.init_scope():
            self.conv = conv7x7_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2)
            self.pool = partial(
                F.max_pooling_2d,
                ksize=3,
                stride=2,
                pad=1,
                cover_all=False)

    def __call__(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PreActivation(Chain):
    """
    Pre-activation block without convolution layer. It's used by itself as the final block in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PreActivation, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5)
            self.activ = F.relu

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ResAttNet(Chain):
    """
    ResAttNet model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    attentions : list of list of int
        Whether to use a attention unit or residual one.
    att_scales : list of int
        Attention block specific scales.
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
                 attentions,
                 att_scales,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(ResAttNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", ResAttInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                in_size = tuple([x // 4 for x in in_size])
                for i, channels_per_stage in enumerate(channels):
                    hourglass_depth = len(channels) - 1 - i
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            if attentions[i][j]:
                                setattr(stage, "unit{}".format(j + 1), AttBlock(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    hourglass_depth=hourglass_depth,
                                    att_scales=att_scales,
                                    in_size=in_size))
                            else:
                                setattr(stage, "unit{}".format(j + 1), ResBlock(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    stride=stride))
                            in_channels = out_channels
                            in_size = tuple([x // stride for x in in_size])
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, 'post_activ', PreActivation(
                    in_channels=in_channels))
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


def get_resattnet(blocks,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create ResAttNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    if blocks == 56:
        att_layers = [1, 1, 1]
        att_scales = [1, 2, 1]
    elif blocks == 92:
        att_layers = [1, 2, 3]
        att_scales = [1, 2, 1]
    elif blocks == 128:
        att_layers = [2, 3, 4]
        att_scales = [1, 2, 1]
    elif blocks == 164:
        att_layers = [3, 4, 5]
        att_scales = [1, 2, 1]
    elif blocks == 200:
        att_layers = [4, 5, 6]
        att_scales = [1, 2, 1]
    elif blocks == 236:
        att_layers = [5, 6, 7]
        att_scales = [1, 2, 1]
    elif blocks == 452:
        att_layers = [5, 6, 7]
        att_scales = [2, 4, 3]
    else:
        raise ValueError("Unsupported ResAttNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    layers = att_layers + [2]
    channels = [[ci] * (li + 1) for (ci, li) in zip(channels_per_layers, layers)]
    attentions = [[0] + [1] * li for li in att_layers] + [[0] * 3]

    net = ResAttNet(
        channels=channels,
        init_block_channels=init_block_channels,
        attentions=attentions,
        att_scales=att_scales,
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


def resattnet56(**kwargs):
    """
    ResAttNet-56 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=56, model_name="resattnet56", **kwargs)


def resattnet92(**kwargs):
    """
    ResAttNet-92 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=92, model_name="resattnet92", **kwargs)


def resattnet128(**kwargs):
    """
    ResAttNet-128 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=128, model_name="resattnet128", **kwargs)


def resattnet164(**kwargs):
    """
    ResAttNet-164 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=164, model_name="resattnet164", **kwargs)


def resattnet200(**kwargs):
    """
    ResAttNet-200 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=200, model_name="resattnet200", **kwargs)


def resattnet236(**kwargs):
    """
    ResAttNet-236 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=236, model_name="resattnet236", **kwargs)


def resattnet452(**kwargs):
    """
    ResAttNet-452 model from 'Residual Attention Network for Image Classification,' https://arxiv.org/abs/1704.06904.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_resattnet(blocks=452, model_name="resattnet452", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        resattnet56,
        resattnet92,
        resattnet128,
        resattnet164,
        resattnet200,
        resattnet236,
        resattnet452,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resattnet56 or weight_count == 31810728)
        assert (model != resattnet92 or weight_count == 52466344)
        assert (model != resattnet128 or weight_count == 65294504)
        assert (model != resattnet164 or weight_count == 78122664)
        assert (model != resattnet200 or weight_count == 90950824)
        assert (model != resattnet236 or weight_count == 103778984)
        assert (model != resattnet452 or weight_count == 182285224)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
