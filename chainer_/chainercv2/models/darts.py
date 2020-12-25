"""
    DARTS for ImageNet-1K, implemented in Chainer.
    Original paper: 'DARTS: Differentiable Architecture Search,' https://arxiv.org/abs/1806.09055.
"""

__all__ = ['DARTS', 'darts']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1, SimpleSequential
from .nasnet import nasnet_dual_path_sequential


class DwsConv(Chain):
    """
    Standard dilated depthwise separable convolution block with.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layers use a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate,
                 use_bias=False):
        super(DwsConv, self).__init__()
        with self.init_scope():
            self.dw_conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=in_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate,
                groups=in_channels)
            self.pw_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DartsConv(Chain):
    """
    DARTS specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 activate=True):
        super(DartsConv, self).__init__()
        self.activate = activate

        with self.init_scope():
            if self.activate:
                self.activ = F.relu
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)

    def forward(self, x):
        if self.activate:
            x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def darts_conv1x1(in_channels,
                  out_channels,
                  activate=True):
    """
    1x1 version of the DARTS specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return DartsConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=1,
        pad=0,
        activate=activate)


def darts_conv3x3_s2(in_channels,
                     out_channels,
                     activate=True):
    """
    3x3 version of the DARTS specific convolution block with stride 2.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return DartsConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=2,
        pad=1,
        activate=activate)


class DartsDwsConv(Chain):
    """
    DARTS specific dilated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int
        Dilation value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate):
        super(DartsDwsConv, self).__init__()
        with self.init_scope():
            self.activ = F.relu
            self.conv = DwsConv(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=dilate,
                use_bias=False)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)

    def forward(self, x):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DartsDwsBranch(Chain):
    """
    DARTS specific block with depthwise separable convolution layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    pad : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad):
        super(DartsDwsBranch, self).__init__()
        mid_channels = in_channels

        with self.init_scope():
            self.conv1 = DartsDwsConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=1)
            self.conv2 = DartsDwsConv(
                in_channels=mid_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=1,
                pad=pad,
                dilate=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DartsReduceBranch(Chain):
    """
    DARTS specific factorized reduce block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 2
        Stride of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=2):
        super(DartsReduceBranch, self).__init__()
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        with self.init_scope():
            self.activ = F.relu
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride)
            self.conv2 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=stride)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)

    def forward(self, x):
        x = self.activ(x)
        x1 = self.conv1(x)
        x = x[:, :, 1:, 1:]
        x2 = self.conv2(x)
        x = F.concat((x1, x2), axis=1)
        x = self.bn(x)
        return x


class Stem1Unit(Chain):
    """
    DARTS Stem1 unit.

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
        super(Stem1Unit, self).__init__()
        mid_channels = out_channels // 2

        with self.init_scope():
            self.conv1 = darts_conv3x3_s2(
                in_channels=in_channels,
                out_channels=mid_channels,
                activate=False)
            self.conv2 = darts_conv3x3_s2(
                in_channels=mid_channels,
                out_channels=out_channels,
                activate=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


def stem2_unit(in_channels,
               out_channels):
    """
    DARTS Stem2 unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return darts_conv3x3_s2(
        in_channels=in_channels,
        out_channels=out_channels,
        activate=True)


def darts_maxpool3x3(channels,
                     stride):
    """
    DARTS specific 3x3 Max pooling layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels. Unused parameter.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    assert (channels > 0)
    return partial(
        F.max_pooling_2d,
        ksize=3,
        stride=stride,
        pad=1,
        cover_all=False)


def darts_skip_connection(channels,
                          stride):
    """
    DARTS specific skip connection layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    assert (channels > 0)
    if stride == 1:
        return F.identity
    else:
        assert (stride == 2)
        return DartsReduceBranch(
            in_channels=channels,
            out_channels=channels,
            stride=stride)


def darts_dws_conv3x3(channels,
                      stride):
    """
    3x3 version of DARTS specific dilated convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    return DartsDwsConv(
        in_channels=channels,
        out_channels=channels,
        ksize=3,
        stride=stride,
        pad=2,
        dilate=2)


def darts_dws_branch3x3(channels,
                        stride):
    """
    3x3 version of DARTS specific dilated convolution branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    """
    return DartsDwsBranch(
        in_channels=channels,
        out_channels=channels,
        ksize=3,
        stride=stride,
        pad=1)


# Set of operations in genotype.
GENOTYPE_OPS = {
    'max_pool_3x3': darts_maxpool3x3,
    'skip_connect': darts_skip_connection,
    'dil_conv_3x3': darts_dws_conv3x3,
    'sep_conv_3x3': darts_dws_branch3x3,
}


class DartsMainBlock(Chain):
    """
    DARTS main block, described by genotype.

    Parameters:
    ----------
    genotype : list of tuples (str, int)
        List of genotype elements (operations and linked indices).
    channels : int
        Number of input/output channels.
    reduction : bool
        Whether use reduction.
    """
    def __init__(self,
                 genotype,
                 channels,
                 reduction):
        super(DartsMainBlock, self).__init__()
        self.concat = [2, 3, 4, 5]
        op_names, indices = zip(*genotype)
        self.indices = indices
        self.steps = len(op_names) // 2

        with self.init_scope():
            for i, (name, index) in enumerate(zip(op_names, indices)):
                stride = 2 if reduction and index < 2 else 1
                setattr(self, "ops{}".format(i + 1), GENOTYPE_OPS[name](channels, stride))

    def forward(self, x, x_prev):
        s0 = x_prev
        s1 = x
        states = [s0, s1]
        for i in range(self.steps):
            j1 = 2 * i
            j2 = 2 * i + 1
            op1 = getattr(self, "ops{}".format(j1 + 1))
            op2 = getattr(self, "ops{}".format(j2 + 1))
            y1 = states[self.indices[j1]]
            y2 = states[self.indices[j2]]
            y1 = op1(y1)
            y2 = op2(y2)
            s = y1 + y2
            states += [s]
        x_out = F.concat([states[i] for i in self.concat], axis=1)
        return x_out


class DartsUnit(Chain):
    """
    DARTS unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    prev_in_channels : int
        Number of input channels in previous input.
    out_channels : int
        Number of output channels.
    genotype : list of tuples (str, int)
        List of genotype elements (operations and linked indices).
    reduction : bool
        Whether use reduction.
    prev_reduction : bool
        Whether use previous reduction.
    """
    def __init__(self,
                 in_channels,
                 prev_in_channels,
                 out_channels,
                 genotype,
                 reduction,
                 prev_reduction):
        super(DartsUnit, self).__init__()
        mid_channels = out_channels // 4

        with self.init_scope():
            if prev_reduction:
                self.preprocess_prev = DartsReduceBranch(
                    in_channels=prev_in_channels,
                    out_channels=mid_channels)
            else:
                self.preprocess_prev = darts_conv1x1(
                    in_channels=prev_in_channels,
                    out_channels=mid_channels)

            self.preprocess = darts_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels)

            self.body = DartsMainBlock(
                genotype=genotype,
                channels=mid_channels,
                reduction=reduction)

    def forward(self, x, x_prev):
        x = self.preprocess(x)
        x_prev = self.preprocess_prev(x_prev)
        x_out = self.body(x, x_prev)
        return x_out


class DARTS(Chain):
    """
    DARTS model from 'DARTS: Differentiable Architecture Search,' https://arxiv.org/abs/1806.09055.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    stem_blocks_channels : int
        Number of output channels for the Stem units.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 stem_blocks_channels,
                 normal_genotype,
                 reduce_genotype,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(DARTS, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = nasnet_dual_path_sequential(
                return_two=False,
                first_ordinals=2,
                last_ordinals=1)
            with self.features.init_scope():
                setattr(self.features, "stem1_unit", Stem1Unit(
                    in_channels=in_channels,
                    out_channels=stem_blocks_channels))
                in_channels = stem_blocks_channels
                setattr(self.features, "stem2_unit", stem2_unit(
                    in_channels=in_channels,
                    out_channels=stem_blocks_channels))
                prev_in_channels = in_channels
                in_channels = stem_blocks_channels

                for i, channels_per_stage in enumerate(channels):
                    stage = nasnet_dual_path_sequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            reduction = (i != 0) and (j == 0)
                            prev_reduction = ((i == 0) and (j == 0)) or ((i != 0) and (j == 1))
                            genotype = reduce_genotype if reduction else normal_genotype
                            setattr(stage, "unit{}".format(j + 1), DartsUnit(
                                in_channels=in_channels,
                                prev_in_channels=prev_in_channels,
                                out_channels=out_channels,
                                genotype=genotype,
                                reduction=reduction,
                                prev_reduction=prev_reduction))
                            prev_in_channels = in_channels
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

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_darts(model_name=None,
              pretrained=False,
              root=os.path.join("~", ".chainer", "models"),
              **kwargs):
    """
    Create DARTS model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    stem_blocks_channels = 48
    layers = [4, 5, 5]
    channels_per_layers = [192, 384, 768]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    normal_genotype = [
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 0),
        ('sep_conv_3x3', 1),
        ('sep_conv_3x3', 1),
        ('skip_connect', 0),
        ('skip_connect', 0),
        ('dil_conv_3x3', 2)]
    reduce_genotype = [
        ('max_pool_3x3', 0),
        ('max_pool_3x3', 1),
        ('skip_connect', 2),
        ('max_pool_3x3', 1),
        ('max_pool_3x3', 0),
        ('skip_connect', 2),
        ('skip_connect', 2),
        ('max_pool_3x3', 1)]

    net = DARTS(
        channels=channels,
        stem_blocks_channels=stem_blocks_channels,
        normal_genotype=normal_genotype,
        reduce_genotype=reduce_genotype,
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


def darts(**kwargs):
    """
    DARTS model from 'DARTS: Differentiable Architecture Search,' https://arxiv.org/abs/1806.09055.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_darts(model_name="darts", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        darts,
    ]

    for model in models:
        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != darts or weight_count == 4718752)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
