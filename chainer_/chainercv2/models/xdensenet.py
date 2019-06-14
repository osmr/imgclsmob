"""
    X-DenseNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.
"""

__all__ = ['XDenseNet', 'xdensenet121_2', 'xdensenet161_2', 'xdensenet169_2', 'xdensenet201_2', 'pre_xconv3x3_block',
           'XDenseUnit']

import os
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential
from .preresnet import PreResInitBlock, PreResActivation
from .densenet import TransitionBlock


class XMaskInit(chainer.initializer.Initializer):
    """
    Returns an initializer performing "X-Net" initialization for masks.

    Parameters:
    ----------
    expand_ratio : int
        Ratio of expansion.
    """
    def __init__(self,
                 expand_ratio,
                 **kwargs):
        super(XMaskInit, self).__init__(**kwargs)
        assert (expand_ratio > 0)
        self.expand_ratio = expand_ratio

    def __call__(self, array):
        if self.dtype is not None:
            assert array.dtype == self.dtype
        xp = chainer.backend.get_array_module(array)
        shape = array.shape
        expand_size = max(shape[1] // self.expand_ratio, 1)
        array[:] = 0
        for i in range(shape[0]):
            jj = xp.random.permutation(shape[1])[:expand_size]
            array[i, jj, :, :] = 1


class XConvolution2D(L.Convolution2D):
    """
    X-Convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int or tuple/list of 2 int
        Convolution window size.
    groups : int, default 1
        Number of groups.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 groups=1,
                 expand_ratio=2,
                 **kwargs):
        super(XConvolution2D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            ksize=ksize,
            groups=groups,
            **kwargs)
        if isinstance(ksize, int):
            ksize = (ksize, ksize)
        grouped_in_channels = in_channels // groups

        self.mask = chainer.initializers.generate_array(
            initializer=XMaskInit(expand_ratio=expand_ratio),
            shape=(out_channels, grouped_in_channels, ksize[0], ksize[1]),
            xp=self.xp)
        self.register_persistent('mask')

    def forward(self, x):
        if self.W.array is None:
            self._initialize_params(x.shape[1])
        masked_weight = self.W * self.mask
        # print("self.W.sum()={}".format(self.W.array.sum()))
        # print("self.mask.sum()={}".format(self.mask.sum()))
        # print("masked_weight.sum()={}".format(masked_weight.array.sum()))
        return F.convolution_2d(
            x=x,
            W=masked_weight,
            b=self.b,
            stride=self.stride,
            pad=self.pad,
            dilate=self.dilate,
            groups=self.groups)


class PreXConvBlock(Chain):
    """
    X-Convolution block with Batch normalization and ReLU pre-activation.

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
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 return_preact=False,
                 activate=True,
                 expand_ratio=2):
        super(PreXConvBlock, self).__init__()
        self.return_preact = return_preact
        self.activate = activate

        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5)
            if self.activate:
                self.activ = F.relu
            self.conv = XConvolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=(not use_bias),
                dilate=dilate,
                expand_ratio=expand_ratio)

    def __call__(self, x):
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_xconv1x1_block(in_channels,
                       out_channels,
                       stride=1,
                       use_bias=False,
                       return_preact=False,
                       activate=True,
                       expand_ratio=2):
    """
    1x1 version of the pre-activated x-convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    return PreXConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=0,
        use_bias=use_bias,
        return_preact=return_preact,
        activate=activate,
        expand_ratio=expand_ratio)


def pre_xconv3x3_block(in_channels,
                       out_channels,
                       stride=1,
                       pad=1,
                       dilate=1,
                       return_preact=False,
                       activate=True,
                       expand_ratio=2):
    """
    3x3 version of the pre-activated x-convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    expand_ratio : int, default 2
        Ratio of expansion.
    """
    return PreXConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        return_preact=return_preact,
        activate=activate,
        expand_ratio=expand_ratio)


class XDenseUnit(Chain):
    """
    X-DenseNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    expand_ratio : int
        Ratio of expansion.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 dropout_rate,
                 expand_ratio):
        super(XDenseUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        bn_size = 4
        inc_channels = out_channels - in_channels
        mid_channels = inc_channels * bn_size

        with self.init_scope():
            self.conv1 = pre_xconv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                expand_ratio=expand_ratio)
            self.conv2 = pre_xconv3x3_block(
                in_channels=mid_channels,
                out_channels=inc_channels,
                expand_ratio=expand_ratio)
            if self.use_dropout:
                self.dropout = partial(
                    F.dropout,
                    ratio=dropout_rate)

    def __call__(self, x):
        identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = F.concat((identity, x), axis=1)
        return x


class XDenseNet(Chain):
    """
    X-DenseNet model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
    expand_ratio : int, default 2
        Ratio of expansion.
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
                 dropout_rate=0.0,
                 expand_ratio=2,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(XDenseNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", PreResInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        if i != 0:
                            setattr(stage, "trans{}".format(i + 1), TransitionBlock(
                                in_channels=in_channels,
                                out_channels=(in_channels // 2)))
                            in_channels = in_channels // 2
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), XDenseUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                dropout_rate=dropout_rate,
                                expand_ratio=expand_ratio))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "post_activ", PreResActivation(in_channels=in_channels))
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


def get_xdensenet(blocks,
                  expand_ratio=2,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create X-DenseNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    expand_ratio : int, default 2
        Ratio of expansion.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    if blocks == 121:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 24, 16]
    elif blocks == 161:
        init_block_channels = 96
        growth_rate = 48
        layers = [6, 12, 36, 24]
    elif blocks == 169:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 32, 32]
    elif blocks == 201:
        init_block_channels = 64
        growth_rate = 32
        layers = [6, 12, 48, 32]
    else:
        raise ValueError("Unsupported X-DenseNet version with number of layers {}".format(blocks))

    from functools import reduce
    channels = reduce(
        lambda xi, yi: xi + [reduce(
            lambda xj, yj: xj + [xj[-1] + yj],
            [growth_rate] * yi,
            [xi[-1][-1] // 2])[1:]],
        layers,
        [[init_block_channels * 2]])[1:]

    net = XDenseNet(
        channels=channels,
        init_block_channels=init_block_channels,
        expand_ratio=expand_ratio,
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


def xdensenet121_2(**kwargs):
    """
    X-DenseNet-121-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=121, model_name="xdensenet121_2", **kwargs)


def xdensenet161_2(**kwargs):
    """
    X-DenseNet-161-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=161, model_name="xdensenet161_2", **kwargs)


def xdensenet169_2(**kwargs):
    """
    X-DenseNet-169-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=169, model_name="xdensenet169_2", **kwargs)


def xdensenet201_2(**kwargs):
    """
    X-DenseNet-201-2 model from 'Deep Expander Networks: Efficient Deep Networks from Graph Theory,'
    https://arxiv.org/abs/1711.08757.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_xdensenet(blocks=201, model_name="xdensenet201_2", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        xdensenet121_2,
        xdensenet161_2,
        xdensenet169_2,
        xdensenet201_2,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != xdensenet121_2 or weight_count == 7978856)
        assert (model != xdensenet161_2 or weight_count == 28681000)
        assert (model != xdensenet169_2 or weight_count == 14149480)
        assert (model != xdensenet201_2 or weight_count == 20013928)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
