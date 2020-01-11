"""
    WRN-1bit for CIFAR/SVHN, implemented in Chainer.
    Original paper: 'Training wide residual networks for deployment using a single bit for each weight,'
    https://arxiv.org/abs/1802.08530.
"""

__all__ = ['CIFARWRN1bit', 'wrn20_10_1bit_cifar10', 'wrn20_10_1bit_cifar100', 'wrn20_10_1bit_svhn',
           'wrn20_10_32bit_cifar10', 'wrn20_10_32bit_cifar100', 'wrn20_10_32bit_svhn']

import os
import math
import chainer
from chainer import backend
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential


class Binarize(chainer.function.Function):
    """
    Fake sign op for 1-bit weights.
    """

    def forward(self, inputs):
        x, = inputs
        xp = backend.get_array_module(x)
        return math.sqrt(2.0 / (x.shape[1] * x.shape[2] * x.shape[3])) * xp.sign(x),

    def backward(self, inputs, grad_outputs):
        dy, = grad_outputs
        return dy,


class Convolution2D1bit(L.Convolution2D):
    """
    Standard convolution block with binarization.

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
    pad : int or tuple/list of 2 int, default 1
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad=1,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 binarized=False):
        super(Convolution2D1bit, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            ksize=ksize,
            stride=stride,
            pad=pad,
            dilate=dilate,
            groups=groups,
            nobias=(not use_bias))
        self.binarized = binarized

    def forward(self, x):
        W_1bit = Binarize()(self.W) if self.binarized else self.W
        b_1bit = Binarize()(self.b) if self.b is not None and self.binarized else self.b
        return F.convolution_2d(
            x=x,
            W=W_1bit,
            b=b_1bit,
            stride=self.stride,
            pad=self.pad,
            dilate=self.dilate,
            groups=self.groups)


def conv1x1_1bit(in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 use_bias=False,
                 binarized=False):
    """
    Convolution 1x1 layer with binarization.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    binarized : bool, default False
        Whether to use binarization.
    """
    return Convolution2D1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        groups=groups,
        use_bias=use_bias,
        binarized=binarized)


def conv3x3_1bit(in_channels,
                 out_channels,
                 stride=1,
                 pad=1,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 binarized=False):
    """
    Convolution 3x3 layer with binarization.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        pad value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    binarized : bool, default False
        Whether to use binarization.
    """
    return Convolution2D1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        groups=groups,
        use_bias=use_bias,
        binarized=binarized)


class ConvBlock1bit(Chain):
    """
    Standard convolution block with Batch normalization and ReLU activation, and binarization.

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
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 groups=1,
                 use_bias=False,
                 bn_affine=True,
                 activate=True,
                 binarized=False):
        super(ConvBlock1bit, self).__init__()
        self.activate = activate

        with self.init_scope():
            self.conv = Convolution2D1bit(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=dilate,
                groups=groups,
                use_bias=use_bias,
                binarized=binarized)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5,
                use_gamma=bn_affine,
                use_beta=bn_affine)
            if self.activate:
                self.activ = F.relu

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block_1bit(in_channels,
                       out_channels,
                       stride=1,
                       pad=0,
                       groups=1,
                       use_bias=False,
                       bn_affine=True,
                       activate=True,
                       binarized=False):
    """
    1x1 version of the standard convolution block with binarization.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 0
        pad value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    return ConvBlock1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=pad,
        groups=groups,
        use_bias=use_bias,
        bn_affine=bn_affine,
        activate=activate,
        binarized=binarized)


class PreConvBlock1bit(Chain):
    """
    Convolution block with Batch normalization and ReLU pre-activation, and binarization.

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
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate=1,
                 use_bias=False,
                 bn_affine=True,
                 return_preact=False,
                 activate=True,
                 binarized=False):
        super(PreConvBlock1bit, self).__init__()
        self.return_preact = return_preact
        self.activate = activate

        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5,
                use_gamma=bn_affine,
                use_beta=bn_affine)
            if self.activate:
                self.activ = F.relu
            self.conv = Convolution2D1bit(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                dilate=dilate,
                use_bias=use_bias,
                binarized=binarized)

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


def pre_conv3x3_block_1bit(in_channels,
                           out_channels,
                           stride=1,
                           pad=1,
                           dilate=1,
                           bn_affine=True,
                           return_preact=False,
                           activate=True,
                           binarized=False):
    """
    3x3 version of the pre-activated convolution block with binarization.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Stride of the convolution.
    pad : int or tuple/list of 2 int, default 1
        pad value for convolution layer.
    dilate : int or tuple/list of 2 int, default 1
        dilate value for convolution layer.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    return_preact : bool, default False
        Whether return pre-activation.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    return PreConvBlock1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=pad,
        dilate=dilate,
        bn_affine=bn_affine,
        return_preact=return_preact,
        activate=activate,
        binarized=binarized)


class PreResBlock1bit(Chain):
    """
    Simple PreResNet block for residual path in ResNet unit (with binarization).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 binarized=False):
        super(PreResBlock1bit, self).__init__()
        with self.init_scope():
            self.conv1 = pre_conv3x3_block_1bit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                bn_affine=False,
                return_preact=False,
                binarized=binarized)
            self.conv2 = pre_conv3x3_block_1bit(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_affine=False,
                binarized=binarized)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PreResUnit1bit(Chain):
    """
    PreResNet unit with residual connection (with binarization).

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 binarized=False):
        super(PreResUnit1bit, self).__init__()
        self.resize_identity = (stride != 1)

        with self.init_scope():
            self.body = PreResBlock1bit(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                binarized=binarized)
            if self.resize_identity:
                self.identity_pool = partial(
                    F.average_pooling_2d,
                    ksize=3,
                    stride=2,
                    pad=1)

    def __call__(self, x):
        identity = x
        x = self.body(x)
        if self.resize_identity:
            identity = self.identity_pool(identity)
            channels = identity.shape[1]
            identity = F.pad(identity, pad_width=((0, 0), (0, channels), (0, 0), (0, 0)), mode="constant", constant_values=0)
        x = x + identity
        return x


class PreResActivation(Chain):
    """
    PreResNet pure pre-activation block without convolution layer. It's used by itself as the final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    """
    def __init__(self,
                 in_channels,
                 bn_affine=True):
        super(PreResActivation, self).__init__()
        with self.init_scope():
            self.bn = L.BatchNormalization(
                size=in_channels,
                eps=1e-5,
                use_gamma=bn_affine,
                use_beta=bn_affine)
            self.activ = F.relu

    def __call__(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class CIFARWRN1bit(Chain):
    """
    WRN-1bit model for CIFAR from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    binarized : bool, default True
        Whether to use binarization.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (32, 32)
        Spatial size of the expected input image.
    classes : int, default 10
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 binarized=True,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10):
        super(CIFARWRN1bit, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3_1bit(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    binarized=binarized))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), PreResUnit1bit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                binarized=binarized))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "post_activ", PreResActivation(
                    in_channels=in_channels,
                    bn_affine=False))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "final_conv", conv1x1_block_1bit(
                    in_channels=in_channels,
                    out_channels=classes,
                    activate=False,
                    binarized=binarized))
                setattr(self.output, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=8,
                    stride=1))
                setattr(self.output, "final_flatten", partial(
                    F.reshape,
                    shape=(-1, classes)))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_wrn1bit_cifar(classes,
                      blocks,
                      width_factor,
                      binarized=True,
                      model_name=None,
                      pretrained=False,
                      root=os.path.join("~", ".chainer", "models"),
                      **kwargs):
    """
    Create WRN-1bit model for CIFAR with specific parameters.

    Parameters:
    ----------
    classes : int
        Number of classification classes.
    blocks : int
        Number of blocks.
    width_factor : int
        Wide scale factor for width of layers.
    binarized : bool, default True
        Whether to use binarization.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    assert ((blocks - 2) % 6 == 0)
    layers = [(blocks - 2) // 6] * 3
    channels_per_layers = [16, 32, 64]
    init_block_channels = 16

    channels = [[ci * width_factor] * li for (ci, li) in zip(channels_per_layers, layers)]
    init_block_channels *= width_factor

    net = CIFARWRN1bit(
        channels=channels,
        init_block_channels=init_block_channels,
        binarized=binarized,
        classes=classes,
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


def wrn20_10_1bit_cifar10(classes=10, **kwargs):
    """
    WRN-20-10-1bit model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn1bit_cifar(classes=classes, blocks=20, width_factor=10, binarized=True,
                             model_name="wrn20_10_1bit_cifar10", **kwargs)


def wrn20_10_1bit_cifar100(classes=100, **kwargs):
    """
    WRN-20-10-1bit model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn1bit_cifar(classes=classes, blocks=20, width_factor=10, binarized=True,
                             model_name="wrn20_10_1bit_cifar100", **kwargs)


def wrn20_10_1bit_svhn(classes=10, **kwargs):
    """
    WRN-20-10-1bit model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn1bit_cifar(classes=classes, blocks=20, width_factor=10, binarized=True,
                             model_name="wrn20_10_1bit_svhn", **kwargs)


def wrn20_10_32bit_cifar10(classes=10, **kwargs):
    """
    WRN-20-10-32bit model for CIFAR-10 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn1bit_cifar(classes=classes, blocks=20, width_factor=10, binarized=False,
                             model_name="wrn20_10_32bit_cifar10", **kwargs)


def wrn20_10_32bit_cifar100(classes=100, **kwargs):
    """
    WRN-20-10-32bit model for CIFAR-100 from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 100
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn1bit_cifar(classes=classes, blocks=20, width_factor=10, binarized=False,
                             model_name="wrn20_10_32bit_cifar100", **kwargs)


def wrn20_10_32bit_svhn(classes=10, **kwargs):
    """
    WRN-20-10-32bit model for SVHN from 'Wide Residual Networks,' https://arxiv.org/abs/1605.07146.

    Parameters:
    ----------
    classes : int, default 10
        Number of classification classes.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_wrn1bit_cifar(classes=classes, blocks=20, width_factor=10, binarized=False,
                             model_name="wrn20_10_32bit_svhn", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        (wrn20_10_1bit_cifar10, 10),
        (wrn20_10_1bit_cifar100, 100),
        (wrn20_10_1bit_svhn, 10),
        (wrn20_10_32bit_cifar10, 10),
        (wrn20_10_32bit_cifar100, 100),
        (wrn20_10_32bit_svhn, 10),
    ]

    for model, classes in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != wrn20_10_1bit_cifar10 or weight_count == 26737140)
        assert (model != wrn20_10_1bit_cifar100 or weight_count == 26794920)
        assert (model != wrn20_10_1bit_svhn or weight_count == 26737140)
        assert (model != wrn20_10_32bit_cifar10 or weight_count == 26737140)
        assert (model != wrn20_10_32bit_cifar100 or weight_count == 26794920)
        assert (model != wrn20_10_32bit_svhn or weight_count == 26737140)

        x = np.zeros((1, 3, 32, 32), np.float32)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
