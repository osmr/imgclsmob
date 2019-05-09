"""
    WRN-1bit for CIFAR/SVHN, implemented in Gluon.
    Original paper: 'Training wide residual networks for deployment using a single bit for each weight,'
    https://arxiv.org/abs/1802.08530.
"""

__all__ = ['CIFARWRN1bit', 'wrn20_10_1bit_cifar10', 'wrn20_10_1bit_cifar100', 'wrn20_10_1bit_svhn',
           'wrn20_10_32bit_cifar10', 'wrn20_10_32bit_cifar100', 'wrn20_10_32bit_svhn']

import os
import math
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class Binarize(mx.autograd.Function):
    """
    Fake sign op for 1-bit weights.
    """

    def forward(self, x):
        return math.sqrt(2.0 / (x.shape[1] * x.shape[2] * x.shape[3])) * x.sign()

    def backward(self, dy):
        return dy


class Conv2D1bit(nn.Conv2D):
    """
    Standard convolution block with binarization.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
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
                 kernel_size,
                 strides,
                 padding=1,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 binarized=False,
                 **kwargs):
        super(Conv2D1bit, self).__init__(
            in_channels=in_channels,
            channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            dilation=dilation,
            groups=groups,
            use_bias=use_bias,
            **kwargs)
        self.binarized = binarized

    def hybrid_forward(self, F, x, weight, bias=None):
        weight_1bit = Binarize()(weight) if self.binarized else weight
        bias_1bit = Binarize()(bias) if bias is not None and self.binarized else bias
        return super(Conv2D1bit, self).hybrid_forward(F, x, weight=weight_1bit, bias=bias_1bit)


def conv1x1_1bit(in_channels,
                 out_channels,
                 strides=1,
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
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    binarized : bool, default False
        Whether to use binarization.
    """
    return Conv2D1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        groups=groups,
        use_bias=use_bias,
        binarized=binarized)


def conv3x3_1bit(in_channels,
                 out_channels,
                 strides=1,
                 padding=1,
                 dilation=1,
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
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    binarized : bool, default False
        Whether to use binarization.
    """
    return Conv2D1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        groups=groups,
        use_bias=use_bias,
        binarized=binarized)


class ConvBlock1bit(HybridBlock):
    """
    Standard convolution block with Batch normalization and ReLU activation, and binarization.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 groups=1,
                 use_bias=False,
                 bn_affine=True,
                 bn_use_global_stats=False,
                 activate=True,
                 binarized=False,
                 **kwargs):
        super(ConvBlock1bit, self).__init__(**kwargs)
        self.activate = activate

        with self.name_scope():
            self.conv = Conv2D1bit(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                use_bias=use_bias,
                binarized=binarized)
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                center=bn_affine,
                scale=bn_affine,
                use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv1x1_block_1bit(in_channels,
                       out_channels,
                       strides=1,
                       padding=0,
                       groups=1,
                       use_bias=False,
                       bn_affine=True,
                       bn_use_global_stats=False,
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
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 0
        Padding value for convolution layer.
    groups : int, default 1
        Number of groups.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activate : bool, default True
        Whether activate the convolution block.
    binarized : bool, default False
        Whether to use binarization.
    """
    return ConvBlock1bit(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=padding,
        groups=groups,
        use_bias=use_bias,
        bn_affine=bn_affine,
        bn_use_global_stats=bn_use_global_stats,
        activate=activate,
        binarized=binarized)


class PreConvBlock1bit(HybridBlock):
    """
    Convolution block with Batch normalization and ReLU pre-activation, and binarization.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layer uses a bias vector.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
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
                 kernel_size,
                 strides,
                 padding,
                 dilation=1,
                 use_bias=False,
                 bn_affine=True,
                 bn_use_global_stats=False,
                 return_preact=False,
                 activate=True,
                 binarized=False,
                 **kwargs):
        super(PreConvBlock1bit, self).__init__(**kwargs)
        self.return_preact = return_preact
        self.activate = activate

        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                center=bn_affine,
                scale=bn_affine,
                use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = nn.Activation("relu")
            self.conv = Conv2D1bit(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                use_bias=use_bias,
                binarized=binarized)

    def hybrid_forward(self, F, x):
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
                           strides=1,
                           padding=1,
                           dilation=1,
                           bn_affine=True,
                           bn_use_global_stats=False,
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
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
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
        kernel_size=3,
        strides=strides,
        padding=padding,
        dilation=dilation,
        bn_affine=bn_affine,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        activate=activate,
        binarized=binarized)


class PreResBlock1bit(HybridBlock):
    """
    Simple PreResNet block for residual path in ResNet unit (with binarization).

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
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 binarized=False,
                 **kwargs):
        super(PreResBlock1bit, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = pre_conv3x3_block_1bit(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_affine=False,
                bn_use_global_stats=bn_use_global_stats,
                return_preact=False,
                binarized=binarized)
            self.conv2 = pre_conv3x3_block_1bit(
                in_channels=out_channels,
                out_channels=out_channels,
                bn_affine=False,
                bn_use_global_stats=bn_use_global_stats,
                binarized=binarized)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class PreResUnit1bit(HybridBlock):
    """
    PreResNet unit with residual connection (with binarization).

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
    binarized : bool, default False
        Whether to use binarization.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bn_use_global_stats,
                 binarized=False,
                 **kwargs):
        super(PreResUnit1bit, self).__init__(**kwargs)
        self.resize_identity = (strides != 1)

        with self.name_scope():
            self.body = PreResBlock1bit(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats,
                binarized=binarized)
            if self.resize_identity:
                self.identity_pool = nn.AvgPool2D(
                    pool_size=3,
                    strides=2,
                    padding=1)

    def hybrid_forward(self, F, x):
        identity = x
        x = self.body(x)
        if self.resize_identity:
            identity = self.identity_pool(identity)
            identity = F.concat(identity, F.zeros_like(identity), dim=1)
        x = x + identity
        return x


class PreResActivation(HybridBlock):
    """
    PreResNet pure pre-activation block without convolution layer. It's used by itself as the final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    bn_affine : bool, default True
        Whether the BatchNorm layer learns affine parameters.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 bn_affine=True,
                 bn_use_global_stats=False,
                 **kwargs):
        super(PreResActivation, self).__init__(**kwargs)
        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                center=bn_affine,
                scale=bn_affine,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class CIFARWRN1bit(HybridBlock):
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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
        Useful for fine-tuning.
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
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(32, 32),
                 classes=10,
                 **kwargs):
        super(CIFARWRN1bit, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(conv3x3_1bit(
                in_channels=in_channels,
                out_channels=init_block_channels,
                binarized=binarized))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(PreResUnit1bit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats,
                            binarized=binarized))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(PreResActivation(
                in_channels=in_channels,
                bn_use_global_stats=bn_use_global_stats,
                bn_affine=False))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(conv1x1_block_1bit(
                in_channels=in_channels,
                out_channels=classes,
                activate=False,
                binarized=binarized))
            self.output.add(nn.AvgPool2D(
                pool_size=8,
                strides=1))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_wrn1bit_cifar(classes,
                      blocks,
                      width_factor,
                      binarized=True,
                      model_name=None,
                      pretrained=False,
                      ctx=cpu(),
                      root=os.path.join("~", ".mxnet", "models"),
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
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
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_wrn1bit_cifar(classes=classes, blocks=20, width_factor=10, binarized=False,
                             model_name="wrn20_10_32bit_svhn", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

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
        assert (model != wrn20_10_1bit_cifar10 or weight_count == 26737140)
        assert (model != wrn20_10_1bit_cifar100 or weight_count == 26794920)
        assert (model != wrn20_10_1bit_svhn or weight_count == 26737140)
        assert (model != wrn20_10_32bit_cifar10 or weight_count == 26737140)
        assert (model != wrn20_10_32bit_cifar100 or weight_count == 26794920)
        assert (model != wrn20_10_32bit_svhn or weight_count == 26737140)

        x = mx.nd.zeros((1, 3, 32, 32), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, classes))


if __name__ == "__main__":
    _test()
