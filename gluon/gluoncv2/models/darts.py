"""
    DARTS for ImageNet-1K, implemented in Gluon.
    Original paper: 'DARTS: Differentiable Architecture Search,' https://arxiv.org/abs/1806.09055.
"""

__all__ = ['DARTS', 'darts']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from mxnet.gluon.contrib.nn import Identity
from .common import conv1x1
from .nasnet import nasnet_dual_path_sequential


class DwsConv(HybridBlock):
    """
    Standard dilated depthwise separable convolution block with.

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
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    use_bias : bool, default False
        Whether the layers use a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation,
                 use_bias=False,
                 **kwargs):
        super(DwsConv, self).__init__(**kwargs)
        with self.name_scope():
            self.dw_conv = nn.Conv2D(
                channels=in_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                use_bias=use_bias,
                in_channels=in_channels)
            self.pw_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias)

    def hybrid_forward(self, F, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DartsConv(HybridBlock):
    """
    DARTS specific convolution block.

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
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 activate=True,
                 **kwargs):
        super(DartsConv, self).__init__(**kwargs)
        self.activate = activate

        with self.name_scope():
            if self.activate:
                self.activ = nn.Activation("relu")
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(in_channels=out_channels)

    def hybrid_forward(self, F, x):
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
        kernel_size=1,
        strides=1,
        padding=0,
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
        kernel_size=3,
        strides=2,
        padding=1,
        activate=activate)


class DartsDwsConv(HybridBlock):
    """
    DARTS specific dilated convolution block.

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
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation,
                 **kwargs):
        super(DartsDwsConv, self).__init__(**kwargs)
        with self.name_scope():
            self.activ = nn.Activation("relu")
            self.conv = DwsConv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                use_bias=False)
            self.bn = nn.BatchNorm(in_channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DartsDwsBranch(HybridBlock):
    """
    DARTS specific block with depthwise separable convolution layers.

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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 **kwargs):
        super(DartsDwsBranch, self).__init__(**kwargs)
        mid_channels = in_channels

        with self.name_scope():
            self.conv1 = DartsDwsConv(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=1)
            self.conv2 = DartsDwsConv(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                dilation=1)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DartsReduceBranch(HybridBlock):
    """
    DARTS specific factorized reduce block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides=2,
                 **kwargs):
        super(DartsReduceBranch, self).__init__(**kwargs)
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.activ = nn.Activation("relu")
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=strides)
            self.conv2 = conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=strides)
            self.bn = nn.BatchNorm(in_channels=out_channels)

    def hybrid_forward(self, F, x):
        x = self.activ(x)
        x1 = self.conv1(x)
        x = F.slice(x, begin=(None, None, 1, 1), end=(None, None, None, None))
        x2 = self.conv2(x)
        x = F.concat(x1, x2, dim=1)
        x = self.bn(x)
        return x


class Stem1Unit(HybridBlock):
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
                 out_channels,
                 **kwargs):
        super(Stem1Unit, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        with self.name_scope():
            self.conv1 = darts_conv3x3_s2(
                in_channels=in_channels,
                out_channels=mid_channels,
                activate=False)
            self.conv2 = darts_conv3x3_s2(
                in_channels=mid_channels,
                out_channels=out_channels,
                activate=True)

    def hybrid_forward(self, F, x):
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
                     strides):
    """
    DARTS specific 3x3 Max pooling layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels. Unused parameter.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    """
    assert (channels > 0)
    return nn.MaxPool2D(
        pool_size=3,
        strides=strides,
        padding=1)


def darts_skip_connection(channels,
                          strides):
    """
    DARTS specific skip connection layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    """
    assert (channels > 0)
    if strides == 1:
        return Identity()
    else:
        assert (strides == 2)
        return DartsReduceBranch(
            in_channels=channels,
            out_channels=channels,
            strides=strides)


def darts_dws_conv3x3(channels,
                      strides):
    """
    3x3 version of DARTS specific dilated convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return DartsDwsConv(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        strides=strides,
        padding=2,
        dilation=2)


def darts_dws_branch3x3(channels,
                        strides):
    """
    3x3 version of DARTS specific dilated convolution branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return DartsDwsBranch(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        strides=strides,
        padding=1)


# Set of operations in genotype.
GENOTYPE_OPS = {
    'max_pool_3x3': darts_maxpool3x3,
    'skip_connect': darts_skip_connection,
    'dil_conv_3x3': darts_dws_conv3x3,
    'sep_conv_3x3': darts_dws_branch3x3,
}


class DartsMainBlock(HybridBlock):
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
                 reduction,
                 **kwargs):
        super(DartsMainBlock, self).__init__(**kwargs)
        self.concat = [2, 3, 4, 5]
        op_names, indices = zip(*genotype)
        self.indices = indices
        self.steps = len(op_names) // 2

        with self.name_scope():
            for i, (name, index) in enumerate(zip(op_names, indices)):
                stride = 2 if reduction and index < 2 else 1
                setattr(self, "ops{}".format(i + 1), GENOTYPE_OPS[name](channels, stride))

    def hybrid_forward(self, F, x, x_prev):
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
        x_out = F.concat(*[states[i] for i in self.concat], dim=1)
        return x_out


class DartsUnit(HybridBlock):
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
                 prev_reduction,
                 **kwargs):
        super(DartsUnit, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
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

    def hybrid_forward(self, F, x, x_prev):
        x = self.preprocess(x)
        x_prev = self.preprocess_prev(x_prev)
        x_out = self.body(x, x_prev)
        return x_out


class DARTS(HybridBlock):
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
                 classes=1000,
                 **kwargs):
        super(DARTS, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nasnet_dual_path_sequential(
                return_two=False,
                first_ordinals=2,
                last_ordinals=1)
            self.features.add(Stem1Unit(
                in_channels=in_channels,
                out_channels=stem_blocks_channels))
            in_channels = stem_blocks_channels
            self.features.add(stem2_unit(
                in_channels=in_channels,
                out_channels=stem_blocks_channels))
            prev_in_channels = in_channels
            in_channels = stem_blocks_channels

            for i, channels_per_stage in enumerate(channels):
                stage = nasnet_dual_path_sequential(prefix="stage{}_".format(i + 1))
                for j, out_channels in enumerate(channels_per_stage):
                    reduction = (i != 0) and (j == 0)
                    prev_reduction = ((i == 0) and (j == 0)) or ((i != 0) and (j == 1))
                    genotype = reduce_genotype if reduction else normal_genotype
                    stage.add(DartsUnit(
                        in_channels=in_channels,
                        prev_in_channels=prev_in_channels,
                        out_channels=out_channels,
                        genotype=genotype,
                        reduction=reduction,
                        prev_reduction=prev_reduction))
                    prev_in_channels = in_channels
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


def get_darts(model_name=None,
              pretrained=False,
              ctx=cpu(),
              root=os.path.join("~", ".mxnet", "models"),
              **kwargs):
    """
    Create DARTS model with specific parameters.

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
        net.load_parameters(
            filename=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            ctx=ctx)

    return net


def darts(**kwargs):
    """
    DARTS model from 'DARTS: Differentiable Architecture Search,' https://arxiv.org/abs/1806.09055.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_darts(model_name="darts", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        darts,
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
        assert (model != darts or weight_count == 4718752)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
