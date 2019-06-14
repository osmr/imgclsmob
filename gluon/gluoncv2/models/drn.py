"""
    DRN for ImageNet-1K, implemented in Gluon.
    Original paper: 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.
"""

__all__ = ['DRN', 'drnc26', 'drnc42', 'drnc58', 'drnd22', 'drnd38', 'drnd54', 'drnd105']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock


class DRNConv(HybridBlock):
    """
    DRN specific convolution block.

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
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 dilation,
                 bn_use_global_stats,
                 activate,
                 **kwargs):
        super(DRNConv, self).__init__(**kwargs)
        self.activate = activate

        with self.name_scope():
            self.conv = nn.Conv2D(
                channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                dilation=dilation,
                use_bias=False,
                in_channels=in_channels)
            self.bn = nn.BatchNorm(
                in_channels=out_channels,
                use_global_stats=bn_use_global_stats)
            if self.activate:
                self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def drn_conv1x1(in_channels,
                out_channels,
                strides,
                bn_use_global_stats,
                activate):
    """
    1x1 version of the DRN specific convolution block.

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
    activate : bool
        Whether activate the convolution block.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        dilation=1,
        bn_use_global_stats=bn_use_global_stats,
        activate=activate)


def drn_conv3x3(in_channels,
                out_channels,
                strides,
                dilation,
                bn_use_global_stats,
                activate):
    """
    3x3 version of the DRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for convolution layer.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    activate : bool
        Whether activate the convolution block.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=dilation,
        dilation=dilation,
        bn_use_global_stats=bn_use_global_stats,
        activate=activate)


class DRNBlock(HybridBlock):
    """
    Simple DRN block for residual path in DRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for convolution layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilation,
                 bn_use_global_stats,
                 **kwargs):
        super(DRNBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = drn_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=stride,
                dilation=dilation,
                bn_use_global_stats=bn_use_global_stats,
                activate=True)
            self.conv2 = drn_conv3x3(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                dilation=dilation,
                bn_use_global_stats=bn_use_global_stats,
                activate=False)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DRNBottleneck(HybridBlock):
    """
    DRN bottleneck block for residual path in DRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for 3x3 convolution layer.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 dilation,
                 bn_use_global_stats,
                 **kwargs):
        super(DRNBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4

        with self.name_scope():
            self.conv1 = drn_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=1,
                bn_use_global_stats=bn_use_global_stats,
                activate=True)
            self.conv2 = drn_conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                strides=strides,
                dilation=dilation,
                bn_use_global_stats=bn_use_global_stats,
                activate=True)
            self.conv3 = drn_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                strides=1,
                bn_use_global_stats=bn_use_global_stats,
                activate=False)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DRNUnit(HybridBlock):
    """
    DRN unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    dilation : int or tuple/list of 2 int
        Padding/dilation value for 3x3 convolution layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    simplified : bool
        Whether to use a simple or simplified block in units.
    residual : bool
        Whether do residual calculations.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 dilation,
                 bn_use_global_stats,
                 bottleneck,
                 simplified,
                 residual,
                 **kwargs):
        super(DRNUnit, self).__init__(**kwargs)
        assert residual or (not bottleneck)
        assert (not (bottleneck and simplified))
        assert (not (residual and simplified))
        self.residual = residual
        self.resize_identity = ((in_channels != out_channels) or (strides != 1)) and self.residual and (not simplified)

        with self.name_scope():
            if bottleneck:
                self.body = DRNBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    dilation=dilation,
                    bn_use_global_stats=bn_use_global_stats)
            elif simplified:
                self.body = drn_conv3x3(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    dilation=dilation,
                    bn_use_global_stats=bn_use_global_stats,
                    activate=False)
            else:
                self.body = DRNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=strides,
                    dilation=dilation,
                    bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = drn_conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activate=False)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        if self.residual:
            x = x + identity
        x = self.activ(x)
        return x


def drn_init_block(in_channels,
                   out_channels,
                   bn_use_global_stats):
    """
    DRN specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=1,
        padding=3,
        dilation=1,
        bn_use_global_stats=bn_use_global_stats,
        activate=True)


class DRN(HybridBlock):
    """
    DRN-C&D model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    dilations : list of list of int
        Dilation values for 3x3 convolution layers for each unit.
    bottlenecks : list of list of int
        Whether to use a bottleneck or simple block in each unit.
    simplifieds : list of list of int
        Whether to use a simple or simplified block in each unit.
    residuals : list of list of int
        Whether to use residual block in each unit.
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
                 dilations,
                 bottlenecks,
                 simplifieds,
                 residuals,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(DRN, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(drn_init_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != 0) else 1
                        stage.add(DRNUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            strides=strides,
                            dilation=dilations[i][j],
                            bn_use_global_stats=bn_use_global_stats,
                            bottleneck=(bottlenecks[i][j] == 1),
                            simplified=(simplifieds[i][j] == 1),
                            residual=(residuals[i][j] == 1)))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=28,
                strides=1))

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Conv2D(
                channels=classes,
                kernel_size=1,
                in_channels=in_channels))
            self.output.add(nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_drn(blocks,
            simplified=False,
            model_name=None,
            pretrained=False,
            ctx=cpu(),
            root=os.path.join("~", ".mxnet", "models"),
            **kwargs):
    """
    Create DRN-C or DRN-D model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    simplified : bool, default False
        Whether to use simplified scheme (D architecture).
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    if blocks == 22:
        assert simplified
        layers = [1, 1, 2, 2, 2, 2, 1, 1]
    elif blocks == 26:
        layers = [1, 1, 2, 2, 2, 2, 1, 1]
    elif blocks == 38:
        assert simplified
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 42:
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 54:
        assert simplified
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 58:
        layers = [1, 1, 3, 4, 6, 3, 1, 1]
    elif blocks == 105:
        assert simplified
        layers = [1, 1, 3, 4, 23, 3, 1, 1]
    else:
        raise ValueError("Unsupported DRN with number of blocks: {}".format(blocks))

    if blocks < 50:
        channels_per_layers = [16, 32, 64, 128, 256, 512, 512, 512]
        bottlenecks_per_layers = [0, 0, 0, 0, 0, 0, 0, 0]
    else:
        channels_per_layers = [16, 32, 256, 512, 1024, 2048, 512, 512]
        bottlenecks_per_layers = [0, 0, 1, 1, 1, 1, 0, 0]

    if simplified:
        simplifieds_per_layers = [1, 1, 0, 0, 0, 0, 1, 1]
        residuals_per_layers = [0, 0, 1, 1, 1, 1, 0, 0]
    else:
        simplifieds_per_layers = [0, 0, 0, 0, 0, 0, 0, 0]
        residuals_per_layers = [1, 1, 1, 1, 1, 1, 0, 0]

    dilations_per_layers = [1, 1, 1, 1, 2, 4, 2, 1]
    downsample = [0, 1, 1, 1, 0, 0, 0, 0]

    def expand(property_per_layers):
        from functools import reduce
        return reduce(
            lambda x, y: x + [[y[0]] * y[1]] if y[2] != 0 else x[:-1] + [x[-1] + [y[0]] * y[1]],
            zip(property_per_layers, layers, downsample),
            [[]])

    channels = expand(channels_per_layers)
    dilations = expand(dilations_per_layers)
    bottlenecks = expand(bottlenecks_per_layers)
    residuals = expand(residuals_per_layers)
    simplifieds = expand(simplifieds_per_layers)

    init_block_channels = channels_per_layers[0]

    net = DRN(
        channels=channels,
        init_block_channels=init_block_channels,
        dilations=dilations,
        bottlenecks=bottlenecks,
        simplifieds=simplifieds,
        residuals=residuals,
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


def drnc26(**kwargs):
    """
    DRN-C-26 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=26, model_name="drnc26", **kwargs)


def drnc42(**kwargs):
    """
    DRN-C-42 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=42, model_name="drnc42", **kwargs)


def drnc58(**kwargs):
    """
    DRN-C-58 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=58, model_name="drnc58", **kwargs)


def drnd22(**kwargs):
    """
    DRN-D-58 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=22, simplified=True, model_name="drnd22", **kwargs)


def drnd38(**kwargs):
    """
    DRN-D-38 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=38, simplified=True, model_name="drnd38", **kwargs)


def drnd54(**kwargs):
    """
    DRN-D-54 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=54, simplified=True, model_name="drnd54", **kwargs)


def drnd105(**kwargs):
    """
    DRN-D-105 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=105, simplified=True, model_name="drnd105", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        drnc26,
        drnc42,
        drnc58,
        drnd22,
        drnd38,
        drnd54,
        drnd105,
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
        assert (model != drnc26 or weight_count == 21126584)
        assert (model != drnc42 or weight_count == 31234744)
        assert (model != drnc58 or weight_count == 40542008)  # 41591608
        assert (model != drnd22 or weight_count == 16393752)
        assert (model != drnd38 or weight_count == 26501912)
        assert (model != drnd54 or weight_count == 35809176)
        assert (model != drnd105 or weight_count == 54801304)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
