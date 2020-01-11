"""
    DRN for ImageNet-1K, implemented in Chainer.
    Original paper: 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.
"""

__all__ = ['DRN', 'drnc26', 'drnc42', 'drnc58', 'drnd22', 'drnd38', 'drnd54', 'drnd105']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import SimpleSequential


class DRNConv(Chain):
    """
    DRN specific convolution block.

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
    activate : bool
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 pad,
                 dilate,
                 activate):
        super(DRNConv, self).__init__()
        self.activate = activate

        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels=in_channels,
                out_channels=out_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                nobias=True,
                dilate=dilate)
            self.bn = L.BatchNormalization(
                size=out_channels,
                eps=1e-5)
            if self.activate:
                self.activ = F.relu

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def drn_conv1x1(in_channels,
                out_channels,
                stride,
                activate):
    """
    1x1 version of the DRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    activate : bool
        Whether activate the convolution block.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=1,
        stride=stride,
        pad=0,
        dilate=1,
        activate=activate)


def drn_conv3x3(in_channels,
                out_channels,
                stride,
                dilate,
                activate):
    """
    3x3 version of the DRN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    dilate : int or tuple/list of 2 int
        Padding/dilation value for convolution layer.
    activate : bool
        Whether activate the convolution block.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=3,
        stride=stride,
        pad=dilate,
        dilate=dilate,
        activate=activate)


class DRNBlock(Chain):
    """
    Simple DRN block for residual path in DRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    dilate : int or tuple/list of 2 int
        Padding/dilation value for convolution layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilate):
        super(DRNBlock, self).__init__()
        with self.init_scope():
            self.conv1 = drn_conv3x3(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilate=dilate,
                activate=True)
            self.conv2 = drn_conv3x3(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                dilate=dilate,
                activate=False)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DRNBottleneck(Chain):
    """
    DRN bottleneck block for residual path in DRN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    dilate : int or tuple/list of 2 int
        Padding/dilation value for 3x3 convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dilate):
        super(DRNBottleneck, self).__init__()
        mid_channels = out_channels // 4

        with self.init_scope():
            self.conv1 = drn_conv1x1(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=1,
                activate=True)
            self.conv2 = drn_conv3x3(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=stride,
                dilate=dilate,
                activate=True)
            self.conv3 = drn_conv1x1(
                in_channels=mid_channels,
                out_channels=out_channels,
                stride=1,
                activate=False)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class DRNUnit(Chain):
    """
    DRN unit with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    dilate : int or tuple/list of 2 int
        Padding/dilation value for 3x3 convolution layers.
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
                 stride,
                 dilate,
                 bottleneck,
                 simplified,
                 residual):
        super(DRNUnit, self).__init__()
        assert residual or (not bottleneck)
        assert (not (bottleneck and simplified))
        assert (not (residual and simplified))
        self.residual = residual
        self.resize_identity = ((in_channels != out_channels) or (stride != 1)) and self.residual and (not simplified)

        with self.init_scope():
            if bottleneck:
                self.body = DRNBottleneck(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dilate=dilate)
            elif simplified:
                self.body = drn_conv3x3(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dilate=dilate,
                    activate=False)
            else:
                self.body = DRNBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    dilate=dilate)
            if self.resize_identity:
                self.identity_conv = drn_conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    activate=False)
            self.activ = F.relu

    def __call__(self, x):
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
                   out_channels):
    """
    DRN specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return DRNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        ksize=7,
        stride=1,
        pad=3,
        dilate=1,
        activate=True)


class DRN(Chain):
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
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(DRN, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", drn_init_block(
                    in_channels=in_channels,
                    out_channels=init_block_channels))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), DRNUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                dilate=dilations[i][j],
                                bottleneck=(bottlenecks[i][j] == 1),
                                simplified=(simplifieds[i][j] == 1),
                                residual=(residuals[i][j] == 1)))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=28,
                    stride=1))

            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "final_conv", L.Convolution2D(
                    in_channels=in_channels,
                    out_channels=classes,
                    ksize=1))
                setattr(self.output, "final_flatten", partial(
                    F.reshape,
                    shape=(-1, classes)))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_drn(blocks,
            simplified=False,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def drnc26(**kwargs):
    """
    DRN-C-26 model from 'Dilated Residual Networks,' https://arxiv.org/abs/1705.09914.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_drn(blocks=105, simplified=True, model_name="drnd105", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

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
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != drnc26 or weight_count == 21126584)
        assert (model != drnc42 or weight_count == 31234744)
        assert (model != drnc58 or weight_count == 40542008)  # 41591608
        assert (model != drnd22 or weight_count == 16393752)
        assert (model != drnd38 or weight_count == 26501912)
        assert (model != drnd54 or weight_count == 35809176)
        assert (model != drnd105 or weight_count == 54801304)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
