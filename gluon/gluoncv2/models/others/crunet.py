"""
    CRU-Net, implemented in Gluon.
    Original paper: 'Sharing Residual Units Through Collective Tensor Factorization To Improve Deep Neural Networks,'
    https://www.ijcai.org/proceedings/2018/88.
"""

__all__ = ['CRUNet', 'crunet56']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from gluon.gluoncv2.models.common import conv1x1, conv3x3, pre_conv1x1_block, pre_conv3x3_block
from gluon.gluoncv2.models.resnet import ResInitBlock


class CRUConv(HybridBlock):
    """
    CRU-Net specific convolution block.

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
    groups : int, default 1
        Number of groups.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    conv : nn.Conv2D, default None
        Predefined convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 padding,
                 groups=1,
                 bn_use_global_stats=False,
                 return_preact=False,
                 conv=None,
                 **kwargs):
        super(CRUConv, self).__init__(**kwargs)
        self.return_preact = return_preact

        with self.name_scope():
            self.bn = nn.BatchNorm(
                in_channels=in_channels,
                use_global_stats=bn_use_global_stats)
            self.activ = nn.Activation("relu")
            if conv is not None:
                self.conv = conv
            else:
                self.conv = nn.Conv2D(
                    channels=out_channels,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding,
                    groups=groups,
                    use_bias=False,
                    in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.bn(x)
        x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def cru_conv1x1(in_channels,
                out_channels,
                strides=1,
                bn_use_global_stats=False,
                return_preact=False,
                conv=None):
    """
    1x1 version of the CRU-Net specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    conv : nn.Conv2D, default None
        Predefined convolution layer.
    """
    return CRUConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        conv=conv)


def cru_conv3x3(in_channels,
                out_channels,
                strides=1,
                groups=1,
                bn_use_global_stats=False,
                return_preact=False,
                conv=None):
    """
    3x3 version of the CRU-Net specific convolution block.

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
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    return_preact : bool, default False
        Whether return pre-activation.
    conv : nn.Conv2D, default None
        Predefined convolution layer.
    """
    return CRUConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        groups=groups,
        bn_use_global_stats=bn_use_global_stats,
        return_preact=return_preact,
        conv=conv)


class ResBottleneck(HybridBlock):
    """
    Pre-ResNeXt bottleneck block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 bn_use_global_stats,
                 **kwargs):
        super(ResBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D

        with self.name_scope():
            self.conv1 = pre_conv1x1_block(
                in_channels=in_channels,
                out_channels=group_width,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = pre_conv3x3_block(
                in_channels=group_width,
                out_channels=group_width,
                strides=strides,
                groups=cardinality,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = pre_conv1x1_block(
                in_channels=group_width,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class CRUBottleneck(HybridBlock):
    """
    CRU-Net bottleneck block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    conv1 : nn.Conv2D, default None
        Predefined 1x1 convolution layer.
    conv2 : nn.Conv2D, default None
        Predefined 3x3 convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 bn_use_global_stats,
                 conv1=None,
                 conv2=None,
                 **kwargs):
        super(CRUBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // 4
        D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
        group_width = cardinality * D

        with self.name_scope():
            self.conv1 = cru_conv1x1(
                in_channels=in_channels,
                out_channels=group_width,
                bn_use_global_stats=bn_use_global_stats,
                conv=conv1)
            self.conv2 = cru_conv3x3(
                in_channels=group_width,
                out_channels=group_width,
                strides=strides,
                groups=cardinality,
                bn_use_global_stats=bn_use_global_stats,
                conv=conv2)
            self.conv3 = pre_conv1x1_block(
                in_channels=group_width,
                out_channels=group_width,
                bn_use_global_stats=bn_use_global_stats)
            self.conv4 = pre_conv1x1_block(
                in_channels=group_width,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


class ResUnit(HybridBlock):
    """
    CRU-Net residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 bn_use_global_stats,
                 **kwargs):
        super(ResUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = ResBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                cardinality=cardinality,
                bottleneck_width=bottleneck_width,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = pre_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


class CRUUnit(HybridBlock):
    """
    CRU-Net collective residual unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    conv1 : nn.Conv2D, default None
        Predefined 1x1 convolution layer.
    conv2 : nn.Conv2D, default None
        Predefined 3x3 convolution layer with strides.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality,
                 bottleneck_width,
                 bn_use_global_stats,
                 conv1=None,
                 conv2=None,
                 **kwargs):
        super(CRUUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = CRUBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                cardinality=cardinality,
                bottleneck_width=bottleneck_width,
                bn_use_global_stats=bn_use_global_stats,
                conv1=conv1,
                conv2=(conv2 if strides == 1 else None))
            if self.resize_identity:
                self.identity_conv = pre_conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        x = x + identity
        return x


def predefine_convs(in_channels,
                    out_channels,
                    cardinality,
                    bottleneck_width):
    """
    Predefine CRU-Net specific shared convolution blocks.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
    """
    mid_channels = out_channels // 4
    D = int(math.floor(mid_channels * (bottleneck_width / 64.0)))
    group_width = cardinality * D
    conv1 = conv1x1(
        in_channels=in_channels,
        out_channels=group_width)
    conv2 = conv3x3(
        in_channels=group_width,
        out_channels=group_width,
        groups=cardinality)
    return conv1, conv2

class CRUNet(HybridBlock):
    """
    CRU-Net model from 'Sharing Residual Units Through Collective Tensor Factorization To Improve Deep Neural Networks,'
    https://www.ijcai.org/proceedings/2018/88.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    cardinality: int
        Number of groups.
    bottleneck_width: int
        Width of bottleneck block.
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
                 cardinality,
                 bottleneck_width,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(CRUNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
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
                        if i == 2:
                            if j == 0:
                                conv1, conv2 = None, None
                            elif j == 1:
                                conv1, conv2 = predefine_convs(
                                    in_channels=in_channels,
                                    out_channels=out_channels,
                                    cardinality=cardinality,
                                    bottleneck_width=bottleneck_width)
                            stage.add(CRUUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                cardinality=cardinality,
                                bottleneck_width=bottleneck_width,
                                bn_use_global_stats=bn_use_global_stats,
                                conv1=conv1,
                                conv2=conv2))
                        else:
                            stage.add(ResUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                cardinality=cardinality,
                                bottleneck_width=bottleneck_width,
                                bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=7,
                strides=1))

            self.output = nn.HybridSequential(prefix='')
            self.output.add(nn.Flatten())
            self.output.add(nn.Dense(
                units=classes,
                in_units=in_channels))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_crunet(blocks,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join('~', '.mxnet', 'models'),
               **kwargs):
    """
    Create CRU-Net model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    cardinality = 32
    bottleneck_width = 4

    if blocks == 56:
        layers = [3, 4, 6, 3]
    elif blocks == 116:
        layers = [3, 6, 18, 3]
    else:
        raise ValueError("Unsupported CRU-Net with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = CRUNet(
        channels=channels,
        init_block_channels=init_block_channels,
        cardinality=cardinality,
        bottleneck_width=bottleneck_width,
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


def crunet56(**kwargs):
    """
    CRU-Net-56 model from 'Sharing Residual Units Through Collective Tensor Factorization To Improve Deep Neural
    Networks,' https://www.ijcai.org/proceedings/2018/88.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_crunet(blocks=56, model_name="crunet56", **kwargs)


def crunet116(**kwargs):
    """
    CRU-Net-116 model from 'Sharing Residual Units Through Collective Tensor Factorization To Improve Deep Neural
    Networks,' https://www.ijcai.org/proceedings/2018/88.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_crunet(blocks=116, model_name="crunet116", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        crunet56,
        crunet116,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        ctx = mx.cpu()
        if not pretrained:
            net.initialize(ctx=ctx)

        # net.hybridize()
        net_params = net.collect_params()
        weight_count = 0
        for param in net_params.values():
            if (param.shape is None) or (not param._differentiable):
                continue
            weight_count += np.prod(param.shape)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != crunet56 or weight_count == 24207912)
        assert (model != crunet116 or weight_count == 34271784)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
