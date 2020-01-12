"""
    SelecSLS for ImageNet-1K, implemented in Gluon.
    Original paper: 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.
"""

__all__ = ['SelecSLS', 'selecsls42', 'selecsls42b', 'selecsls60', 'selecsls60b', 'selecsls84']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1_block, conv3x3_block, DualPathSequential


class SelecSLSBlock(HybridBlock):
    """
    SelecSLS block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bn_use_global_stats,
                 **kwargs):
        super(SelecSLSBlock, self).__init__(**kwargs)
        mid_channels = 2 * out_channels

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SelecSLSUnit(HybridBlock):
    """
    SelecSLS unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    skip_channels : int
        Number of skipped channels.
    mid_channels : int
        Number of middle channels.
    strides : int or tuple/list of 2 int
        Strides of the branch convolution layers.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 skip_channels,
                 mid_channels,
                 strides,
                 bn_use_global_stats,
                 **kwargs):
        super(SelecSLSUnit, self).__init__(**kwargs)
        self.resize = (strides == 2)
        mid2_channels = mid_channels // 2
        last_channels = 2 * mid_channels + (skip_channels if strides == 1 else 0)

        with self.name_scope():
            self.branch1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            self.branch2 = SelecSLSBlock(
                in_channels=mid_channels,
                out_channels=mid2_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.branch3 = SelecSLSBlock(
                in_channels=mid2_channels,
                out_channels=mid2_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.last_conv = conv1x1_block(
                in_channels=last_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x, x0=None):
        x1 = self.branch1(x)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        if self.resize:
            y = F.concat(x1, x2, x3, dim=1)
            y = self.last_conv(y)
            return y, y
        else:
            y = F.concat(x1, x2, x3, x0, dim=1)
            y = self.last_conv(y)
            return y, x0


class SelecSLS(HybridBlock):
    """
    SelecSLS model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    skip_channels : list of list of int
        Number of skipped channels for each unit.
    mid_channels : list of list of int
        Number of middle channels for each unit.
    kernels3 : list of list of int/bool
        Using 3x3 (instead of 1x1) kernel for each head unit.
    bn_use_global_stats : bool, default False
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 skip_channels,
                 mid_channels,
                 kernels3,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(SelecSLS, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        init_block_channels = 32

        with self.name_scope():
            self.features = DualPathSequential(
                return_two=False,
                first_ordinals=1,
                last_ordinals=(1 + len(kernels3)),
                prefix="")
            self.features.add(conv3x3_block(
                in_channels=in_channels,
                out_channels=init_block_channels,
                strides=2,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                k = i - len(skip_channels)
                stage = DualPathSequential(prefix="stage{}_".format(i + 1)) if k < 0 else\
                    nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if j == 0 else 1
                        if k < 0:
                            unit = SelecSLSUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                skip_channels=skip_channels[i][j],
                                mid_channels=mid_channels[i][j],
                                strides=strides,
                                bn_use_global_stats=bn_use_global_stats)
                        else:
                            conv_block_class = conv3x3_block if kernels3[k][j] == 1 else conv1x1_block
                            unit = conv_block_class(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                strides=strides,
                                bn_use_global_stats=bn_use_global_stats)
                        stage.add(unit)
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=4,
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


def get_selecsls(version,
                 model_name=None,
                 pretrained=False,
                 ctx=cpu(),
                 root=os.path.join("~", ".mxnet", "models"),
                 **kwargs):
    """
    Create SelecSLS model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SelecSLS.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if version in ("42", "42b"):
        channels = [[64, 128], [144, 288], [304, 480]]
        skip_channels = [[0, 64], [0, 144], [0, 304]]
        mid_channels = [[64, 64], [144, 144], [304, 304]]
        kernels3 = [[1, 1], [1, 0]]
        if version == "42":
            head_channels = [[960, 1024], [1024, 1280]]
        else:
            head_channels = [[960, 1024], [1280, 1024]]
    elif version in ("60", "60b"):
        channels = [[64, 128], [128, 128, 288], [288, 288, 288, 416]]
        skip_channels = [[0, 64], [0, 128, 128], [0, 288, 288, 288]]
        mid_channels = [[64, 64], [128, 128, 128], [288, 288, 288, 288]]
        kernels3 = [[1, 1], [1, 0]]
        if version == "60":
            head_channels = [[756, 1024], [1024, 1280]]
        else:
            head_channels = [[756, 1024], [1280, 1024]]
    elif version == "84":
        channels = [[64, 144], [144, 144, 144, 144, 304], [304, 304, 304, 304, 304, 512]]
        skip_channels = [[0, 64], [0, 144, 144, 144, 144], [0, 304, 304, 304, 304, 304]]
        mid_channels = [[64, 64], [144, 144, 144, 144, 144], [304, 304, 304, 304, 304, 304]]
        kernels3 = [[1, 1], [1, 1]]
        head_channels = [[960, 1024], [1024, 1280]]
    else:
        raise ValueError("Unsupported SelecSLS version {}".format(version))

    channels += head_channels

    net = SelecSLS(
        channels=channels,
        skip_channels=skip_channels,
        mid_channels=mid_channels,
        kernels3=kernels3,
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


def selecsls42(**kwargs):
    """
    SelecSLS-42 model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="42", model_name="selecsls42", **kwargs)


def selecsls42b(**kwargs):
    """
    SelecSLS-42b model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="42b", model_name="selecsls42b", **kwargs)


def selecsls60(**kwargs):
    """
    SelecSLS-60 model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="60", model_name="selecsls60", **kwargs)


def selecsls60b(**kwargs):
    """
    SelecSLS-60b model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="60b", model_name="selecsls60b", **kwargs)


def selecsls84(**kwargs):
    """
    SelecSLS-84 model from 'XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera,'
    https://arxiv.org/abs/1907.00837.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_selecsls(version="84", model_name="selecsls84", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        selecsls42,
        selecsls42b,
        selecsls60,
        selecsls60b,
        selecsls84,
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
        assert (model != selecsls42 or weight_count == 30354952)
        assert (model != selecsls42b or weight_count == 32458248)
        assert (model != selecsls60 or weight_count == 30670768)
        assert (model != selecsls60b or weight_count == 32774064)
        assert (model != selecsls84 or weight_count == 50954600)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
