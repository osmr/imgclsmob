"""
    BagNet for ImageNet-1K, implemented in Gluon.
    Original paper: 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.
"""

__all__ = ['BagNet', 'bagnet9', 'bagnet17', 'bagnet33']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv1x1, conv1x1_block, conv3x3_block, ConvBlock


class BagNetBottleneck(HybridBlock):
    """
    BagNet bottleneck block for residual path in BagNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size of the second convolution.
    strides : int or tuple/list of 2 int
        Strides of the second convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    bottleneck_factor : int, default 4
        Bottleneck factor.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 bn_use_global_stats,
                 bottleneck_factor=4,
                 **kwargs):
        super(BagNetBottleneck, self).__init__(**kwargs)
        mid_channels = out_channels // bottleneck_factor

        with self.name_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                bn_use_global_stats=bn_use_global_stats)
            self.conv2 = ConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)
            self.conv3 = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_use_global_stats=bn_use_global_stats,
                activation=None)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class BagNetUnit(HybridBlock):
    """
    BagNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size of the second body convolution.
    strides : int or tuple/list of 2 int
        Strides of the second body convolution.
    bn_use_global_stats : bool
        Whether global moving statistics is used instead of local batch-norm for BatchNorm layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides,
                 bn_use_global_stats,
                 **kwargs):
        super(BagNetUnit, self).__init__(**kwargs)
        self.resize_identity = (in_channels != out_channels) or (strides != 1)

        with self.name_scope():
            self.body = BagNetBottleneck(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                bn_use_global_stats=bn_use_global_stats)
            if self.resize_identity:
                self.identity_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    bn_use_global_stats=bn_use_global_stats,
                    activation=None)
            self.activ = nn.Activation("relu")

    def hybrid_forward(self, F, x):
        if self.resize_identity:
            identity = self.identity_conv(x)
        else:
            identity = x
        x = self.body(x)
        if self.resize_identity:
            identity = F.slice_like(identity, x, axes=(2, 3))
        x = x + identity
        x = self.activ(x)
        return x


class BagNetInitBlock(HybridBlock):
    """
    BagNet specific initial block.

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
        super(BagNetInitBlock, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels)
            self.conv2 = conv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                padding=0,
                bn_use_global_stats=bn_use_global_stats)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BagNet(HybridBlock):
    """
    BagNet model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_pool_size : int
        Size of the pooling windows for final pool.
    normal_kernel_sizes : list of int
        Count of the first units with 3x3 convolution window size for each stage.
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
                 final_pool_size,
                 normal_kernel_sizes,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(BagNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            self.features.add(BagNetInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                bn_use_global_stats=bn_use_global_stats))
            in_channels = init_block_channels
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        strides = 2 if (j == 0) and (i != len(channels) - 1) else 1
                        kernel_size = 3 if j < normal_kernel_sizes[i] else 1
                        stage.add(BagNetUnit(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            strides=strides,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                self.features.add(stage)
            self.features.add(nn.AvgPool2D(
                pool_size=final_pool_size,
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


def get_bagnet(field,
               model_name=None,
               pretrained=False,
               ctx=cpu(),
               root=os.path.join("~", ".mxnet", "models"),
               **kwargs):
    """
    Create BagNet model with specific parameters.

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

    layers = [3, 4, 6, 3]

    if field == 9:
        normal_kernel_sizes = [1, 1, 0, 0]
        final_pool_size = 27
    elif field == 17:
        normal_kernel_sizes = [1, 1, 1, 0]
        final_pool_size = 26
    elif field == 33:
        normal_kernel_sizes = [1, 1, 1, 1]
        final_pool_size = 24
    else:
        raise ValueError("Unsupported BagNet with field: {}".format(field))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = BagNet(
        channels=channels,
        init_block_channels=init_block_channels,
        final_pool_size=final_pool_size,
        normal_kernel_sizes=normal_kernel_sizes,
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


def bagnet9(**kwargs):
    """
    BagNet-9 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=9, model_name="bagnet9", **kwargs)


def bagnet17(**kwargs):
    """
    BagNet-17 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=17, model_name="bagnet17", **kwargs)


def bagnet33(**kwargs):
    """
    BagNet-33 model from 'Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet,'
    https://openreview.net/pdf?id=SkfMWhAqYQ.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_bagnet(field=33, model_name="bagnet33", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        bagnet9,
        bagnet17,
        bagnet33,
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
        assert (model != bagnet9 or weight_count == 15688744)
        assert (model != bagnet17 or weight_count == 16213032)
        assert (model != bagnet33 or weight_count == 18310184)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
