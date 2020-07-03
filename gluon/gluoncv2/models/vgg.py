"""
    VGG for ImageNet-1K, implemented in Gluon.
    Original paper: 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.
"""

__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'bn_vgg11', 'bn_vgg13', 'bn_vgg16', 'bn_vgg19', 'bn_vgg11b',
           'bn_vgg13b', 'bn_vgg16b', 'bn_vgg19b']

import os
from mxnet import cpu
from mxnet.gluon import nn, HybridBlock
from .common import conv3x3_block


class VGGDense(HybridBlock):
    """
    VGG specific dense block.

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
        super(VGGDense, self).__init__(**kwargs)
        with self.name_scope():
            self.fc = nn.Dense(
                units=out_channels,
                in_units=in_channels)
            self.activ = nn.Activation("relu")
            self.dropout = nn.Dropout(rate=0.5)

    def hybrid_forward(self, F, x):
        x = self.fc(x)
        x = self.activ(x)
        x = self.dropout(x)
        return x


class VGGOutputBlock(HybridBlock):
    """
    VGG specific output block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    classes : int
        Number of classification classes.
    """
    def __init__(self,
                 in_channels,
                 classes,
                 **kwargs):
        super(VGGOutputBlock, self).__init__(**kwargs)
        mid_channels = 4096

        with self.name_scope():
            self.fc1 = VGGDense(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.fc2 = VGGDense(
                in_channels=mid_channels,
                out_channels=mid_channels)
            self.fc3 = nn.Dense(
                units=classes,
                in_units=mid_channels)

    def hybrid_forward(self, F, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class VGG(HybridBlock):
    """
    VGG models from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default False
        Whether to use BatchNorm layers.
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
                 use_bias=True,
                 use_bn=False,
                 bn_use_global_stats=False,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 **kwargs):
        super(VGG, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes

        with self.name_scope():
            self.features = nn.HybridSequential(prefix="")
            for i, channels_per_stage in enumerate(channels):
                stage = nn.HybridSequential(prefix="stage{}_".format(i + 1))
                with stage.name_scope():
                    for j, out_channels in enumerate(channels_per_stage):
                        stage.add(conv3x3_block(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            use_bias=use_bias,
                            use_bn=use_bn,
                            bn_use_global_stats=bn_use_global_stats))
                        in_channels = out_channels
                    stage.add(nn.MaxPool2D(
                        pool_size=2,
                        strides=2,
                        padding=0))
                self.features.add(stage)

            self.output = nn.HybridSequential(prefix="")
            self.output.add(nn.Flatten())
            in_channels = in_channels * 7 * 7
            self.output.add(VGGOutputBlock(
                in_channels=in_channels,
                classes=classes))

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_vgg(blocks,
            use_bias=True,
            use_bn=False,
            model_name=None,
            pretrained=False,
            ctx=cpu(),
            root=os.path.join("~", ".mxnet", "models"),
            **kwargs):
    """
    Create VGG model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default False
        Whether to use BatchNorm layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    if blocks == 11:
        layers = [1, 1, 2, 2, 2]
    elif blocks == 13:
        layers = [2, 2, 2, 2, 2]
    elif blocks == 16:
        layers = [2, 2, 3, 3, 3]
    elif blocks == 19:
        layers = [2, 2, 4, 4, 4]
    else:
        raise ValueError("Unsupported VGG with number of blocks: {}".format(blocks))

    channels_per_layers = [64, 128, 256, 512, 512]
    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = VGG(
        channels=channels,
        use_bias=use_bias,
        use_bn=use_bn,
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


def vgg11(**kwargs):
    """
    VGG-11 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=11, model_name="vgg11", **kwargs)


def vgg13(**kwargs):
    """
    VGG-13 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=13, model_name="vgg13", **kwargs)


def vgg16(**kwargs):
    """
    VGG-16 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=16, model_name="vgg16", **kwargs)


def vgg19(**kwargs):
    """
    VGG-19 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=19, model_name="vgg19", **kwargs)


def bn_vgg11(**kwargs):
    """
    VGG-11 model with batch normalization from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=11, use_bias=False, use_bn=True, model_name="bn_vgg11", **kwargs)


def bn_vgg13(**kwargs):
    """
    VGG-13 model with batch normalization from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=13, use_bias=False, use_bn=True, model_name="bn_vgg13", **kwargs)


def bn_vgg16(**kwargs):
    """
    VGG-16 model with batch normalization from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=16, use_bias=False, use_bn=True, model_name="bn_vgg16", **kwargs)


def bn_vgg19(**kwargs):
    """
    VGG-19 model with batch normalization from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=19, use_bias=False, use_bn=True, model_name="bn_vgg19", **kwargs)


def bn_vgg11b(**kwargs):
    """
    VGG-11 model with batch normalization and biases in convolution layers from 'Very Deep Convolutional Networks for
    Large-Scale Image Recognition,' https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=11, use_bias=True, use_bn=True, model_name="bn_vgg11b", **kwargs)


def bn_vgg13b(**kwargs):
    """
    VGG-13 model with batch normalization and biases in convolution layers from 'Very Deep Convolutional Networks for
    Large-Scale Image Recognition,' https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=13, use_bias=True, use_bn=True, model_name="bn_vgg13b", **kwargs)


def bn_vgg16b(**kwargs):
    """
    VGG-16 model with batch normalization and biases in convolution layers from 'Very Deep Convolutional Networks for
    Large-Scale Image Recognition,' https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=16, use_bias=True, use_bn=True, model_name="bn_vgg16b", **kwargs)


def bn_vgg19b(**kwargs):
    """
    VGG-19 model with batch normalization and biases in convolution layers from 'Very Deep Convolutional Networks for
    Large-Scale Image Recognition,' https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=19, use_bias=True, use_bn=True, model_name="bn_vgg19b", **kwargs)


def _test():
    import numpy as np
    import mxnet as mx

    pretrained = False

    models = [
        vgg11,
        vgg13,
        vgg16,
        vgg19,
        bn_vgg11,
        bn_vgg13,
        bn_vgg16,
        bn_vgg19,
        bn_vgg11b,
        bn_vgg13b,
        bn_vgg16b,
        bn_vgg19b,
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
        assert (model != vgg11 or weight_count == 132863336)
        assert (model != vgg13 or weight_count == 133047848)
        assert (model != vgg16 or weight_count == 138357544)
        assert (model != vgg19 or weight_count == 143667240)
        assert (model != bn_vgg11 or weight_count == 132866088)
        assert (model != bn_vgg13 or weight_count == 133050792)
        assert (model != bn_vgg16 or weight_count == 138361768)
        assert (model != bn_vgg19 or weight_count == 143672744)
        assert (model != bn_vgg11b or weight_count == 132868840)
        assert (model != bn_vgg13b or weight_count == 133053736)
        assert (model != bn_vgg16b or weight_count == 138365992)
        assert (model != bn_vgg19b or weight_count == 143678248)

        x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
