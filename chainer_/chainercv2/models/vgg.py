"""
    VGG for ImageNet-1K, implemented in Chainer.
    Original paper: 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.
"""

__all__ = ['VGG', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'bn_vgg11', 'bn_vgg13', 'bn_vgg16', 'bn_vgg19', 'bn_vgg11b',
           'bn_vgg13b', 'bn_vgg16b', 'bn_vgg19b']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3_block, SimpleSequential


class VGGDense(Chain):
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
                 out_channels):
        super(VGGDense, self).__init__()
        with self.init_scope():
            self.fc = L.Linear(
                in_size=in_channels,
                out_size=out_channels)
            self.activ = F.relu
            self.dropout = partial(
                F.dropout,
                ratio=0.5)

    def __call__(self, x):
        x = self.fc(x)
        x = self.activ(x)
        x = self.dropout(x)
        return x


class VGGOutputBlock(Chain):
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
                 classes):
        super(VGGOutputBlock, self).__init__()
        mid_channels = 4096

        with self.init_scope():
            self.fc1 = VGGDense(
                in_channels=in_channels,
                out_channels=mid_channels)
            self.fc2 = VGGDense(
                in_channels=mid_channels,
                out_channels=mid_channels)
            self.fc3 = L.Linear(
                in_size=mid_channels,
                out_size=classes)

    def __call__(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class VGG(Chain):
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
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(VGG, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            setattr(stage, "unit{}".format(j + 1), conv3x3_block(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                use_bias=use_bias,
                                use_bn=use_bn))
                            in_channels = out_channels
                        setattr(stage, "pool{}".format(i + 1), partial(
                            F.max_pooling_2d,
                            ksize=2,
                            stride=2,
                            pad=0))
                    setattr(self.features, "stage{}".format(i + 1), stage)

            in_channels = in_channels * 7 * 7
            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "classifier", VGGOutputBlock(
                    in_channels=in_channels,
                    classes=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_vgg(blocks,
            use_bias=True,
            use_bn=False,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def vgg11(**kwargs):
    """
    VGG-11 model from 'Very Deep Convolutional Networks for Large-Scale Image Recognition,'
    https://arxiv.org/abs/1409.1556.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_vgg(blocks=19, use_bias=True, use_bn=True, model_name="bn_vgg19b", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

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
        weight_count = net.count_params()
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

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
