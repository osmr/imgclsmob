"""
    ResNet(D) with dilation for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['ResNetD', 'resnetd50b', 'resnetd101b', 'resnetd152b']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import MultiOutputSequential, SimpleSequential, is_channels_first
from .resnet import ResUnit, ResInitBlock
from .senet import SEInitBlock


class ResNetD(tf.keras.Model):
    """
    ResNet(D) with dilation model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    ordinary_init : bool, default False
        Whether to use original initial block or SENet one.
    bends : tuple of int, default None
        Numbers of bends for multiple output.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 ordinary_init=False,
                 bends=None,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(ResNetD, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.multi_output = (bends is not None)
        self.data_format = data_format

        self.features = MultiOutputSequential(name="features")
        if ordinary_init:
            self.features.add(ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                data_format=data_format,
                name="init_block"))
        else:
            init_block_channels = 2 * init_block_channels
            self.features.add(SEInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels,
                data_format=data_format,
                name="init_block"))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                strides = 2 if ((j == 0) and (i != 0) and (i < 2)) else 1
                dilation = (2 ** max(0, i - 1 - int(j == 0)))
                stage.add(ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    padding=dilation,
                    dilation=dilation,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels = out_channels
            if self.multi_output and ((i + 1) in bends):
                stage.do_output = True
            self.features.add(stage)
        self.features.add(nn.GlobalAvgPool2D(
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1")

    def call(self, x, training=None):
        outs = self.features(x, training=training)
        x = outs[0]
        x = self.output1(x)
        if self.multi_output:
            return [x] + outs[1:]
        else:
            return x


def get_resnetd(blocks,
                conv1_stride=True,
                width_scale=1.0,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".tensorflow", "models"),
                **kwargs):
    """
    Create ResNet(D) with dilation model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if blocks == 10:
        layers = [1, 1, 1, 1]
    elif blocks == 12:
        layers = [2, 1, 1, 1]
    elif blocks == 14:
        layers = [2, 2, 1, 1]
    elif blocks == 16:
        layers = [2, 2, 2, 1]
    elif blocks == 18:
        layers = [2, 2, 2, 2]
    elif blocks == 34:
        layers = [3, 4, 6, 3]
    elif blocks == 50:
        layers = [3, 4, 6, 3]
    elif blocks == 101:
        layers = [3, 4, 23, 3]
    elif blocks == 152:
        layers = [3, 8, 36, 3]
    elif blocks == 200:
        layers = [3, 24, 36, 3]
    else:
        raise ValueError("Unsupported ResNet(D) with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = ResNetD(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        in_channels = kwargs["in_channels"] if ("in_channels" in kwargs) else 3
        input_shape = (1,) + (in_channels,) + net.in_size if net.data_format == "channels_first" else\
            (1,) + net.in_size + (in_channels,)
        net.build(input_shape=input_shape)
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def resnetd50b(**kwargs):
    """
    ResNet(D)-50 with dilation model with stride at the second convolution in bottleneck block from 'Deep Residual
    Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnetd(blocks=50, conv1_stride=False, model_name="resnetd50b", **kwargs)


def resnetd101b(**kwargs):
    """
    ResNet(D)-101 with dilation model with stride at the second convolution in bottleneck block from 'Deep Residual
    Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnetd(blocks=101, conv1_stride=False, model_name="resnetd101b", **kwargs)


def resnetd152b(**kwargs):
    """
    ResNet(D)-152 with dilation model with stride at the second convolution in bottleneck block from 'Deep Residual
    Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnetd(blocks=152, conv1_stride=False, model_name="resnetd152b", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    # data_format = "channels_first"
    ordinary_init = False
    bends = None
    pretrained = False

    models = [
        resnetd50b,
        resnetd101b,
        resnetd152b,
    ]

    for model in models:

        net = model(
            pretrained=pretrained,
            ordinary_init=ordinary_init,
            bends=bends,
            data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        if bends is not None:
            y = y[0]
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        if ordinary_init:
            assert (model != resnetd50b or weight_count == 25557032)
            assert (model != resnetd101b or weight_count == 44549160)
            assert (model != resnetd152b or weight_count == 60192808)
        else:
            assert (model != resnetd50b or weight_count == 25680808)
            assert (model != resnetd101b or weight_count == 44672936)
            assert (model != resnetd152b or weight_count == 60316584)


if __name__ == "__main__":
    _test()
