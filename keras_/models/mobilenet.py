"""
    MobileNet & FD-MobileNet, implemented in Keras.
    Original papers:
    - 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
       https://arxiv.org/abs/1704.04861.
    - 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,' https://arxiv.org/abs/1802.03750.
"""

__all__ = ['MobileNet', 'mobilenet_w1', 'mobilenet_w3d4', 'mobilenet_wd2', 'mobilenet_wd4', 'fdmobilenet_w1',
           'fdmobilenet_w3d4', 'fdmobilenet_wd2', 'fdmobilenet_wd4']

import os
from keras.models import Model, Sequential
from keras import layers as nn


class ConvBlock(Model):
    """
    Standard enough convolution block with BatchNorm and activation.

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
    depthwise : bool, default False
        Whether depthwise convolution is used.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 strides=1,
                 padding=0,
                 depthwise=False,
                 **kwargs):
        super(ConvBlock, self).__init__(**kwargs)
        ke_padding = 'valid' if padding == 0 else 'same'
        if depthwise:
            self.conv = nn.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=strides,
                padding=ke_padding,
                use_bias=False)
        else:
            self.conv = nn.Conv2D(
                filters=out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=ke_padding,
                use_bias=False)
        self.bn = nn.BatchNormalization()
        self.activ = nn.Activation('relu')

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        return x


class DwsConvBlock(Model):
    """
    Depthwise separable convolution block with BatchNorms and activations at each convolution layers. It is used as
    a MobileNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 **kwargs):
        super(DwsConvBlock, self).__init__(**kwargs)
        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            strides=strides,
            padding=1,
            depthwise=True)
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)

    def call(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MobileNet(Model):
    """
    MobileNet model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861. Also this class implements FD-MobileNet from 'FD-MobileNet: Improved MobileNet
    with A Fast Downsampling Strategy,' https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_stage_stride : bool
        Whether stride is used at the first stage.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 first_stage_stride,
                 in_channels=3,
                 classes=1000,
                 **kwargs):
        super(MobileNet, self).__init__(**kwargs)

        self.features = Sequential()
        init_block_channels = channels[0][0]
        # self.features.add(ConvBlock(
        #     in_channels=in_channels,
        #     out_channels=init_block_channels,
        #     kernel_size=3,
        #     strides=2,
        #     padding=1))
        # in_channels = init_block_channels
        # for i, channels_per_stage in enumerate(channels[1:]):
        #     stage = Sequential()
        #     for j, out_channels in enumerate(channels_per_stage):
        #         strides = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
        #         stage.add(DwsConvBlock(
        #             in_channels=in_channels,
        #             out_channels=out_channels,
        #             strides=strides))
        #         in_channels = out_channels
        #     self.features.add(stage)

        # self.features.add(nn.Conv2D(
        #     filters=init_block_channels,
        #     kernel_size=3,
        #     strides=2,
        #     padding='same',
        #     use_bias=False))
        # self.features.add(nn.BatchNormalization())
        # self.features.add(nn.Activation('relu'))
        self.features.add(ConvBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=3,
            strides=2,
            padding=1))
        self.features.add(nn.AvgPool2D(
            pool_size=7,
            strides=1))

        self.classifier = Sequential()
        self.classifier.add(nn.Flatten())
        self.classifier.add(nn.Dense(units=classes))

    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_mobilenet(version,
                  width_scale,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join('~', '.keras', 'models'),
                  **kwargs):
    """
    Create MobileNet or FD-MobileNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of SqueezeNet ('orig' or 'fd').
    width_scale : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """

    if version == 'orig':
        channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
        first_stage_stride = False
    elif version == 'fd':
        channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 1024]]
        first_stage_stride = True
    else:
        raise ValueError("Unsupported MobileNet version {}".format(version))

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    net = MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride,
        **kwargs)

    if pretrained:
        raise ValueError("Pretrained model doesn't supported")
        # if (model_name is None) or (not model_name):
        #     raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        # from .model_store import get_model_file
        # net.load_parameters(
        #     filename=get_model_file(
        #         model_name=model_name,
        #         local_model_store_dir_path=root),
        #     ctx=ctx)

    return net


def mobilenet_w1(**kwargs):
    """
    1.0 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="orig", width_scale=1.0, model_name="mobilenet_w1", **kwargs)


def mobilenet_w3d4(**kwargs):
    """
    0.75 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="orig", width_scale=0.75, model_name="mobilenet_w3d4", **kwargs)


def mobilenet_wd2(**kwargs):
    """
    0.5 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="orig", width_scale=0.5, model_name="mobilenet_wd2", **kwargs)


def mobilenet_wd4(**kwargs):
    """
    0.25 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="orig", width_scale=0.25, model_name="mobilenet_wd4", **kwargs)


def fdmobilenet_w1(**kwargs):
    """
    FD-MobileNet 1.0x from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="fd", width_scale=1.0, model_name="fdmobilenet_w1", **kwargs)


def fdmobilenet_w3d4(**kwargs):
    """
    FD-MobileNet 0.75x from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="fd", width_scale=0.75, model_name="fdmobilenet_w3d4", **kwargs)


def fdmobilenet_wd2(**kwargs):
    """
    FD-MobileNet 0.5x from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="fd", width_scale=0.5, model_name="fdmobilenet_wd2", **kwargs)


def fdmobilenet_wd4(**kwargs):
    """
    FD-MobileNet 0.25x from 'FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy,'
    https://arxiv.org/abs/1802.03750.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(version="fd", width_scale=0.25, model_name="fdmobilenet_wd4", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    # class SimpleMLP(keras.Model):
    #
    #     def __init__(self, use_bn=False, use_dp=False, num_classes=1000):
    #         super(SimpleMLP, self).__init__(name='mlp')
    #         self.use_bn = use_bn
    #         self.use_dp = use_dp
    #         self.num_classes = num_classes
    #
    #         self.fl = nn.Flatten()
    #         self.dense1 = keras.layers.Dense(32, activation='relu')
    #         self.dense2 = keras.layers.Dense(num_classes, activation='softmax')
    #         if self.use_dp:
    #             self.dp = keras.layers.Dropout(0.5)
    #         if self.use_bn:
    #             self.bn = keras.layers.BatchNormalization(axis=-1)
    #
    #     def call(self, inputs):
    #         x = self.fl(inputs)
    #         x = self.dense1(x)
    #         if self.use_dp:
    #             x = self.dp(x)
    #         if self.use_bn:
    #             x = self.bn(x)
    #         return self.dense2(x)
    # model = SimpleMLP()

    # model = Sequential()
    # model.add(nn.Dense(512, activation='relu', input_shape=(3,,)))
    # model.add(nn.Dropout(0.2))
    # model.add(nn.Dense(512, activation='relu'))
    # model.add(nn.Dropout(0.2))
    # model.add(nn.Dense(1000, activation='softmax'))

    model = mobilenet_wd4()
    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    data = np.random.random((20, 3, 224, 224)).astype(np.float32)
    labels = keras.utils.to_categorical(np.random.randint(1000, size=(20, 1)), num_classes=1000)
    model.fit(data, labels, epochs=2, batch_size=10)
    model.summary()
    weight_count = model.count_params()
    print("{}".format(weight_count))

    # models = [
    #     mobilenet_w1,
    #     mobilenet_w3d4,
    #     mobilenet_wd2,
    #     mobilenet_wd4,
    #     fdmobilenet_w1,
    #     fdmobilenet_w3d4,
    #     fdmobilenet_wd2,
    #     fdmobilenet_wd4,
    # ]
    #
    # for model in models:
    #
    #     net = model(pretrained=pretrained)
    #     net.predict(np.zeros((1, 3, 224, 224), np.float32))
    #     #net.build(input_shape=(1, 3, 224, 224))
    #     weight_count = net.count_params()
    #     print("m={}, {}".format(model.__name__, weight_count))
    #     assert (model != mobilenet_w1 or weight_count == 4231976)
    #     assert (model != mobilenet_w3d4 or weight_count == 2585560)
    #     assert (model != mobilenet_wd2 or weight_count == 1331592)
    #     assert (model != mobilenet_wd4 or weight_count == 470072)
    #     assert (model != fdmobilenet_w1 or weight_count == 2901288)
    #     assert (model != fdmobilenet_w3d4 or weight_count == 1833304)
    #     assert (model != fdmobilenet_wd2 or weight_count == 993928)
    #     assert (model != fdmobilenet_wd4 or weight_count == 383160)
    #
    #     x = mx.nd.zeros((1, 3, 224, 224), ctx=ctx)
    #     y = net(x)
    #     assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
