
__all__ = ['ShuffleNetV2', 'shufflenetv2_wd2_oth']


import os
import tensorflow as tf
from tensorpack.models import Conv2D, BNReLU, BatchNorm, MaxPooling, AvgPooling, GlobalAvgPooling, FullyConnected,\
    layer_register
from tensorpack.tfutils import argscope
from tensorflow_.tensorflowcv.models import ImageNetModel, depthwise_conv, channel_shuffle


@layer_register()
def shufflenet_unit_v2(x,
                       out_channels,
                       downsample):

    if downsample:
        shortcut, x = x, x
    else:
        shortcut, x = tf.split(x, 2, axis=1)

    shortcut_channel = int(shortcut.shape[1])
    mid_channels = out_channels // 2

    x = Conv2D(
        'conv1',
        x,
        filters=mid_channels,
        kernel_size=1,
        activation=BNReLU)
    x = depthwise_conv(
        'dconv',
        x,
        channels=mid_channels,
        kernel_size=3,
        strides=(2 if downsample else 1))
    x = BatchNorm('dconv_bn', x)
    x = Conv2D(
        'conv2',
        x,
        filters=(out_channels - shortcut_channel),
        kernel_size=1,
        activation=BNReLU)

    if downsample:
        shortcut = depthwise_conv(
            'shortcut_dconv',
            shortcut,
            channels=shortcut_channel,
            kernel_size=3,
            strides=2)
        shortcut = BatchNorm('shortcut_dconv_bn', shortcut)
        shortcut = Conv2D(
            'shortcut_conv',
            shortcut,
            filters=shortcut_channel,
            kernel_size=1,
            activation=BNReLU)
    output = tf.concat([shortcut, x], axis=1)
    output = channel_shuffle(output, 2)
    return output


@layer_register(log_shape=True)
def shufflenet_stage(x,
                     out_channels,
                     units_per_stage):
    for i in range(units_per_stage):
        name = 'block{}'.format(i)
        x = shufflenet_unit_v2(
            name,
            x,
            out_channels=out_channels,
            downsample=(i == 0))
    return x


class ShuffleNetV2(ImageNetModel):

    def __init__(self,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 classes=1000,
                 **kwargs):
        super(ShuffleNetV2, self).__init__(**kwargs)
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.final_block_channels = final_block_channels
        self.classes = classes
        self.weight_decay = 4e-5

    def get_logits(self, x, training=False):

        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm],
                      data_format='channels_first'), argscope(Conv2D, use_bias=False):

            x = Conv2D(
                'conv1',
                x,
                filters=self.init_block_channels,
                kernel_size=3,
                strides=2,
                activation=BNReLU)
            x = MaxPooling(
                'pool1',
                x,
                pool_size=3,
                strides=2,
                padding='SAME')

            x = shufflenet_stage('stage2', x, out_channels=self.channels[0], units_per_stage=4)
            x = shufflenet_stage('stage3', x, out_channels=self.channels[1], units_per_stage=8)
            x = shufflenet_stage('stage4', x, out_channels=self.channels[2], units_per_stage=4)

            x = Conv2D(
                'conv5',
                x,
                filters=self.final_block_channels,
                kernel_size=1,
                activation=BNReLU)

            x = GlobalAvgPooling('gap', x)
            x = FullyConnected(
                'linear',
                x,
                units=self.classes)
            return x


def get_shufflenetv2(ratio,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join('~', '.tensorflow', 'models'),
                     **kwargs):
    """
    Create ShuffleNetV2 model with specific parameters.

    Parameters:
    ----------
    ratio : float
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 24
    final_block_channels = 1024
    channels = {
        0.5: [48, 96, 192],
        1.0: [116, 232, 464]
    }[ratio]
    net = ShuffleNetV2(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        **kwargs)
    return net.get_logits


def shufflenetv2_wd2_oth(**kwargs):
    """
    ShuffleNetV2 0.5x model from 'ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,'
    https://arxiv.org/abs/1807.11164.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_shufflenetv2(ratio=0.5, model_name="shufflenetv2_wd2", **kwargs)


if __name__ == '__main__':
    pass
