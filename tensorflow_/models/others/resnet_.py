"""
    ResNet & SE-ResNet, implemented in TensorFlow.
    Original papers:
    - 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    - 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['ResNet', 'resnet10', 'resnet12', 'resnet14', 'resnet16', 'resnet18_wd4', 'resnet18_wd2', 'resnet18_w3d4',
           'resnet18', 'resnet34', 'resnet50', 'resnet50b', 'resnet101', 'resnet101b', 'resnet152', 'resnet152b',
           'resnet200', 'resnet200b', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet50b', 'seresnet101',
           'seresnet101b', 'seresnet152', 'seresnet152b', 'seresnet200', 'seresnet200b']

import os
import tensorflow as tf
# import tensorflow.layers as nn
from tensorpack.models import Conv2D, BatchNorm, MaxPooling, AvgPooling, GlobalAvgPooling, FullyConnected,\
    layer_register
from tensorpack.tfutils import argscope
import tensorflow.contrib.slim as slim
from tensorflow_.models.common import ImageNetModel, conv2d, se_block


@layer_register(log_shape=True)
def res_conv(x,
             in_channels,
             out_channels,
             kernel_size,
             strides,
             padding,
             activate):
    """
    ResNet specific convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
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
    activate : bool
        Whether activate the convolution block.
    name : str, default 'res_conv'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name="conv")
    #x = BatchNorm("bn", x)
    x = tf.layers.batch_normalization(
        inputs=x,
        axis=1,
        name="bn")
    if activate:
        x = tf.nn.relu(x, name="activ")
    return x


def res_conv1x1(x,
                in_channels,
                out_channels,
                strides,
                activate,
                name="res_conv1x1"):
    """
    1x1 version of the ResNet specific convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    name : str, default 'res_conv1x1'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return res_conv(
        name,
        x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        activate=activate)


def res_conv3x3(x,
                in_channels,
                out_channels,
                strides,
                activate,
                name="res_conv3x3"):
    """
    3x3 version of the ResNet specific convolution block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    activate : bool
        Whether activate the convolution block.
    name : str, default 'res_conv3x3'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    return res_conv(
        name,
        x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        activate=activate)


@layer_register(log_shape=True)
def res_block(x,
              in_channels,
              out_channels,
              strides):
    """
    Simple ResNet block for residual path in ResNet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    name : str, default 'res_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = res_conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        activate=True,
        name="conv1")
    x = res_conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=1,
        activate=False,
        name="conv2")
    return x


@layer_register(log_shape=True)
def res_bottleneck_block(x,
                         in_channels,
                         out_channels,
                         strides,
                         conv1_stride):
    """
    ResNet bottleneck block for residual path in ResNet unit.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    name : str, default 'res_bottleneck_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    mid_channels = out_channels // 4

    x = res_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        strides=(strides if conv1_stride else 1),
        activate=True,
        name="conv1")
    x = res_conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        strides=(1 if conv1_stride else strides),
        activate=True,
        name="conv2")
    x = res_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=1,
        activate=False,
        name="conv3")
    return x


@layer_register(log_shape=True)
def res_unit(x,
             in_channels,
             out_channels,
             strides,
             bottleneck,
             conv1_stride,
             use_se):
    """
    ResNet unit with residual connection.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck : bool
        Whether to use a bottleneck or simple block in units.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    use_se : bool
        Whether to use SE block.
    name : str, default 'res_unit'
        Unit name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    resize_identity = (in_channels != out_channels) or (strides != 1)
    if resize_identity:
        identity = res_conv1x1(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            activate=False,
            name="identity_conv")
    else:
        identity = x

    if bottleneck:
        x = res_bottleneck_block(
            "body",
            x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            conv1_stride=conv1_stride)
    else:
        x = res_block(
            "body",
            x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides)

    if use_se:
        x = se_block(
            "se",
            x,
            channels=out_channels)

    x = x + identity

    x = tf.nn.relu(x, name="activ")
    return x


def res_init_block(x,
                   in_channels,
                   out_channels,
                   name):
    """
    ResNet specific initial block.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'res_init_block'
        Block name.

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = res_conv(
        name+"/conv",
        x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=2,
        padding=3,
        activate=True)
    # x = MaxPooling(
    #     name+"/pool",
    #     x,
    #     pool_size=3,
    #     strides=2,
    #     padding='SAME')
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=3,
        strides=2,
        padding='same',
        data_format='channels_first',
        name=name + "/pool")
    # x = slim.layers.max_pool2d(
    #     inputs=x,
    #     kernel_size=[3, 3],
    #     stride=2,
    #     padding='SAME',
    #     data_format=slim.layers.DATA_FORMAT_NCHW,
    #     scope=name + "/pool")
    return x


class ResNet(ImageNetModel):
    """
    ResNet model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385. Also this class
    implements SE-ResNet from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    use_se : bool
        Whether to use SE block.
    in_channels : int, default 3
        Number of input channels.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 use_se,
                 in_channels=3,
                 classes=1000):
        super(ResNet, self).__init__()
        self.channels = channels
        self.init_block_channels = init_block_channels
        self.bottleneck = bottleneck
        self.conv1_stride = conv1_stride
        self.use_se = use_se
        self.in_channels = in_channels
        self.classes = classes
        self.weight_decay = 4e-5

    def get_logits(self, x):
        with argscope([Conv2D, MaxPooling, AvgPooling, GlobalAvgPooling, BatchNorm], data_format='channels_first'):

            x = res_init_block(
                x=x,
                in_channels=self.in_channels,
                out_channels=self.init_block_channels,
                name="features/init_block")
            in_channels = self.init_block_channels
            for i, channels_per_stage in enumerate(self.channels):
                for j, out_channels in enumerate(channels_per_stage):
                    strides = 2 if (j == 0) and (i != 0) else 1
                    x = res_unit(
                        "features/stage{}/unit{}".format(i + 1, j + 1),
                        x,
                        in_channels=in_channels,
                        out_channels=out_channels,
                        strides=strides,
                        bottleneck=self.bottleneck,
                        conv1_stride=self.conv1_stride,
                        use_se=self.use_se)
                    in_channels = out_channels
            # x = AvgPooling(
            #     "final_pool",
            #     x,
            #     pool_size=7,
            #     strides=1)
            x = GlobalAvgPooling("features/final_pool", x)

            x = FullyConnected(
                "output",
                x,
                units=self.classes)

            return x


def get_resnet(blocks,
               conv1_stride=True,
               use_se=False,
               width_scale=1.0,
               model_name=None,
               pretrained=False,
               root=os.path.join('~', '.tensorflow', 'models'),
               **kwargs):
    """
    Create ResNet or SE-ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer in units.
    use_se : bool
        Whether to use SE block.
    width_scale : float
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
        raise ValueError("Unsupported ResNet with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]
        init_block_channels = int(init_block_channels * width_scale)

    net = ResNet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        use_se=use_se,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        # from .model_store import get_model_file
        # net.load_weights(
        #     filepath=get_model_file(
        #         model_name=model_name,
        #         local_model_store_dir_path=root))

    return net


def resnet10(**kwargs):
    """
    ResNet-10 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=10, model_name="resnet10", **kwargs)


def resnet12(**kwargs):
    """
    ResNet-12 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=12, model_name="resnet12", **kwargs)


def resnet14(**kwargs):
    """
    ResNet-14 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=14, model_name="resnet14", **kwargs)


def resnet16(**kwargs):
    """
    ResNet-16 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=16, model_name="resnet16", **kwargs)


def resnet18_wd4(**kwargs):
    """
    ResNet-18 model with 0.25 width scale from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, width_scale=0.25, model_name="resnet18_wd4", **kwargs)


def resnet18_wd2(**kwargs):
    """
    ResNet-18 model with 0.5 width scale from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, width_scale=0.5, model_name="resnet18_wd2", **kwargs)


def resnet18_w3d4(**kwargs):
    """
    ResNet-18 model with 0.75 width scale from 'Deep Residual Learning for Image Recognition,'
    https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, width_scale=0.75, model_name="resnet18_w3d4", **kwargs)


def resnet18(**kwargs):
    """
    ResNet-18 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, model_name="resnet18", **kwargs)


def resnet34(**kwargs):
    """
    ResNet-34 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, model_name="resnet34", **kwargs)


def resnet50(**kwargs):
    """
    ResNet-50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, model_name="resnet50", **kwargs)


def resnet50b(**kwargs):
    """
    ResNet-50 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, conv1_stride=False, model_name="resnet50b", **kwargs)


def resnet101(**kwargs):
    """
    ResNet-101 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, model_name="resnet101", **kwargs)


def resnet101b(**kwargs):
    """
    ResNet-101 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, conv1_stride=False, model_name="resnet101b", **kwargs)


def resnet152(**kwargs):
    """
    ResNet-152 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, model_name="resnet152", **kwargs)


def resnet152b(**kwargs):
    """
    ResNet-152 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, conv1_stride=False, model_name="resnet152b", **kwargs)


def resnet200(**kwargs):
    """
    ResNet-200 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=200, model_name="resnet200", **kwargs)


def resnet200b(**kwargs):
    """
    ResNet-200 model with stride at the second convolution in bottleneck block from 'Deep Residual Learning for Image
    Recognition,' https://arxiv.org/abs/1512.03385. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=200, conv1_stride=False, model_name="resnet200b", **kwargs)


def seresnet18(**kwargs):
    """
    SE-ResNet-18 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=18, use_se=True, model_name="seresnet18", **kwargs)


def seresnet34(**kwargs):
    """
    SE-ResNet-34 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=34, use_se=True, model_name="seresnet34", **kwargs)


def seresnet50(**kwargs):
    """
    SE-ResNet-50 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, use_se=True, model_name="seresnet50", **kwargs)


def seresnet50b(**kwargs):
    """
    SE-ResNet-50 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=50, conv1_stride=False, use_se=True, model_name="seresnet50b", **kwargs)


def seresnet101(**kwargs):
    """
    SE-ResNet-101 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, use_se=True, model_name="seresnet101", **kwargs)


def seresnet101b(**kwargs):
    """
    SE-ResNet-101 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=101, conv1_stride=False, use_se=True, model_name="seresnet101b", **kwargs)


def seresnet152(**kwargs):
    """
    SE-ResNet-152 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, use_se=True, model_name="seresnet152", **kwargs)


def seresnet152b(**kwargs):
    """
    SE-ResNet-152 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=152, conv1_stride=False, use_se=True, model_name="seresnet152b", **kwargs)


def seresnet200(**kwargs):
    """
    SE-ResNet-200 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=200, use_se=True, model_name="seresnet200", **kwargs)


def seresnet200b(**kwargs):
    """
    SE-ResNet-200 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=200, conv1_stride=False, use_se=True, model_name="seresnet200b", **kwargs)


def _test():
    import numpy as np
    # from tensorpack.tfutils import TowerContext
    from tensorpack import PredictConfig, OfflinePredictor

    pretrained = False

    models = [
        resnet10,
        resnet12,
        resnet14,
        resnet16,
        resnet18_wd4,
        resnet18_wd2,
        resnet18_w3d4,

        resnet18,
        resnet34,
        resnet50,
        resnet50b,
        resnet101,
        resnet101b,
        resnet152,
        resnet152b,
        resnet200,
        resnet200b,

        seresnet18,
        seresnet34,
        seresnet50,
        seresnet50b,
        seresnet101,
        seresnet101b,
        seresnet152,
        seresnet152b,
        seresnet200,
        seresnet200b,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        pred_config = PredictConfig(
            session_init=None,
            model=net,
            input_names=['input'],
            output_names=['label'])

        pred = OfflinePredictor(pred_config)
        img = np.zeros((224, 224, 3), np.uint8)
        prediction = pred([img])[0]
        print(prediction)
        pass

        # x = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
        # y = tf.placeholder(tf.int32, shape=(1, ))
        # with TowerContext('', is_training=False):
        #     net.build_graph(x, y)
        #     pass

        # weight_count = ...
        # print("m={}, {}".format(model.__name__, weight_count))
        # assert (model != resnet10 or weight_count == 5418792)
        # assert (model != resnet12 or weight_count == 5492776)
        # assert (model != resnet14 or weight_count == 5788200)
        # assert (model != resnet16 or weight_count == 6968872)
        # assert (model != resnet18_wd4 or weight_count == 831096)
        # assert (model != resnet18_wd2 or weight_count == 3055880)
        # assert (model != resnet18_w3d4 or weight_count == 6675352)
        # assert (model != resnet18 or weight_count == 11689512)
        # assert (model != resnet34 or weight_count == 21797672)
        # assert (model != resnet50 or weight_count == 25557032)
        # assert (model != resnet50b or weight_count == 25557032)
        # assert (model != resnet101 or weight_count == 44549160)
        # assert (model != resnet101b or weight_count == 44549160)
        # assert (model != resnet152 or weight_count == 60192808)
        # assert (model != resnet152b or weight_count == 60192808)
        # assert (model != resnet200 or weight_count == 64673832)
        # assert (model != resnet200b or weight_count == 64673832)
        # assert (model != seresnet18 or weight_count == 11778592)
        # assert (model != seresnet34 or weight_count == 21958868)
        # assert (model != seresnet50 or weight_count == 28088024)
        # assert (model != seresnet50b or weight_count == 28088024)
        # assert (model != seresnet101 or weight_count == 49326872)
        # assert (model != seresnet101b or weight_count == 49326872)
        # assert (model != seresnet152 or weight_count == 66821848)
        # assert (model != seresnet152b or weight_count == 66821848)
        # assert (model != seresnet200 or weight_count == 71835864)
        # assert (model != seresnet200b or weight_count == 71835864)

        # x = np.zeros((1, 3, 224, 224), np.float32)
        # y = net.predict(x)
        # assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
