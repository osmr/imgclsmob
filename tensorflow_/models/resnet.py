"""
    ResNet & SE-ResNet, implemented in TensorFlow.
    Original papers:
    - 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    - 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['resnet', 'resnet10', 'resnet12', 'resnet14', 'resnet16', 'resnet18_wd4', 'resnet18_wd2', 'resnet18_w3d4',
           'resnet18', 'resnet34', 'resnet50', 'resnet50b', 'resnet101', 'resnet101b', 'resnet152', 'resnet152b',
           'resnet200', 'resnet200b', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet50b', 'seresnet101',
           'seresnet101b', 'seresnet152', 'seresnet152b', 'seresnet200', 'seresnet200b']

import os
import tensorflow as tf
from .common import conv2d, se_block


def res_conv(x,
             in_channels,
             out_channels,
             kernel_size,
             strides,
             padding,
             activate,
             name="res_conv"):
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
        name=name + "/conv")
    x = tf.layers.batch_normalization(
        inputs=x,
        axis=1,
        training=False,
        name=name + "/bn")
    if activate:
        x = tf.nn.relu(x, name=name + "/activ")
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
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        activate=activate,
        name=name)


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
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        activate=activate,
        name=name)


def res_block(x,
              in_channels,
              out_channels,
              strides,
              name="res_block"):
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
        name=name + "/conv1")
    x = res_conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=1,
        activate=False,
        name=name + "/conv2")
    return x


def res_bottleneck_block(x,
                         in_channels,
                         out_channels,
                         strides,
                         conv1_stride,
                         name="res_bottleneck_block"):
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
        name=name + "/conv1")
    x = res_conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        strides=(1 if conv1_stride else strides),
        activate=True,
        name=name + "/conv2")
    x = res_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=1,
        activate=False,
        name=name + "/conv3")
    return x


def res_unit(x,
             in_channels,
             out_channels,
             strides,
             bottleneck,
             conv1_stride,
             use_se,
             name="res_unit"):
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
            name=name + "/identity_conv")
    else:
        identity = x

    if bottleneck:
        x = res_bottleneck_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            conv1_stride=conv1_stride,
            name=name + "/body")
    else:
        x = res_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            name=name + "/body")

    if use_se:
        x = se_block(
            x=x,
            channels=out_channels,
            name=name + "/se")

    x = x + identity

    x = tf.nn.relu(x, name=name + "activ")
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
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=2,
        padding=3,
        activate=True,
        name=name + "/conv")
    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=3,
        strides=2,
        padding='same',
        data_format='channels_first',
        name=name + "/pool")
    return x


def resnet(x,
           channels,
           init_block_channels,
           bottleneck,
           conv1_stride,
           use_se,
           in_channels=3,
           classes=1000):
    """
    ResNet model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385. Also this class
    implements SE-ResNet from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    x : Tensor
        Input tensor.
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

    Returns
    -------
    Tensor
        Resulted tensor.
    """
    x = res_init_block(
        x=x,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            x = res_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bottleneck=bottleneck,
                conv1_stride=conv1_stride,
                use_se=use_se,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = tf.layers.average_pooling2d(
        inputs=x,
        pool_size=7,
        strides=1,
        data_format='channels_first',
        name="features/final_pool")

    x = tf.layers.flatten(x)
    x = tf.layers.dense(
        inputs=x,
        units=classes,
        name="output")

    return x


def get_resnet(blocks,
               conv1_stride=True,
               use_se=False,
               width_scale=1.0,
               model_name=None,
               pretrained=False,
               sess=None,
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
    sess: Session or None, default None
        A Session to use to load the weights.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.

    Returns
    -------
    Function
        Model script.
    Dict or None
        Model parameter dict.
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

    if pretrained and ((model_name is None) or (not model_name)):
        raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")

    def net_lambda(x,
                   channels=channels,
                   init_block_channels=init_block_channels,
                   bottleneck=bottleneck,
                   conv1_stride=conv1_stride,
                   use_se=use_se,
                   pretrained=pretrained,
                   sess=sess,
                   model_name=model_name,
                   root=root):
        y_net = resnet(
            x=x,
            channels=channels,
            init_block_channels=init_block_channels,
            bottleneck=bottleneck,
            conv1_stride=conv1_stride,
            use_se=use_se,
            **kwargs)
        if pretrained:
            from .model_store import download_model
            download_model(
                sess=sess,
                model_name=model_name,
                local_model_store_dir_path=root)
        return y_net

    return net_lambda


def resnet10(**kwargs):
    """
    ResNet-10 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
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
    sess: Session or None, default None
        A Session to use to load the weights.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_resnet(blocks=200, conv1_stride=False, use_se=True, model_name="seresnet200b", **kwargs)


def _test():
    import numpy as np

    pretrained = False

    models = [
        # resnet10,
        # resnet12,
        # resnet14,
        # resnet16,
        # resnet18_wd4,
        # resnet18_wd2,
        # resnet18_w3d4,

        resnet18,
        # resnet34,
        # resnet50,
        # resnet50b,
        # resnet101,
        # resnet101b,
        # resnet152,
        # resnet152b,
        # resnet200,
        # resnet200b,
        #
        # seresnet18,
        # seresnet34,
        # seresnet50,
        # seresnet50b,
        # seresnet101,
        # seresnet101b,
        # seresnet152,
        # seresnet152b,
        # seresnet200,
        # seresnet200b,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        x = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 3, 224, 224),
            name='xx')
        y_net = net(x)

        weight_count = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != resnet10 or weight_count == 5418792)
        assert (model != resnet12 or weight_count == 5492776)
        assert (model != resnet14 or weight_count == 5788200)
        assert (model != resnet16 or weight_count == 6968872)
        assert (model != resnet18_wd4 or weight_count == 831096)
        assert (model != resnet18_wd2 or weight_count == 3055880)
        assert (model != resnet18_w3d4 or weight_count == 6675352)
        assert (model != resnet18 or weight_count == 11689512)
        assert (model != resnet34 or weight_count == 21797672)
        assert (model != resnet50 or weight_count == 25557032)
        assert (model != resnet50b or weight_count == 25557032)
        assert (model != resnet101 or weight_count == 44549160)
        assert (model != resnet101b or weight_count == 44549160)
        assert (model != resnet152 or weight_count == 60192808)
        assert (model != resnet152b or weight_count == 60192808)
        assert (model != resnet200 or weight_count == 64673832)
        assert (model != resnet200b or weight_count == 64673832)
        assert (model != seresnet18 or weight_count == 11778592)
        assert (model != seresnet34 or weight_count == 21958868)
        assert (model != seresnet50 or weight_count == 28088024)
        assert (model != seresnet50b or weight_count == 28088024)
        assert (model != seresnet101 or weight_count == 49326872)
        assert (model != seresnet101b or weight_count == 49326872)
        assert (model != seresnet152 or weight_count == 66821848)
        assert (model != seresnet152b or weight_count == 66821848)
        assert (model != seresnet200 or weight_count == 71835864)
        assert (model != seresnet200b or weight_count == 71835864)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            x_value = np.zeros((1, 3, 224, 224), np.float32)
            y = sess.run(y_net, feed_dict={x: x_value})
            assert (y.shape == (1, 1000))
        tf.reset_default_graph()


if __name__ == "__main__":
    _test()
