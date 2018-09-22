"""
    PreResNet & SE-PreResNet, implemented in Keras.
    Original papers:
    - 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    - 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
"""

__all__ = ['preresnet', 'preresnet10', 'preresnet12', 'preresnet14', 'preresnet16', 'preresnet18_wd4',
           'preresnet18_wd2', 'preresnet18_w3d4', 'preresnet18', 'preresnet34', 'preresnet50', 'preresnet50b',
           'preresnet101', 'preresnet101b', 'preresnet152', 'preresnet152b', 'preresnet200', 'preresnet200b',
           'sepreresnet18', 'sepreresnet34', 'sepreresnet50', 'sepreresnet50b', 'sepreresnet101', 'sepreresnet101b',
           'sepreresnet152', 'sepreresnet152b', 'sepreresnet200', 'sepreresnet200b']

import os
from keras import backend as K
from keras import layers as nn
from keras.models import Model
from .common import conv2d, conv1x1, se_block, GluonBatchNormalization


def preres_conv(x,
                in_channels,
                out_channels,
                kernel_size,
                strides,
                padding,
                name="preres_conv"):
    """
    PreResNet specific convolution block, with pre-activation.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    name : str, default 'preres_conv'
        Block name.

    Returns
    -------
    tuple of two keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    x = GluonBatchNormalization(name=name+"/bn")(x)
    x = nn.Activation("relu", name=name+"/activ")(x)
    x_pre_activ = x
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        use_bias=False,
        name=name+"/conv")
    return x, x_pre_activ


def preres_conv1x1(x,
                   in_channels,
                   out_channels,
                   strides,
                   name="preres_conv1x1"):
    """
    1x1 version of the PreResNet specific convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    name : str, default 'preres_conv1x1'
        Block name.

    Returns
    -------
    tuple of two keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    return preres_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        strides=strides,
        padding=0,
        name=name)


def preres_conv3x3(x,
                   in_channels,
                   out_channels,
                   strides,
                   name="preres_conv3x3"):
    """
    3x3 version of the PreResNet specific convolution block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    name : str, default 'preres_conv3x3'
        Block name.

    Returns
    -------
    tuple of two keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    return preres_conv(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        strides=strides,
        padding=1,
        name=name)


def preres_block(x,
                 in_channels,
                 out_channels,
                 strides,
                 name="preres_block"):
    """
    Simple PreResNet block for residual path in PreResNet unit.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    name : str, default 'preres_block'
        Block name.

    Returns
    -------
    tuple of two keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    x, x_pre_activ = preres_conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=strides,
        name=name+"/conv1")
    x, _ = preres_conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=1,
        name=name+"/conv2")
    return x, x_pre_activ


def preres_bottleneck_block(x,
                            in_channels,
                            out_channels,
                            strides,
                            conv1_stride,
                            name="preres_bottleneck_block"):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    conv1_stride : bool
        Whether to use stride in the first or the second convolution layer of the block.
    name : str, default 'preres_bottleneck_block'
        Block name.

    Returns
    -------
    tuple of two keras.backend tensor/variable/symbol
        Resulted tensor and preactivated input tensor.
    """
    mid_channels = out_channels // 4

    x, x_pre_activ = preres_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        strides=(strides if conv1_stride else 1),
        name=name+"/conv1")
    x, _ = preres_conv3x3(
        x=x,
        in_channels=in_channels,
        out_channels=mid_channels,
        strides=(1 if conv1_stride else strides),
        name=name+"/conv2")
    x, _ = preres_conv1x1(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        strides=1,
        name=name+"/conv3")
    return x, x_pre_activ


def preres_unit(x,
                in_channels,
                out_channels,
                strides,
                bottleneck,
                conv1_stride,
                use_se,
                name="preres_unit"):
    """
    ResNet unit with residual connection.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
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
    name : str, default 'preres_unit'
        Unit name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor.
    """
    identity = x

    if bottleneck:
        x, x_pre_activ = preres_bottleneck_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            conv1_stride=conv1_stride,
            name=name+"/body")
    else:
        x, x_pre_activ = preres_block(
            x=x,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            name=name+"/body")

    if use_se:
        x = se_block(
            x=x,
            channels=out_channels,
            name=name+"/se")

    resize_identity = (in_channels != out_channels) or (strides != 1)
    if resize_identity:
        identity = conv1x1(
            out_channels=out_channels,
            strides=strides,
            name=name+"/identity_conv")(x_pre_activ)

    x = nn.add([x, identity])
    return x


def preres_init_block(x,
                      in_channels,
                      out_channels,
                      name="preres_init_block"):
    """
    PreResNet specific initial block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    name : str, default 'preres_init_block'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = conv2d(
        x=x,
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        strides=2,
        padding=3,
        use_bias=False,
        name=name+"/conv")
    x = GluonBatchNormalization(name=name+"/bn")(x)
    x = nn.Activation("relu", name=name+"/activ")(x)
    x = nn.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same',
        name=name+"/pool")(x)
    return x


def preres_activation(x,
                      name="preres_activation"):
    """
    PreResNet pure pre-activation block without convolution layer. It's used by itself as the final block.

    Parameters:
    ----------
    x : keras.backend tensor/variable/symbol
        Input tensor/variable/symbol.
    name : str, default 'preres_activation'
        Block name.

    Returns
    -------
    keras.backend tensor/variable/symbol
        Resulted tensor/variable/symbol.
    """
    x = GluonBatchNormalization(name=name+"/bn")(x)
    x = nn.Activation("relu", name=name+"/activ")(x)
    return x


def preresnet(channels,
              init_block_channels,
              bottleneck,
              conv1_stride,
              use_se,
              in_channels=3,
              classes=1000):
    """
    PreResNet model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027. Also this
    class implements SE-PreResNet from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

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
    input_shape = (in_channels, 224, 224) if K.image_data_format() == 'channels_first' else (224, 224, in_channels)
    input = nn.Input(shape=input_shape)

    x = preres_init_block(
        x=input,
        in_channels=in_channels,
        out_channels=init_block_channels,
        name="features/init_block")
    in_channels = init_block_channels
    for i, channels_per_stage in enumerate(channels):
        for j, out_channels in enumerate(channels_per_stage):
            strides = 2 if (j == 0) and (i != 0) else 1
            x = preres_unit(
                x=x,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                bottleneck=bottleneck,
                conv1_stride=conv1_stride,
                use_se=use_se,
                name="features/stage{}/unit{}".format(i + 1, j + 1))
            in_channels = out_channels
    x = preres_activation(
        x=x,
        name="features/post_activ")
    x = nn.AvgPool2D(
        pool_size=7,
        strides=1,
        name="features/final_pool")(x)

    x = nn.Flatten()(x)
    x = nn.Dense(
        units=classes,
        input_dim=in_channels,
        name="output")(x)

    model = Model(inputs=input, outputs=x)
    return model


def get_preresnet(blocks,
                  conv1_stride=True,
                  use_se=False,
                  width_scale=1.0,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join('~', '.keras', 'models'),
                  **kwargs):
    """
    Create PreResNet or SE-PreResNet model with specific parameters.

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
    root : str, default '~/.keras/models'
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

    net = preresnet(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        use_se=use_se,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import get_model_file
        net.load_weights(
            filepath=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root))

    return net


def preresnet10(**kwargs):
    """
    PreResNet-10 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=10, model_name="preresnet10", **kwargs)


def preresnet12(**kwargs):
    """
    PreResNet-12 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=12, model_name="preresnet12", **kwargs)


def preresnet14(**kwargs):
    """
    PreResNet-14 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=14, model_name="preresnet14", **kwargs)


def preresnet16(**kwargs):
    """
    PreResNet-16 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.
    It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=16, model_name="preresnet16", **kwargs)


def preresnet18_wd4(**kwargs):
    """
    PreResNet-18 model with 0.25 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.25, model_name="preresnet18_wd4", **kwargs)


def preresnet18_wd2(**kwargs):
    """
    PreResNet-18 model with 0.5 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.5, model_name="preresnet18_wd2", **kwargs)


def preresnet18_w3d4(**kwargs):
    """
    PreResNet-18 model with 0.75 width scale from 'Identity Mappings in Deep Residual Networks,'
    https://arxiv.org/abs/1603.05027. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, width_scale=0.75, model_name="preresnet18_w3d4", **kwargs)


def preresnet18(**kwargs):
    """
    PreResNet-18 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, model_name="preresnet18", **kwargs)


def preresnet34(**kwargs):
    """
    PreResNet-34 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=34, model_name="preresnet34", **kwargs)


def preresnet50(**kwargs):
    """
    PreResNet-50 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=50, model_name="preresnet50", **kwargs)


def preresnet50b(**kwargs):
    """
    PreResNet-50 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=50, conv1_stride=False, model_name="preresnet50b", **kwargs)


def preresnet101(**kwargs):
    """
    PreResNet-101 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=101, model_name="preresnet101", **kwargs)


def preresnet101b(**kwargs):
    """
    PreResNet-101 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=101, conv1_stride=False, model_name="preresnet101b", **kwargs)


def preresnet152(**kwargs):
    """
    PreResNet-152 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=152, model_name="preresnet152", **kwargs)


def preresnet152b(**kwargs):
    """
    PreResNet-152 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=152, conv1_stride=False, model_name="preresnet152b", **kwargs)


def preresnet200(**kwargs):
    """
    PreResNet-200 model from 'Identity Mappings in Deep Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=200, model_name="preresnet200", **kwargs)


def preresnet200b(**kwargs):
    """
    PreResNet-200 model with stride at the second convolution in bottleneck block from 'Identity Mappings in Deep
    Residual Networks,' https://arxiv.org/abs/1603.05027.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=200, conv1_stride=False, model_name="preresnet200b", **kwargs)


def sepreresnet18(**kwargs):
    """
    SE-PreResNet-18 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=18, use_se=True, model_name="sepreresnet18", **kwargs)


def sepreresnet34(**kwargs):
    """
    SE-PreResNet-34 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=34, use_se=True, model_name="sepreresnet34", **kwargs)


def sepreresnet50(**kwargs):
    """
    SE-PreResNet-50 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=50, use_se=True, model_name="sepreresnet50", **kwargs)


def sepreresnet50b(**kwargs):
    """
    SE-PreResNet-50 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=50, conv1_stride=False, use_se=True, model_name="sepreresnet50b", **kwargs)


def sepreresnet101(**kwargs):
    """
    SE-PreResNet-101 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=101, use_se=True, model_name="sepreresnet101", **kwargs)


def sepreresnet101b(**kwargs):
    """
    SE-PreResNet-101 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=101, conv1_stride=False, use_se=True, model_name="sepreresnet101b", **kwargs)


def sepreresnet152(**kwargs):
    """
    SE-PreResNet-152 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=152, use_se=True, model_name="sepreresnet152", **kwargs)


def sepreresnet152b(**kwargs):
    """
    SE-PreResNet-152 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=152, conv1_stride=False, use_se=True, model_name="sepreresnet152b", **kwargs)


def sepreresnet200(**kwargs):
    """
    SE-PreResNet-200 model from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507. It's an
    experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=200, use_se=True, model_name="sepreresnet200", **kwargs)


def sepreresnet200b(**kwargs):
    """
    SE-PreResNet-200 model with stride at the second convolution in bottleneck block from 'Squeeze-and-Excitation
    Networks,' https://arxiv.org/abs/1709.01507. It's an experimental model.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.keras/models'
        Location for keeping the model parameters.
    """
    return get_preresnet(blocks=200, conv1_stride=False, use_se=True, model_name="sepreresnet200b", **kwargs)


def _test():
    import numpy as np
    import keras

    pretrained = False

    models = [
        preresnet10,
        preresnet12,
        preresnet14,
        preresnet16,
        preresnet18_wd4,
        preresnet18_wd2,
        preresnet18_w3d4,

        preresnet18,
        preresnet34,
        preresnet50,
        preresnet50b,
        preresnet101,
        preresnet101b,
        preresnet152,
        preresnet152b,
        preresnet200,
        preresnet200b,

        sepreresnet18,
        sepreresnet34,
        sepreresnet50,
        sepreresnet50b,
        sepreresnet101,
        sepreresnet101b,
        sepreresnet152,
        sepreresnet152b,
        sepreresnet200,
        sepreresnet200b,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        #net.summary()
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != preresnet10 or weight_count == 5417128)
        assert (model != preresnet12 or weight_count == 5491112)
        assert (model != preresnet14 or weight_count == 5786536)
        assert (model != preresnet16 or weight_count == 6967208)
        assert (model != preresnet18_wd4 or weight_count == 830680)
        assert (model != preresnet18_wd2 or weight_count == 3055048)
        assert (model != preresnet18_w3d4 or weight_count == 6674104)
        assert (model != preresnet18 or weight_count == 11687848)
        assert (model != preresnet34 or weight_count == 21796008)
        assert (model != preresnet50 or weight_count == 25549480)
        assert (model != preresnet50b or weight_count == 25549480)
        assert (model != preresnet101 or weight_count == 44541608)
        assert (model != preresnet101b or weight_count == 44541608)
        assert (model != preresnet152 or weight_count == 60185256)
        assert (model != preresnet152b or weight_count == 60185256)
        assert (model != preresnet200 or weight_count == 64666280)
        assert (model != preresnet200b or weight_count == 64666280)
        assert (model != sepreresnet18 or weight_count == 11776928)
        assert (model != sepreresnet34 or weight_count == 21957204)
        assert (model != sepreresnet50 or weight_count == 28080472)
        assert (model != sepreresnet50b or weight_count == 28080472)
        assert (model != sepreresnet101 or weight_count == 49319320)
        assert (model != sepreresnet101b or weight_count == 49319320)
        assert (model != sepreresnet152 or weight_count == 66814296)
        assert (model != sepreresnet152b or weight_count == 66814296)
        assert (model != sepreresnet200 or weight_count == 71828312)
        assert (model != sepreresnet200b or weight_count == 71828312)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net.predict(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
