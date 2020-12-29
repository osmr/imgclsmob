"""
    DLA for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.
"""

__all__ = ['DLA', 'dla34', 'dla46c', 'dla46xc', 'dla60', 'dla60x', 'dla60xc', 'dla102', 'dla102x', 'dla102x2', 'dla169']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1, conv1x1_block, conv3x3_block, conv7x7_block, SimpleSequential, flatten, is_channels_first,\
    get_channel_axis
from .resnet import ResBlock, ResBottleneck
from .resnext import ResNeXtBottleneck


class DLABottleneck(ResBottleneck):
    """
    DLA bottleneck block for residual path in residual block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    bottleneck_factor : int, default 2
        Bottleneck factor.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 bottleneck_factor=2,
                 data_format="channels_last",
                 **kwargs):
        super(DLABottleneck, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            bottleneck_factor=bottleneck_factor,
            data_format=data_format,
            **kwargs)


class DLABottleneckX(ResNeXtBottleneck):
    """
    DLA ResNeXt-like bottleneck block for residual path in residual block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    cardinality: int, default 32
        Number of groups.
    bottleneck_width: int, default 8
        Width of bottleneck block.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 cardinality=32,
                 bottleneck_width=8,
                 data_format="channels_last",
                 **kwargs):
        super(DLABottleneckX, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            cardinality=cardinality,
            bottleneck_width=bottleneck_width,
            data_format=data_format,
            **kwargs)


class DLAResBlock(nn.Layer):
    """
    DLA residual block with residual connection.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    strides : int or tuple/list of 2 int
        Strides of the convolution.
    body_class : nn.Module, default ResBlock
        Residual block body class.
    return_down : bool, default False
        Whether return downsample result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 strides,
                 body_class=ResBlock,
                 return_down=False,
                 data_format="channels_last",
                 **kwargs):
        super(DLAResBlock, self).__init__(**kwargs)
        self.return_down = return_down
        self.downsample = (strides > 1)
        self.project = (in_channels != out_channels)

        self.body = body_class(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            data_format=data_format,
            name="body")
        self.activ = nn.ReLU()
        if self.downsample:
            self.downsample_pool = nn.MaxPool2D(
                pool_size=strides,
                strides=strides,
                data_format=data_format,
                name="downsample_pool")
        if self.project:
            self.project_conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                activation=None,
                data_format=data_format,
                name="project_conv")

    def call(self, x, training=None):
        down = self.downsample_pool(x) if self.downsample else x
        identity = self.project_conv(down, training=training) if self.project else down
        if identity is None:
            identity = x
        x = self.body(x, training=training)
        x = x + identity
        x = self.activ(x)
        if self.return_down:
            return x, down
        else:
            return x


class DLARoot(nn.Layer):
    """
    DLA root block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    residual : bool
        Whether use residual connection.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 residual,
                 data_format="channels_last",
                 **kwargs):
        super(DLARoot, self).__init__(**kwargs)
        self.residual = residual
        self.data_format = data_format

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=None,
            data_format=data_format,
            name="conv")
        self.activ = nn.ReLU()

    def call(self, x2, x1, extra, training=None):
        last_branch = x2
        x = tf.concat([x2, x1] + list(extra), axis=get_channel_axis(self.data_format))
        x = self.conv(x, training=training)
        if self.residual:
            x += last_branch
        x = self.activ(x)
        return x


class DLATree(nn.Layer):
    """
    DLA tree unit. It's like iterative stage.

    Parameters:
    ----------
    levels : int
        Number of levels in the stage.
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    res_body_class : nn.Module
        Residual block body class.
    strides : int or tuple/list of 2 int
        Strides of the convolution in a residual block.
    root_residual : bool
        Whether use residual connection in the root.
    root_dim : int
        Number of input channels in the root block.
    first_tree : bool, default False
        Is this tree stage the first stage in the net.
    input_level : bool, default True
        Is this tree unit the first unit in the stage.
    return_down : bool, default False
        Whether return downsample result.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 levels,
                 in_channels,
                 out_channels,
                 res_body_class,
                 strides,
                 root_residual,
                 root_dim=0,
                 first_tree=False,
                 input_level=True,
                 return_down=False,
                 data_format="channels_last",
                 **kwargs):
        super(DLATree, self).__init__(**kwargs)
        self.return_down = return_down
        self.add_down = (input_level and not first_tree)
        self.root_level = (levels == 1)

        if root_dim == 0:
            root_dim = 2 * out_channels
        if self.add_down:
            root_dim += in_channels

        if self.root_level:
            self.tree1 = DLAResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                body_class=res_body_class,
                return_down=True,
                data_format=data_format,
                name="tree1")
            self.tree2 = DLAResBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                body_class=res_body_class,
                return_down=False,
                data_format=data_format,
                name="tree2")
        else:
            self.tree1 = DLATree(
                levels=levels - 1,
                in_channels=in_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                strides=strides,
                root_residual=root_residual,
                root_dim=0,
                input_level=False,
                return_down=True,
                data_format=data_format,
                name="tree1")
            self.tree2 = DLATree(
                levels=levels - 1,
                in_channels=out_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                strides=1,
                root_residual=root_residual,
                root_dim=root_dim + out_channels,
                input_level=False,
                return_down=False,
                data_format=data_format,
                name="tree2")
        if self.root_level:
            self.root = DLARoot(
                in_channels=root_dim,
                out_channels=out_channels,
                residual=root_residual,
                data_format=data_format,
                name="root")

    def call(self, x, extra=None, training=None):
        extra = [] if extra is None else extra
        x1, down = self.tree1(x, training=training)
        if self.add_down:
            extra.append(down)
        if self.root_level:
            x2 = self.tree2(x1, training=training)
            x = self.root(x2, x1, extra, training=training)
        else:
            extra.append(x1)
            x = self.tree2(x1, extra, training=training)
        if self.return_down:
            return x, down
        else:
            return x


class DLAInitBlock(nn.Layer):
    """
    DLA specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 data_format="channels_last",
                 **kwargs):
        super(DLAInitBlock, self).__init__(**kwargs)
        mid_channels = out_channels // 2

        self.conv1 = conv7x7_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=2,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class DLA(tf.keras.Model):
    """
    DLA model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    levels : int
        Number of levels in each stage.
    channels : list of int
        Number of output channels for each stage.
    init_block_channels : int
        Number of output channels for the initial unit.
    res_body_class : nn.Module
        Residual block body class.
    residual_root : bool
        Whether use residual connection in the root blocks.
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
                 levels,
                 channels,
                 init_block_channels,
                 res_body_class,
                 residual_root,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(DLA, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(DLAInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels

        for i in range(len(levels)):
            levels_i = levels[i]
            out_channels = channels[i]
            first_tree = (i == 0)
            self.features.add(DLATree(
                levels=levels_i,
                in_channels=in_channels,
                out_channels=out_channels,
                res_body_class=res_body_class,
                strides=2,
                root_residual=residual_root,
                first_tree=first_tree,
                data_format=data_format,
                name="stage{}".format(i + 1)))
            in_channels = out_channels

        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = conv1x1(
            in_channels=in_channels,
            out_channels=classes,
            use_bias=True,
            data_format=data_format,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = self.output1(x)
        x = flatten(x, self.data_format)
        return x


def get_dla(levels,
            channels,
            res_body_class,
            residual_root=False,
            model_name=None,
            pretrained=False,
            root=os.path.join("~", ".tensorflow", "models"),
            **kwargs):
    """
    Create DLA model with specific parameters.

    Parameters:
    ----------
    levels : int
        Number of levels in each stage.
    channels : list of int
        Number of output channels for each stage.
    res_body_class : nn.Module
        Residual block body class.
    residual_root : bool, default False
        Whether use residual connection in the root blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels = 32

    net = DLA(
        levels=levels,
        channels=channels,
        init_block_channels=init_block_channels,
        res_body_class=res_body_class,
        residual_root=residual_root,
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


def dla34(**kwargs):
    """
    DLA-34 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 2, 1], channels=[64, 128, 256, 512], res_body_class=ResBlock, model_name="dla34",
                   **kwargs)


def dla46c(**kwargs):
    """
    DLA-46-C model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 2, 1], channels=[64, 64, 128, 256], res_body_class=DLABottleneck, model_name="dla46c",
                   **kwargs)


def dla46xc(**kwargs):
    """
    DLA-X-46-C model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 2, 1], channels=[64, 64, 128, 256], res_body_class=DLABottleneckX,
                   model_name="dla46xc", **kwargs)


def dla60(**kwargs):
    """
    DLA-60 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 3, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneck,
                   model_name="dla60", **kwargs)


def dla60x(**kwargs):
    """
    DLA-X-60 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 3, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneckX,
                   model_name="dla60x", **kwargs)


def dla60xc(**kwargs):
    """
    DLA-X-60-C model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 2, 3, 1], channels=[64, 64, 128, 256], res_body_class=DLABottleneckX,
                   model_name="dla60xc", **kwargs)


def dla102(**kwargs):
    """
    DLA-102 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 3, 4, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneck,
                   residual_root=True, model_name="dla102", **kwargs)


def dla102x(**kwargs):
    """
    DLA-X-102 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[1, 3, 4, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneckX,
                   residual_root=True, model_name="dla102x", **kwargs)


def dla102x2(**kwargs):
    """
    DLA-X2-102 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    class DLABottleneckX64(DLABottleneckX):
        def __init__(self, in_channels, out_channels, strides, **kwargs):
            super(DLABottleneckX64, self).__init__(in_channels, out_channels, strides, cardinality=64, **kwargs)

    return get_dla(levels=[1, 3, 4, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneckX64,
                   residual_root=True, model_name="dla102x2", **kwargs)


def dla169(**kwargs):
    """
    DLA-169 model from 'Deep Layer Aggregation,' https://arxiv.org/abs/1707.06484.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_dla(levels=[2, 3, 5, 1], channels=[128, 256, 512, 1024], res_body_class=DLABottleneck,
                   residual_root=True, model_name="dla169", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        dla34,
        dla46c,
        dla46xc,
        dla60,
        dla60x,
        dla60xc,
        dla102,
        dla102x,
        dla102x2,
        dla169,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dla34 or weight_count == 15742104)
        assert (model != dla46c or weight_count == 1301400)
        assert (model != dla46xc or weight_count == 1068440)
        assert (model != dla60 or weight_count == 22036632)
        assert (model != dla60x or weight_count == 17352344)
        assert (model != dla60xc or weight_count == 1319832)
        assert (model != dla102 or weight_count == 33268888)
        assert (model != dla102x or weight_count == 26309272)
        assert (model != dla102x2 or weight_count == 41282200)
        assert (model != dla169 or weight_count == 53389720)


if __name__ == "__main__":
    _test()
