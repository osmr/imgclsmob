"""
    BN-Inception for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,'
    https://arxiv.org/abs/1502.03167.
"""

__all__ = ['BNInception', 'bninception']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, conv7x7_block, MaxPool2d, AvgPool2d, Concurrent, SimpleSequential,\
    flatten, is_channels_first


class Inception3x3Branch(nn.Layer):
    """
    BN-Inception 3x3 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 strides=1,
                 use_bias=True,
                 use_bn=True,
                 data_format="channels_last",
                 **kwargs):
        super(Inception3x3Branch, self).__init__(**kwargs)
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            strides=strides,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        return x


class InceptionDouble3x3Branch(nn.Layer):
    """
    BN-Inception double 3x3 branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    strides : int or tuple/list of 2 int, default 1
        Strides of the second convolution.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 strides=1,
                 use_bias=True,
                 use_bn=True,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionDouble3x3Branch, self).__init__(**kwargs)
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv2")
        self.conv3 = conv3x3_block(
            in_channels=out_channels,
            out_channels=out_channels,
            strides=strides,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv3")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class InceptionPoolBranch(nn.Layer):
    """
    BN-Inception avg-pool branch block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    avg_pool : bool
        Whether use average pooling or max pooling.
    use_bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 avg_pool,
                 use_bias,
                 use_bn,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionPoolBranch, self).__init__(**kwargs)
        if avg_pool:
            self.pool = AvgPool2d(
                pool_size=3,
                strides=1,
                padding=1,
                ceil_mode=True,
                # count_include_pad=True,
                data_format=data_format,
                name="pool")
        else:
            self.pool = MaxPool2d(
                pool_size=3,
                strides=1,
                padding=1,
                ceil_mode=True,
                data_format=data_format,
                name="pool")
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv")

    def call(self, x, training=None):
        x = self.pool(x)
        x = self.conv(x, training=training)
        return x


class StemBlock(nn.Layer):
    """
    BN-Inception stem block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of intermediate channels.
    use_bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 use_bias,
                 use_bn,
                 data_format="channels_last",
                 **kwargs):
        super(StemBlock, self).__init__(**kwargs)
        self.conv1 = conv7x7_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv1")
        self.pool1 = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            ceil_mode=True,
            data_format=data_format,
            name="pool1")
        self.conv2 = Inception3x3Branch(
            in_channels=mid_channels,
            out_channels=out_channels,
            mid_channels=mid_channels,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="conv2")
        self.pool2 = MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            ceil_mode=True,
            data_format=data_format,
            name="pool2")

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.pool1(x)
        x = self.conv2(x, training=training)
        x = self.pool2(x)
        return x


class InceptionBlock(nn.Layer):
    """
    BN-Inception unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid1_channels_list : list of int
        Number of pre-middle channels for branches.
    mid2_channels_list : list of int
        Number of middle channels for branches.
    avg_pool : bool
        Whether use average pooling or max pooling.
    use_bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid1_channels_list,
                 mid2_channels_list,
                 avg_pool,
                 use_bias,
                 use_bn,
                 data_format="channels_last",
                 **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)
        assert (len(mid1_channels_list) == 2)
        assert (len(mid2_channels_list) == 4)

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(conv1x1_block(
            in_channels=in_channels,
            out_channels=mid2_channels_list[0],
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(Inception3x3Branch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[1],
            mid_channels=mid1_channels_list[0],
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(InceptionDouble3x3Branch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[2],
            mid_channels=mid1_channels_list[1],
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="branch3"))
        self.branches.children.append(InceptionPoolBranch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[3],
            avg_pool=avg_pool,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="branch4"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class ReductionBlock(nn.Layer):
    """
    BN-Inception reduction block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid1_channels_list : list of int
        Number of pre-middle channels for branches.
    mid2_channels_list : list of int
        Number of middle channels for branches.
    use_bias : bool
        Whether the convolution layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layers.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 mid1_channels_list,
                 mid2_channels_list,
                 use_bias,
                 use_bn,
                 data_format="channels_last",
                 **kwargs):
        super(ReductionBlock, self).__init__(**kwargs)
        assert (len(mid1_channels_list) == 2)
        assert (len(mid2_channels_list) == 4)

        self.branches = Concurrent(
            data_format=data_format,
            name="branches")
        self.branches.children.append(Inception3x3Branch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[1],
            mid_channels=mid1_channels_list[0],
            strides=2,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="branch1"))
        self.branches.children.append(InceptionDouble3x3Branch(
            in_channels=in_channels,
            out_channels=mid2_channels_list[2],
            mid_channels=mid1_channels_list[1],
            strides=2,
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="branch2"))
        self.branches.children.append(MaxPool2d(
            pool_size=3,
            strides=2,
            padding=0,
            ceil_mode=True,
            data_format=data_format,
            name="branch3"))

    def call(self, x, training=None):
        x = self.branches(x, training=training)
        return x


class BNInception(tf.keras.Model):
    """
    BN-Inception model from 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate
    Shift,' https://arxiv.org/abs/1502.03167.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels_list : list of int
        Number of output channels for the initial unit.
    mid1_channels_list : list of list of list of int
        Number of pre-middle channels for each unit.
    mid2_channels_list : list of list of list of int
        Number of middle channels for each unit.
    use_bias : bool, default True
        Whether the convolution layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layers.
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
                 init_block_channels_list,
                 mid1_channels_list,
                 mid2_channels_list,
                 use_bias=True,
                 use_bn=True,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(BNInception, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format

        self.features = SimpleSequential(name="features")
        self.features.add(StemBlock(
            in_channels=in_channels,
            out_channels=init_block_channels_list[1],
            mid_channels=init_block_channels_list[0],
            use_bias=use_bias,
            use_bn=use_bn,
            data_format=data_format,
            name="init_block"))
        in_channels = init_block_channels_list[-1]
        for i, channels_per_stage in enumerate(channels):
            mid1_channels_list_i = mid1_channels_list[i]
            mid2_channels_list_i = mid2_channels_list[i]
            stage = SimpleSequential(name="stage{}".format(i + 1))
            for j, out_channels in enumerate(channels_per_stage):
                if (j == 0) and (i != 0):
                    stage.add(ReductionBlock(
                        in_channels=in_channels,
                        mid1_channels_list=mid1_channels_list_i[j],
                        mid2_channels_list=mid2_channels_list_i[j],
                        use_bias=use_bias,
                        use_bn=use_bn,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                else:
                    avg_pool = (i != len(channels) - 1) or (j != len(channels_per_stage) - 1)
                    stage.add(InceptionBlock(
                        in_channels=in_channels,
                        mid1_channels_list=mid1_channels_list_i[j],
                        mid2_channels_list=mid2_channels_list_i[j],
                        avg_pool=avg_pool,
                        use_bias=use_bias,
                        use_bn=use_bn,
                        data_format=data_format,
                        name="unit{}".format(j + 1)))
                in_channels = out_channels
            self.features.add(stage)
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=in_channels,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_bninception(model_name=None,
                    pretrained=False,
                    root=os.path.join("~", ".tensorflow", "models"),
                    **kwargs):
    """
    Create BN-Inception model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    init_block_channels_list = [64, 192]
    channels = [[256, 320], [576, 576, 576, 608, 608], [1056, 1024, 1024]]
    mid1_channels_list = [
        [[64, 64],
         [64, 64]],
        [[128, 64],  # 3c
         [64, 96],  # 4a
         [96, 96],  # 4a
         [128, 128],  # 4c
         [128, 160]],  # 4d
        [[128, 192],  # 4e
         [192, 160],  # 5a
         [192, 192]],
    ]
    mid2_channels_list = [
        [[64, 64, 96, 32],
         [64, 96, 96, 64]],
        [[0, 160, 96, 0],  # 3c
         [224, 96, 128, 128],  # 4a
         [192, 128, 128, 128],  # 4b
         [160, 160, 160, 128],  # 4c
         [96, 192, 192, 128]],  # 4d
        [[0, 192, 256, 0],  # 4e
         [352, 320, 224, 128],  # 5a
         [352, 320, 224, 128]],
    ]

    net = BNInception(
        channels=channels,
        init_block_channels_list=init_block_channels_list,
        mid1_channels_list=mid1_channels_list,
        mid2_channels_list=mid2_channels_list,
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


def bninception(**kwargs):
    """
    BN-Inception model from 'Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate
    Shift,' https://arxiv.org/abs/1502.03167.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_bninception(model_name="bninception", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        bninception,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != bninception or weight_count == 11295240)


if __name__ == "__main__":
    _test()
