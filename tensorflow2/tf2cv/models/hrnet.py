"""
    HRNet for ImageNet-1K, implemented in TensorFlow.
    Original paper: 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.
"""

__all__ = ['HRNet', 'hrnet_w18_small_v1', 'hrnet_w18_small_v2', 'hrnetv2_w18', 'hrnetv2_w30', 'hrnetv2_w32',
           'hrnetv2_w40', 'hrnetv2_w44', 'hrnetv2_w48', 'hrnetv2_w64']

import os
import tensorflow as tf
import tensorflow.keras.layers as nn
from .common import conv1x1_block, conv3x3_block, Identity, SimpleSequential, flatten, is_channels_first
from .resnet import ResUnit


class UpSamplingBlock(nn.Layer):
    """
    HFNet specific upsampling block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    scale_factor : int
        Multiplier for spatial size.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor,
                 data_format="channels_last",
                 **kwargs):
        super(UpSamplingBlock, self).__init__(**kwargs)
        self.scale_factor = scale_factor

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            strides=1,
            activation=None,
            data_format=data_format,
            name="conv")
        self.upsample = nn.UpSampling2D(
            size=scale_factor,
            data_format=data_format,
            interpolation="nearest",
            name="upsample")

    def call(self, x, training=None):
        x = self.conv(x, training=training)
        x = self.upsample(x)
        return x


class HRBlock(nn.Layer):
    """
    HFNet block.

    Parameters:
    ----------
    in_channels_list : list of int
        Number of input channels.
    out_channels_list : list of int
        Number of output channels.
    num_branches : int
        Number of branches.
    num_subblocks : list of int
        Number of subblock.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 num_branches,
                 num_subblocks,
                 data_format="channels_last",
                 **kwargs):
        super(HRBlock, self).__init__(**kwargs)
        self.in_channels_list = in_channels_list
        self.num_branches = num_branches

        self.branches = SimpleSequential(name="branches")
        for i in range(num_branches):
            layers = SimpleSequential(name="branches/branch{}".format(i + 1))
            in_channels_i = self.in_channels_list[i]
            out_channels_i = out_channels_list[i]
            for j in range(num_subblocks[i]):
                layers.add(ResUnit(
                    in_channels=in_channels_i,
                    out_channels=out_channels_i,
                    strides=1,
                    bottleneck=False,
                    data_format=data_format,
                    name="unit{}".format(j + 1)))
                in_channels_i = out_channels_i
            self.in_channels_list[i] = out_channels_i
            self.branches.add(layers)

        if num_branches > 1:
            self.fuse_layers = SimpleSequential(name="fuse_layers")
            for i in range(num_branches):
                fuse_layer_name = "fuse_layers/fuse_layer{}".format(i + 1)
                fuse_layer = SimpleSequential(name=fuse_layer_name)
                for j in range(num_branches):
                    if j > i:
                        fuse_layer.add(UpSamplingBlock(
                            in_channels=in_channels_list[j],
                            out_channels=in_channels_list[i],
                            scale_factor=2 ** (j - i),
                            data_format=data_format,
                            name=fuse_layer_name + "/block{}".format(j + 1)))
                    elif j == i:
                        fuse_layer.add(Identity(name=fuse_layer_name + "/block{}".format(j + 1)))
                    else:
                        conv3x3_seq_name = fuse_layer_name + "/block{}_conv3x3_seq".format(j + 1)
                        conv3x3_seq = SimpleSequential(name=conv3x3_seq_name)
                        for k in range(i - j):
                            if k == i - j - 1:
                                conv3x3_seq.add(conv3x3_block(
                                    in_channels=in_channels_list[j],
                                    out_channels=in_channels_list[i],
                                    strides=2,
                                    activation=None,
                                    data_format=data_format,
                                    name="subblock{}".format(k + 1)))
                            else:
                                conv3x3_seq.add(conv3x3_block(
                                    in_channels=in_channels_list[j],
                                    out_channels=in_channels_list[j],
                                    strides=2,
                                    data_format=data_format,
                                    name="subblock{}".format(k + 1)))
                        fuse_layer.add(conv3x3_seq)
                self.fuse_layers.add(fuse_layer)
            self.activ = nn.ReLU()

    def call(self, x, training=None):
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i], training=training)

        if self.num_branches == 1:
            return x

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0], training=training)
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j], training=training)
            x_fuse.append(self.activ(y))

        return x_fuse


class HRStage(nn.Layer):
    """
    HRNet stage block.

    Parameters:
    ----------
    in_channels_list : list of int
        Number of output channels from the previous layer.
    out_channels_list : list of int
        Number of output channels in the current layer.
    num_modules : int
        Number of modules.
    num_branches : int
        Number of branches.
    num_subblocks : list of int
        Number of subblocks.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 num_modules,
                 num_branches,
                 num_subblocks,
                 data_format="channels_last",
                 **kwargs):
        super(HRStage, self).__init__(**kwargs)
        self.branches = num_branches
        self.in_channels_list = out_channels_list
        in_branches = len(in_channels_list)
        out_branches = len(out_channels_list)

        self.transition = SimpleSequential(name="transition")
        for i in range(out_branches):
            if i < in_branches:
                if out_channels_list[i] != in_channels_list[i]:
                    self.transition.add(conv3x3_block(
                        in_channels=in_channels_list[i],
                        out_channels=out_channels_list[i],
                        strides=1,
                        data_format=data_format,
                        name="transition/block{}".format(i + 1)))
                else:
                    self.transition.add(Identity(name="transition/block{}".format(i + 1)))
            else:
                conv3x3_seq = SimpleSequential(name="transition/conv3x3_seq{}".format(i + 1))
                for j in range(i + 1 - in_branches):
                    in_channels_i = in_channels_list[-1]
                    out_channels_i = out_channels_list[i] if j == i - in_branches else in_channels_i
                    conv3x3_seq.add(conv3x3_block(
                        in_channels=in_channels_i,
                        out_channels=out_channels_i,
                        strides=2,
                        data_format=data_format,
                        name="subblock{}".format(j + 1)))
                self.transition.add(conv3x3_seq)

        self.layers = SimpleSequential(name="layers")
        for i in range(num_modules):
            self.layers.add(HRBlock(
                in_channels_list=self.in_channels_list,
                out_channels_list=out_channels_list,
                num_branches=num_branches,
                num_subblocks=num_subblocks,
                data_format=data_format,
                name="block{}".format(i + 1)))
            self.in_channels_list = list(self.layers[-1].in_channels_list)

    def call(self, x, training=None):
        x_list = []
        for j in range(self.branches):
            if not isinstance(self.transition[j], Identity):
                x_list.append(self.transition[j](x[-1] if type(x) in (list, tuple) else x, training=training))
            else:
                x_list_j = x[j] if type(x) in (list, tuple) else x
                x_list.append(x_list_j)
        y_list = self.layers(x_list, training=training)
        return y_list


class HRInitBlock(nn.Layer):
    """
    HRNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    mid_channels : int
        Number of middle channels.
    num_subblocks : int
        Number of subblocks.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 num_subblocks,
                 data_format="channels_last",
                 **kwargs):
        super(HRInitBlock, self).__init__(**kwargs)
        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="conv1")
        self.conv2 = conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            strides=2,
            data_format=data_format,
            name="conv2")
        in_channels = mid_channels
        self.subblocks = SimpleSequential(name="subblocks")
        for i in range(num_subblocks):
            self.subblocks.add(ResUnit(
                in_channels=in_channels,
                out_channels=out_channels,
                strides=1,
                bottleneck=True,
                data_format=data_format,
                name="block{}".format(i + 1)))
            in_channels = out_channels

    def call(self, x, training=None):
        x = self.conv1(x, training=training)
        x = self.conv2(x, training=training)
        x = self.subblocks(x, training=training)
        return x


class HRFinalBlock(nn.Layer):
    """
    HRNet specific final block.

    Parameters:
    ----------
    in_channels_list : list of int
        Number of input channels per stage.
    out_channels_list : list of int
        Number of output channels per stage.
    data_format : str, default 'channels_last'
        The ordering of the dimensions in tensors.
    """
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 data_format="channels_last",
                 **kwargs):
        super(HRFinalBlock, self).__init__(**kwargs)
        self.inc_blocks = SimpleSequential(name="inc_blocks")
        for i, in_channels_i in enumerate(in_channels_list):
            self.inc_blocks.add(ResUnit(
                in_channels=in_channels_i,
                out_channels=out_channels_list[i],
                strides=1,
                bottleneck=True,
                data_format=data_format,
                name="inc_blocks/block{}".format(i + 1)))
        self.down_blocks = SimpleSequential(name="down_blocks")
        for i in range(len(in_channels_list) - 1):
            self.down_blocks.add(conv3x3_block(
                in_channels=out_channels_list[i],
                out_channels=out_channels_list[i + 1],
                strides=2,
                use_bias=True,
                data_format=data_format,
                name="down_blocks/block{}".format(i + 1)))
        self.final_layer = conv1x1_block(
            in_channels=1024,
            out_channels=2048,
            strides=1,
            use_bias=True,
            data_format=data_format,
            name="final_layer")

    def call(self, x, training=None):
        y = self.inc_blocks[0](x[0], training=training)
        for i in range(len(self.down_blocks)):
            y = self.inc_blocks[i + 1](x[i + 1], training=training) + self.down_blocks[i](y, training=training)
        y = self.final_layer(y, training=training)
        return y


class HRNet(tf.keras.Model):
    """
    HRNet model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    init_num_subblocks : int
        Number of subblocks in the initial unit.
    num_modules : int
        Number of modules per stage.
    num_subblocks : list of int
        Number of subblocks per stage.
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
                 init_num_subblocks,
                 num_modules,
                 num_subblocks,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000,
                 data_format="channels_last",
                 **kwargs):
        super(HRNet, self).__init__(**kwargs)
        self.in_size = in_size
        self.classes = classes
        self.data_format = data_format
        self.branches = [2, 3, 4]

        self.features = SimpleSequential(name="features")
        self.features.add(HRInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            mid_channels=64,
            num_subblocks=init_num_subblocks,
            data_format=data_format,
            name="init_block"))
        in_channels_list = [init_block_channels]
        for i in range(len(self.branches)):
            self.features.add(HRStage(
                in_channels_list=in_channels_list,
                out_channels_list=channels[i],
                num_modules=num_modules[i],
                num_branches=self.branches[i],
                num_subblocks=num_subblocks[i],
                data_format=data_format,
                name="stage{}".format(i + 1)))
            in_channels_list = self.features[-1].in_channels_list
        self.features.add(HRFinalBlock(
            in_channels_list=in_channels_list,
            out_channels_list=[128, 256, 512, 1024],
            data_format=data_format,
            name="final_block"))
        self.features.add(nn.AveragePooling2D(
            pool_size=7,
            strides=1,
            data_format=data_format,
            name="final_pool"))

        self.output1 = nn.Dense(
            units=classes,
            input_dim=2048,
            name="output1")

    def call(self, x, training=None):
        x = self.features(x, training=training)
        x = flatten(x, self.data_format)
        x = self.output1(x)
        return x


def get_hrnet(version,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".tensorflow", "models"),
              **kwargs):
    """
    Create HRNet model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of MobileNetV3 ('s' or 'm').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    if version == "w18s1":
        init_block_channels = 128
        init_num_subblocks = 1
        channels = [[16, 32], [16, 32, 64], [16, 32, 64, 128]]
        num_modules = [1, 1, 1]
    elif version == "w18s2":
        init_block_channels = 256
        init_num_subblocks = 2
        channels = [[18, 36], [18, 36, 72], [18, 36, 72, 144]]
        num_modules = [1, 3, 2]
    elif version == "w18":
        init_block_channels = 256
        init_num_subblocks = 4
        channels = [[18, 36], [18, 36, 72], [18, 36, 72, 144]]
        num_modules = [1, 4, 3]
    elif version == "w30":
        init_block_channels = 256
        init_num_subblocks = 4
        channels = [[30, 60], [30, 60, 120], [30, 60, 120, 240]]
        num_modules = [1, 4, 3]
    elif version == "w32":
        init_block_channels = 256
        init_num_subblocks = 4
        channels = [[32, 64], [32, 64, 128], [32, 64, 128, 256]]
        num_modules = [1, 4, 3]
    elif version == "w40":
        init_block_channels = 256
        init_num_subblocks = 4
        channels = [[40, 80], [40, 80, 160], [40, 80, 160, 320]]
        num_modules = [1, 4, 3]
    elif version == "w44":
        init_block_channels = 256
        init_num_subblocks = 4
        channels = [[44, 88], [44, 88, 176], [44, 88, 176, 352]]
        num_modules = [1, 4, 3]
    elif version == "w48":
        init_block_channels = 256
        init_num_subblocks = 4
        channels = [[48, 96], [48, 96, 192], [48, 96, 192, 384]]
        num_modules = [1, 4, 3]
    elif version == "w64":
        init_block_channels = 256
        init_num_subblocks = 4
        channels = [[64, 128], [64, 128, 256], [64, 128, 256, 512]]
        num_modules = [1, 4, 3]
    else:
        raise ValueError("Unsupported HRNet version {}".format(version))

    num_subblocks = [[max(2, init_num_subblocks)] * len(ci) for ci in channels]

    net = HRNet(
        channels=channels,
        init_block_channels=init_block_channels,
        init_num_subblocks=init_num_subblocks,
        num_modules=num_modules,
        num_subblocks=num_subblocks,
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


def hrnet_w18_small_v1(**kwargs):
    """
    HRNet-W18 Small V1 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w18s1", model_name="hrnet_w18_small_v1", **kwargs)


def hrnet_w18_small_v2(**kwargs):
    """
    HRNet-W18 Small V2 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w18s2", model_name="hrnet_w18_small_v2", **kwargs)


def hrnetv2_w18(**kwargs):
    """
    HRNetV2-W18 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w18", model_name="hrnetv2_w18", **kwargs)


def hrnetv2_w30(**kwargs):
    """
    HRNetV2-W30 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w30", model_name="hrnetv2_w30", **kwargs)


def hrnetv2_w32(**kwargs):
    """
    HRNetV2-W32 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w32", model_name="hrnetv2_w32", **kwargs)


def hrnetv2_w40(**kwargs):
    """
    HRNetV2-W40 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w40", model_name="hrnetv2_w40", **kwargs)


def hrnetv2_w44(**kwargs):
    """
    HRNetV2-W44 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w44", model_name="hrnetv2_w44", **kwargs)


def hrnetv2_w48(**kwargs):
    """
    HRNetV2-W48 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w48", model_name="hrnetv2_w48", **kwargs)


def hrnetv2_w64(**kwargs):
    """
    HRNetV2-W64 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.tensorflow/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w64", model_name="hrnetv2_w64", **kwargs)


def _test():
    import numpy as np
    import tensorflow.keras.backend as K

    data_format = "channels_last"
    pretrained = False

    models = [
        hrnet_w18_small_v1,
        hrnet_w18_small_v2,
        hrnetv2_w18,
        hrnetv2_w30,
        hrnetv2_w32,
        hrnetv2_w40,
        hrnetv2_w44,
        hrnetv2_w48,
        hrnetv2_w64,
    ]

    for model in models:

        net = model(pretrained=pretrained, data_format=data_format)

        batch = 14
        x = tf.random.normal((batch, 3, 224, 224) if is_channels_first(data_format) else (batch, 224, 224, 3))
        y = net(x)
        assert (tuple(y.shape.as_list()) == (batch, 1000))

        weight_count = sum([np.prod(K.get_value(w).shape) for w in net.trainable_weights])
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != hrnet_w18_small_v1 or weight_count == 13187464)
        assert (model != hrnet_w18_small_v2 or weight_count == 15597464)
        assert (model != hrnetv2_w18 or weight_count == 21299004)
        assert (model != hrnetv2_w30 or weight_count == 37712220)
        assert (model != hrnetv2_w32 or weight_count == 41232680)
        assert (model != hrnetv2_w40 or weight_count == 57557160)
        assert (model != hrnetv2_w44 or weight_count == 67064984)
        assert (model != hrnetv2_w48 or weight_count == 77469864)
        assert (model != hrnetv2_w64 or weight_count == 128059944)


if __name__ == "__main__":
    _test()
