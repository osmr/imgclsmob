"""
    HRNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.
"""

__all__ = ['hrnet_w18_small_v1', 'hrnet_w18_small_v2', 'hrnetv2_w18', 'hrnetv2_w30', 'hrnetv2_w32', 'hrnetv2_w40',
           'hrnetv2_w44', 'hrnetv2_w48', 'hrnetv2_w64']

import os
from inspect import isfunction
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv1x1_block, conv3x3_block, SimpleSequential
from .resnet import ResUnit


class UpSamplingBlock(Chain):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor):
        super(UpSamplingBlock, self).__init__()
        self.scale_factor = scale_factor

        with self.init_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=1,
                activation=None)

    def __call__(self, x):
        x = self.conv(x)
        return F.unpooling_2d(
            x=x,
            ksize=self.scale_factor,
            cover_all=False)


class HRBlock(Chain):
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
    """
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 num_branches,
                 num_subblocks):
        super(HRBlock, self).__init__()
        self.in_channels_list = in_channels_list
        self.num_branches = num_branches

        with self.init_scope():
            self.branches = SimpleSequential()
            with self.branches.init_scope():
                for i in range(num_branches):
                    layers = SimpleSequential()
                    with layers.init_scope():
                        in_channels_i = self.in_channels_list[i]
                        out_channels_i = out_channels_list[i]
                        for j in range(num_subblocks[i]):
                            setattr(layers, "unit{}".format(j + 1), ResUnit(
                                in_channels=in_channels_i,
                                out_channels=out_channels_i,
                                stride=1,
                                bottleneck=False))
                            in_channels_i = out_channels_i
                        self.in_channels_list[i] = out_channels_i
                        setattr(self.branches, "branch{}".format(i + 1), layers)

            if num_branches > 1:
                self.fuse_layers = SimpleSequential()
                with self.fuse_layers.init_scope():
                    for i in range(num_branches):
                        fuse_layer = SimpleSequential()
                        with fuse_layer.init_scope():
                            for j in range(num_branches):
                                if j > i:
                                    setattr(fuse_layer, "block{}".format(j + 1), UpSamplingBlock(
                                        in_channels=in_channels_list[j],
                                        out_channels=in_channels_list[i],
                                        scale_factor=2 ** (j - i)))
                                elif j == i:
                                    setattr(fuse_layer, "block{}".format(j + 1), F.identity)
                                else:
                                    conv3x3_seq = SimpleSequential()
                                    with conv3x3_seq.init_scope():
                                        for k in range(i - j):
                                            if k == i - j - 1:
                                                setattr(conv3x3_seq, "subblock{}".format(k + 1), conv3x3_block(
                                                    in_channels=in_channels_list[j],
                                                    out_channels=in_channels_list[i],
                                                    stride=2,
                                                    activation=None))
                                            else:
                                                setattr(conv3x3_seq, "subblock{}".format(k + 1), conv3x3_block(
                                                    in_channels=in_channels_list[j],
                                                    out_channels=in_channels_list[j],
                                                    stride=2))
                                    setattr(fuse_layer, "block{}".format(j + 1), conv3x3_seq)
                        setattr(self.fuse_layers, "layer{}".format(i + 1), fuse_layer)
                self.activ = F.relu

    def __call__(self, x):
        for i in range(self.num_branches):
            x[i] = self.branches.el(i)(x[i])

        if self.num_branches == 1:
            return x

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers.el(i).el(0)(x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers.el(i).el(j)(x[j])
            x_fuse.append(self.activ(y))

        return x_fuse


class HRStage(Chain):
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
    """
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 num_modules,
                 num_branches,
                 num_subblocks):
        super(HRStage, self).__init__()
        self.branches = num_branches
        self.in_channels_list = out_channels_list
        in_branches = len(in_channels_list)
        out_branches = len(out_channels_list)

        with self.init_scope():
            self.transition = SimpleSequential()
            with self.transition.init_scope():
                for i in range(out_branches):
                    if i < in_branches:
                        if out_channels_list[i] != in_channels_list[i]:
                            setattr(self.transition, "block{}".format(i + 1), conv3x3_block(
                                in_channels=in_channels_list[i],
                                out_channels=out_channels_list[i],
                                stride=1))
                        else:
                            setattr(self.transition, "block{}".format(i + 1), F.identity)
                    else:
                        conv3x3_seq = SimpleSequential()
                        with conv3x3_seq.init_scope():
                            for j in range(i + 1 - in_branches):
                                in_channels_i = in_channels_list[-1]
                                out_channels_i = out_channels_list[i] if j == i - in_branches else in_channels_i
                                setattr(conv3x3_seq, "subblock{}".format(j + 1), conv3x3_block(
                                    in_channels=in_channels_i,
                                    out_channels=out_channels_i,
                                    stride=2))
                        setattr(self.transition, "block{}".format(i + 1), conv3x3_seq)

            self.layers = SimpleSequential()
            with self.layers.init_scope():
                for i in range(num_modules):
                    block = HRBlock(
                        in_channels_list=self.in_channels_list,
                        out_channels_list=out_channels_list,
                        num_branches=num_branches,
                        num_subblocks=num_subblocks)
                    setattr(self.layers, "block{}".format(i + 1), block)
                    self.in_channels_list = block.in_channels_list

    def __call__(self, x):
        x_list = []
        for j in range(self.branches):
            if not isfunction(self.transition.el(j)):
                x_list.append(self.transition.el(j)(x[-1] if type(x) is list else x))
            else:
                x_list_j = x[j] if type(x) is list else x
                x_list.append(x_list_j)
        y_list = self.layers(x_list)
        return y_list


class HRInitBlock(Chain):
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
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 num_subblocks):
        super(HRInitBlock, self).__init__()
        with self.init_scope():
            self.conv1 = conv3x3_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                stride=2)
            self.conv2 = conv3x3_block(
                in_channels=mid_channels,
                out_channels=mid_channels,
                stride=2)
            in_channels = mid_channels
            self.subblocks = SimpleSequential()
            with self.subblocks.init_scope():
                for i in range(num_subblocks):
                    setattr(self.subblocks, "block{}".format(i + 1), ResUnit(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=1,
                        bottleneck=True))
                    in_channels = out_channels

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.subblocks(x)
        return x


class HRFinalBlock(Chain):
    """
    HRNet specific final block.

    Parameters:
    ----------
    in_channels_list : list of int
        Number of input channels per stage.
    out_channels_list : list of int
        Number of output channels per stage.
    """
    def __init__(self,
                 in_channels_list,
                 out_channels_list):
        super(HRFinalBlock, self).__init__()
        with self.init_scope():
            self.inc_blocks = SimpleSequential()
            with self.inc_blocks.init_scope():
                for i, in_channels_i in enumerate(in_channels_list):
                    setattr(self.inc_blocks, "block{}".format(i + 1), ResUnit(
                        in_channels=in_channels_i,
                        out_channels=out_channels_list[i],
                        stride=1,
                        bottleneck=True))
            self.down_blocks = SimpleSequential()
            with self.down_blocks.init_scope():
                for i in range(len(in_channels_list) - 1):
                    setattr(self.down_blocks, "block{}".format(i + 1), conv3x3_block(
                        in_channels=out_channels_list[i],
                        out_channels=out_channels_list[i + 1],
                        stride=2,
                        use_bias=True))
            self.final_layer = conv1x1_block(
                in_channels=1024,
                out_channels=2048,
                stride=1,
                use_bias=True)

    def __call__(self, x):
        y = self.inc_blocks.el(0)(x[0])
        for i in range(len(self.down_blocks)):
            y = self.inc_blocks.el(i + 1)(x[i + 1]) + self.down_blocks.el(i)(y)
        y = self.final_layer(y)
        return y


class HRNet(Chain):
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
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 init_num_subblocks,
                 num_modules,
                 num_subblocks,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(HRNet, self).__init__()
        self.in_size = in_size
        self.classes = classes
        self.branches = [2, 3, 4]

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", HRInitBlock(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    mid_channels=64,
                    num_subblocks=init_num_subblocks))
                in_channels_list = [init_block_channels]
                for i in range(len(self.branches)):
                    stage = HRStage(
                        in_channels_list=in_channels_list,
                        out_channels_list=channels[i],
                        num_modules=num_modules[i],
                        num_branches=self.branches[i],
                        num_subblocks=num_subblocks[i])
                    setattr(self.features, "stage{}".format(i + 1), stage)
                    in_channels_list = stage.in_channels_list
                setattr(self.features, "final_block", HRFinalBlock(
                    in_channels_list=in_channels_list,
                    out_channels_list=[128, 256, 512, 1024]))
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

            in_channels = 2048
            self.output = SimpleSequential()
            with self.output.init_scope():
                setattr(self.output, "flatten", partial(
                    F.reshape,
                    shape=(-1, in_channels)))
                setattr(self.output, "fc", L.Linear(
                    in_size=in_channels,
                    out_size=classes))

    def __call__(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_hrnet(version,
              model_name=None,
              pretrained=False,
              root=os.path.join("~", ".chainer", "models"),
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
    root : str, default '~/.chainer/models'
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
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def hrnet_w18_small_v1(**kwargs):
    """
    HRNet-W18 Small V1 model from 'Deep High-Resolution Representation Learning for Visual Recognition,'
    https://arxiv.org/abs/1908.07919.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
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
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_hrnet(version="w64", model_name="hrnetv2_w64", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

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

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
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

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
