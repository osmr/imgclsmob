"""
    ProxylessNAS for ImageNet-1K, implemented in Chainer.
    Original paper: 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.
"""

__all__ = ['ProxylessNAS', 'proxylessnas_cpu', 'proxylessnas_gpu', 'proxylessnas_mobile', 'proxylessnas_mobile14',
           'get_proxylessnas']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import ConvBlock, conv1x1_block, conv3x3_block, SimpleSequential


class ProxylessBlock(Chain):
    """
    ProxylessNAS block for residual path in ProxylessNAS unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int
        Convolution window size.
    stride : int
        Stride of the convolution.
    bn_eps : float
        Small float added to variance in Batch norm.
    expansion : int
        Expansion ratio.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 bn_eps,
                 expansion):
        super(ProxylessBlock, self).__init__()
        self.use_bc = (expansion > 1)
        mid_channels = in_channels * expansion

        with self.init_scope():
            if self.use_bc:
                self.bc_conv = conv1x1_block(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    bn_eps=bn_eps,
                    activation="relu6")

            pad = (ksize - 1) // 2
            self.dw_conv = ConvBlock(
                in_channels=mid_channels,
                out_channels=mid_channels,
                ksize=ksize,
                stride=stride,
                pad=pad,
                groups=mid_channels,
                bn_eps=bn_eps,
                activation="relu6")
            self.pw_conv = conv1x1_block(
                in_channels=mid_channels,
                out_channels=out_channels,
                bn_eps=bn_eps,
                activation=None)

    def __call__(self, x):
        if self.use_bc:
            x = self.bc_conv(x)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class ProxylessUnit(Chain):
    """
    ProxylessNAS unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ksize : int
        Convolution window size for body block.
    stride : int
        Stride of the convolution.
    bn_eps : float
        Small float added to variance in Batch norm.
    expansion : int
        Expansion ratio for body block.
    residual : bool
        Whether to use residual branch.
    shortcut : bool
        Whether to use identity branch.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 bn_eps,
                 expansion,
                 residual,
                 shortcut):
        super(ProxylessUnit, self).__init__()
        assert (residual or shortcut)
        self.residual = residual
        self.shortcut = shortcut

        with self.init_scope():
            if self.residual:
                self.body = ProxylessBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    ksize=ksize,
                    stride=stride,
                    bn_eps=bn_eps,
                    expansion=expansion)

    def __call__(self, x):
        if not self.residual:
            return x
        if not self.shortcut:
            return self.body(x)
        identity = x
        x = self.body(x)
        x = identity + x
        return x


class ProxylessNAS(Chain):
    """
    ProxylessNAS model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    final_block_channels : int
        Number of output channels for the final unit.
    residuals : list of list of int
        Whether to use residual branch in units.
    shortcuts : list of list of int
        Whether to use identity branch in units.
    ksizes : list of list of int
        Convolution window size for each units.
    expansions : list of list of int
        Expansion ratio for each units.
    bn_eps : float, default 1e-3
        Small float added to variance in Batch norm.
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
                 final_block_channels,
                 residuals,
                 shortcuts,
                 ksizes,
                 expansions,
                 bn_eps=1e-3,
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(ProxylessNAS, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                setattr(self.features, "init_block", conv3x3_block(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    stride=2,
                    bn_eps=bn_eps,
                    activation="relu6"))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels):
                    stage = SimpleSequential()
                    residuals_per_stage = residuals[i]
                    shortcuts_per_stage = shortcuts[i]
                    ksizes_per_stage = ksizes[i]
                    expansions_per_stage = expansions[i]
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            residual = (residuals_per_stage[j] == 1)
                            shortcut = (shortcuts_per_stage[j] == 1)
                            ksize = ksizes_per_stage[j]
                            expansion = expansions_per_stage[j]
                            stride = 2 if (j == 0) and (i != 0) else 1
                            setattr(stage, "unit{}".format(j + 1), ProxylessUnit(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                ksize=ksize,
                                stride=stride,
                                bn_eps=bn_eps,
                                expansion=expansion,
                                residual=residual,
                                shortcut=shortcut))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
                setattr(self.features, "final_block", conv1x1_block(
                    in_channels=in_channels,
                    out_channels=final_block_channels,
                    bn_eps=bn_eps,
                    activation="relu6"))
                in_channels = final_block_channels
                setattr(self.features, "final_pool", partial(
                    F.average_pooling_2d,
                    ksize=7,
                    stride=1))

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


def get_proxylessnas(version,
                     model_name=None,
                     pretrained=False,
                     root=os.path.join("~", ".chainer", "models"),
                     **kwargs):
    """
    Create ProxylessNAS model with specific parameters.

    Parameters:
    ----------
    version : str
        Version of ProxylessNAS ('cpu', 'gpu', 'mobile' or 'mobile14').
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """

    if version == "cpu":
        residuals = [[1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [32, 32, 32, 32], [48, 48, 48, 48], [88, 88, 88, 88, 104, 104, 104, 104],
                    [216, 216, 216, 216, 360]]
        kernel_sizes = [[3], [3, 3, 3, 3], [3, 3, 3, 5], [3, 3, 3, 3, 5, 3, 3, 3], [5, 5, 5, 3, 5]]
        expansions = [[1], [6, 3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 3, 3, 3, 6]]
        init_block_channels = 40
        final_block_channels = 1432
    elif version == "gpu":
        residuals = [[1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [32, 32, 32, 32], [56, 56, 56, 56], [112, 112, 112, 112, 128, 128, 128, 128],
                    [256, 256, 256, 256, 432]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 3, 3], [7, 5, 5, 5, 5, 3, 3, 5], [7, 7, 7, 5, 7]]
        expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 6, 6, 6]]
        init_block_channels = 40
        final_block_channels = 1728
    elif version == "mobile":
        residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[16], [32, 32, 32, 32], [40, 40, 40, 40], [80, 80, 80, 80, 96, 96, 96, 96],
                    [192, 192, 192, 192, 320]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
        expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
        init_block_channels = 32
        final_block_channels = 1280
    elif version == "mobile14":
        residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [40, 40, 40, 40], [56, 56, 56, 56], [112, 112, 112, 112, 136, 136, 136, 136],
                    [256, 256, 256, 256, 448]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
        expansions = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
        init_block_channels = 48
        final_block_channels = 1792
    else:
        raise ValueError("Unsupported ProxylessNAS version: {}".format(version))

    shortcuts = [[0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0]]

    net = ProxylessNAS(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        residuals=residuals,
        shortcuts=shortcuts,
        ksizes=kernel_sizes,
        expansions=expansions,
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


def proxylessnas_cpu(**kwargs):
    """
    ProxylessNAS (CPU) model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(version="cpu", model_name="proxylessnas_cpu", **kwargs)


def proxylessnas_gpu(**kwargs):
    """
    ProxylessNAS (GPU) model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(version="gpu", model_name="proxylessnas_gpu", **kwargs)


def proxylessnas_mobile(**kwargs):
    """
    ProxylessNAS (Mobile) model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(version="mobile", model_name="proxylessnas_mobile", **kwargs)


def proxylessnas_mobile14(**kwargs):
    """
    ProxylessNAS (Mobile-14) model from 'ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,'
    https://arxiv.org/abs/1812.00332.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_proxylessnas(version="mobile14", model_name="proxylessnas_mobile14", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        proxylessnas_cpu,
        proxylessnas_gpu,
        proxylessnas_mobile,
        proxylessnas_mobile14,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != proxylessnas_cpu or weight_count == 4361648)
        assert (model != proxylessnas_gpu or weight_count == 7119848)
        assert (model != proxylessnas_mobile or weight_count == 4080512)
        assert (model != proxylessnas_mobile14 or weight_count == 6857568)

        x = np.zeros((14, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (14, 1000))


if __name__ == "__main__":
    _test()
