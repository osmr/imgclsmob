"""
    MobileNet for ImageNet-1K, implemented in Chainer.
    Original paper: 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.
"""

__all__ = ['MobileNet', 'mobilenet_w1', 'mobilenet_w3d4', 'mobilenet_wd2', 'mobilenet_wd4', 'get_mobilenet']

import os
import chainer.functions as F
import chainer.links as L
from chainer import Chain
from functools import partial
from chainer.serializers import load_npz
from .common import conv3x3_block, dwsconv3x3_block, SimpleSequential


class MobileNet(Chain):
    """
    MobileNet model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    first_stage_stride : bool
        Whether stride is used at the first stage.
    dw_use_bn : bool, default True
        Whether to use BatchNorm layer (depthwise convolution block).
    dw_activation : function or str or None, default F.relu
        Activation function after the depthwise convolution block.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 first_stage_stride,
                 dw_use_bn=True,
                 dw_activation=(lambda: F.relu),
                 in_channels=3,
                 in_size=(224, 224),
                 classes=1000):
        super(MobileNet, self).__init__()
        self.in_size = in_size
        self.classes = classes

        with self.init_scope():
            self.features = SimpleSequential()
            with self.features.init_scope():
                init_block_channels = channels[0][0]
                setattr(self.features, "init_block", conv3x3_block(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    stride=2))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(channels[1:]):
                    stage = SimpleSequential()
                    with stage.init_scope():
                        for j, out_channels in enumerate(channels_per_stage):
                            stride = 2 if (j == 0) and ((i != 0) or first_stage_stride) else 1
                            setattr(stage, "unit{}".format(j + 1), dwsconv3x3_block(
                                in_channels=in_channels,
                                out_channels=out_channels,
                                stride=stride,
                                dw_use_bn=dw_use_bn,
                                dw_activation=dw_activation))
                            in_channels = out_channels
                    setattr(self.features, "stage{}".format(i + 1), stage)
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


def get_mobilenet(width_scale,
                  dws_simplified=False,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".chainer", "models"),
                  **kwargs):
    """
    Create MobileNet model with specific parameters.

    Parameters:
    ----------
    width_scale : float
        Scale factor for width of layers.
    dws_simplified : bool, default False
        Whether to use simplified depthwise separable convolution block.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    channels = [[32], [64], [128, 128], [256, 256], [512, 512, 512, 512, 512, 512], [1024, 1024]]
    first_stage_stride = False

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) for cij in ci] for ci in channels]

    if dws_simplified:
        dw_use_bn = False
        dw_activation = None
    else:
        dw_use_bn = True
        dw_activation = (lambda: F.relu)

    net = MobileNet(
        channels=channels,
        first_stage_stride=first_stage_stride,
        dw_use_bn=dw_use_bn,
        dw_activation=dw_activation,
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


def mobilenet_w1(**kwargs):
    """
    1.0 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=1.0, model_name="mobilenet_w1", **kwargs)


def mobilenet_w3d4(**kwargs):
    """
    0.75 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=0.75, model_name="mobilenet_w3d4", **kwargs)


def mobilenet_wd2(**kwargs):
    """
    0.5 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=0.5, model_name="mobilenet_wd2", **kwargs)


def mobilenet_wd4(**kwargs):
    """
    0.25 MobileNet-224 model from 'MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications,'
    https://arxiv.org/abs/1704.04861.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_mobilenet(width_scale=0.25, model_name="mobilenet_wd4", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    pretrained = False

    models = [
        mobilenet_w1,
        mobilenet_w3d4,
        mobilenet_wd2,
        mobilenet_wd4,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != mobilenet_w1 or weight_count == 4231976)
        assert (model != mobilenet_w3d4 or weight_count == 2585560)
        assert (model != mobilenet_wd2 or weight_count == 1331592)
        assert (model != mobilenet_wd4 or weight_count == 470072)

        x = np.zeros((1, 3, 224, 224), np.float32)
        y = net(x)
        assert (y.shape == (1, 1000))


if __name__ == "__main__":
    _test()
