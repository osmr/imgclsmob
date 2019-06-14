"""
    ResNet(D) with dilation for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

__all__ = ['ResNetD', 'resnetd50b', 'resnetd101b', 'resnetd152b']

import os
import torch.nn as nn
import torch.nn.init as init
from .common import MultiOutputSequential
from .resnet import ResUnit, ResInitBlock
from .senet import SEInitBlock


class ResNetD(nn.Module):
    """
    ResNet(D) with dilation model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

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
    ordinary_init : bool, default False
        Whether to use original initial block or SENet one.
    multi_output : bool, default False
        Whether to return intermediate outputs.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 bottleneck,
                 conv1_stride,
                 ordinary_init=False,
                 multi_output=False,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(ResNetD, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        self.multi_output = multi_output

        self.features = MultiOutputSequential()
        if ordinary_init:
            self.features.add_module("init_block", ResInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels))
        else:
            init_block_channels = 2 * init_block_channels
            self.features.add_module("init_block", SEInitBlock(
                in_channels=in_channels,
                out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if ((j == 0) and (i != 0) and (i < 2)) else 1
                dilation = (2 ** max(0, i - 1 - int(j == 0)))
                stage.add_module("unit{}".format(j + 1), ResUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    padding=dilation,
                    dilation=dilation,
                    bottleneck=bottleneck,
                    conv1_stride=conv1_stride))
                in_channels = out_channels
            if i == 2:
                stage.do_output = True
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        outs = self.features(x)
        x = outs[0]
        x = x.view(x.size(0), -1)
        x = self.output(x)
        if self.multi_output:
            return [x] + outs[1:]
        else:
            return x


def get_resnetd(blocks,
                conv1_stride=True,
                width_scale=1.0,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create ResNet(D) with dilation model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    conv1_stride : bool, default True
        Whether to use stride in the first or the second convolution layer in units.
    width_scale : float, default 1.0
        Scale factor for width of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
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
        raise ValueError("Unsupported ResNet(D) with number of blocks: {}".format(blocks))

    init_block_channels = 64

    if blocks < 50:
        channels_per_layers = [64, 128, 256, 512]
        bottleneck = False
    else:
        channels_per_layers = [256, 512, 1024, 2048]
        bottleneck = True

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    if width_scale != 1.0:
        channels = [[int(cij * width_scale) if (i != len(channels) - 1) or (j != len(ci) - 1) else cij
                     for j, cij in enumerate(ci)] for i, ci in enumerate(channels)]
        init_block_channels = int(init_block_channels * width_scale)

    net = ResNetD(
        channels=channels,
        init_block_channels=init_block_channels,
        bottleneck=bottleneck,
        conv1_stride=conv1_stride,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def resnetd50b(**kwargs):
    """
    ResNet(D)-50 with dilation model with stride at the second convolution in bottleneck block from 'Deep Residual
    Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnetd(blocks=50, conv1_stride=False, model_name="resnetd50b", **kwargs)


def resnetd101b(**kwargs):
    """
    ResNet(D)-101 with dilation model with stride at the second convolution in bottleneck block from 'Deep Residual
    Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnetd(blocks=101, conv1_stride=False, model_name="resnetd101b", **kwargs)


def resnetd152b(**kwargs):
    """
    ResNet(D)-152 with dilation model with stride at the second convolution in bottleneck block from 'Deep Residual
    Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_resnetd(blocks=152, conv1_stride=False, model_name="resnetd152b", **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    ordinary_init = False
    multi_output = False
    pretrained = False

    models = [
        resnetd50b,
        resnetd101b,
        resnetd152b,
    ]

    for model in models:

        net = model(
            pretrained=pretrained,
            ordinary_init=ordinary_init,
            multi_output=multi_output)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        if ordinary_init:
            assert (model != resnetd50b or weight_count == 25557032)
            assert (model != resnetd101b or weight_count == 44549160)
            assert (model != resnetd152b or weight_count == 60192808)
        else:
            assert (model != resnetd50b or weight_count == 25680808)
            assert (model != resnetd101b or weight_count == 44672936)
            assert (model != resnetd152b or weight_count == 60316584)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        if multi_output:
            y = y[0]
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
