import os
import torch
import torch.nn as nn
from collections import OrderedDict


__all__ = ['VovNet', 'oth_vovnet27_slim', 'oth_vovnet39', 'oth_vovnet57']


def conv3x3_block(in_channels, out_channels, module_name, postfix,
                  stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


def conv1x1_block(in_channels, out_channels, module_name, postfix,
                  stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution"""
    return [
        ('{}_{}/conv'.format(module_name, postfix),
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding,
                      groups=groups,
                      bias=False)),
        ('{}_{}/norm'.format(module_name, postfix),
            nn.BatchNorm2d(out_channels)),
        ('{}_{}/relu'.format(module_name, postfix),
            nn.ReLU(inplace=True)),
    ]


class VovUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 branch_channels,
                 num_branches,
                 module_name,
                 resize,
                 use_residual):
        super(VovUnit, self).__init__()
        self.resize = resize
        self.use_residual = use_residual

        if self.resize:
            self.pool = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                ceil_mode=True)

        self.layers = nn.ModuleList()
        branch_in_channels = in_channels
        for i in range(num_branches):
            self.layers.append(nn.Sequential(OrderedDict(conv3x3_block(
                in_channels=branch_in_channels,
                out_channels=branch_channels,
                module_name=module_name,
                postfix=i))))
            branch_in_channels = branch_channels

        self.concat_conv = nn.Sequential(OrderedDict(conv1x1_block(
            in_channels=(in_channels + num_branches * branch_channels),
            out_channels=out_channels,
            module_name=module_name,
            postfix='concat')))

    def forward(self, x):
        if self.resize:
            x = self.pool(x)

        if self.use_residual:
            identity = x

        outs = [x]
        for layer in self.layers:
            x = layer(x)
            outs.append(x)
        x = torch.cat(outs, dim=1)

        x = self.concat_conv(x)

        if self.use_residual:
            x = x + identity

        return x


# class VovStage(nn.Sequential):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  branch_channels,
#                  num_units,
#                  num_branches,
#                  stage_num):
#         super(VovStage, self).__init__()
#
#         module_name = 'OSA{stage_num}_1'.format(stage_num=stage_num)
#         self.add_module(
#             module_name,
#             VovUnit(
#                 in_channels=in_channels,
#                 out_channels=out_channels,
#                 branch_channels=branch_channels,
#                 num_branches=num_branches,
#                 module_name=module_name,
#                 resize=(not stage_num == 2),
#                 use_residual=False))
#         for i in range(num_units - 1):
#             module_name = 'OSA{}_{}'.format(stage_num, i+2)
#             self.add_module(
#                 module_name,
#                 VovUnit(
#                     in_channels=out_channels,
#                     out_channels=out_channels,
#                     branch_channels=branch_channels,
#                     num_branches=num_branches,
#                     module_name=module_name,
#                     resize=False,
#                     use_residual=True))


class VovInitBlock(nn.Module):
    """
    ResNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(VovInitBlock, self).__init__()
        mid_channels = out_channels // 2
        stem = conv3x3_block(in_channels, mid_channels, 'stem', '1', 2)
        stem += conv3x3_block(mid_channels, mid_channels, 'stem', '2', 1)
        stem += conv3x3_block(mid_channels, out_channels, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

    def forward(self, x):
        x = self.stem(x)
        return x


class VovNet(nn.Module):
    def __init__(self,
                 channels,
                 branch_channels,
                 num_branches,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(VovNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        init_block_channels = 128

        self.features = nn.Sequential()
        self.features.add_module("init_block", VovInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                use_residual = (j != 0)
                resize = (j == 0) and (i != 0)
                module_name = 'OSA{}_{}'.format(i + 1, j + 1)
                stage.add_module("unit{}".format(j + 1), VovUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    branch_channels=branch_channels[i][j],
                    num_branches=num_branches,
                    module_name=module_name,
                    resize=resize,
                    use_residual=use_residual))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_vovnet(blocks,
               slim=False,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create ResNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    slim : bool, default False
        Whether to make slim model.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if blocks == 27:
        layers = [1, 1, 1, 1]
    elif blocks == 39:
        layers = [1, 1, 2, 2]
    elif blocks == 57:
        layers = [1, 1, 4, 3]
    else:
        raise ValueError("Unsupported VoVNet with number of blocks: {}".format(blocks))

    num_branches = 5
    channels_per_layers = [256, 512, 768, 1024]
    branch_channels_per_layers = [128, 160, 192, 224]
    if slim:
        channels_per_layers = [ci // 2 for ci in channels_per_layers]
        branch_channels_per_layers = [ci // 2 for ci in branch_channels_per_layers]

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]
    branch_channels = [[ci] * li for (ci, li) in zip(branch_channels_per_layers, layers)]

    model = VovNet(
        channels=channels,
        branch_channels=branch_channels,
        num_branches=num_branches,
        **kwargs)
    return model


def oth_vovnet27_slim(pretrained=False, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return get_vovnet(
        blocks=27,
        slim=True,
        model_name='vovnet27_slim',
        pretrained=pretrained,
        **kwargs)


def oth_vovnet39(pretrained=False, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return get_vovnet(
        blocks=39,
        model_name='vovnet39',
        pretrained=pretrained,
        **kwargs)


def oth_vovnet57(pretrained=False, **kwargs):
    r"""Constructs a VoVNet-57 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return get_vovnet(
        blocks=57,
        model_name='vovnet57',
        pretrained=pretrained,
        **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch

    pretrained = False

    models = [
        oth_vovnet27_slim,
        oth_vovnet39,
        oth_vovnet57,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_vovnet27_slim or weight_count == 3525736)
        assert (model != oth_vovnet39 or weight_count == 22600296)
        assert (model != oth_vovnet57 or weight_count == 36640296)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
