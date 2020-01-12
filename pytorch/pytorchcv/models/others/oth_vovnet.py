import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__all__ = ['VoVNet', 'oth_vovnet27_slim', 'oth_vovnet39', 'oth_vovnet57']


def conv3x3(in_channels, out_channels, module_name, postfix,
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


def conv1x1(in_channels, out_channels, module_name, postfix,
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


class _OSA_module(nn.Module):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 module_name,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(nn.Sequential(
                OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(
            OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 block_per_stage,
                 layer_per_block,
                 stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling',
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        module_name = 'OSA{stage_num}_1'.format(stage_num=stage_num)
        self.add_module(module_name,
            _OSA_module(in_ch,
                        stage_ch,
                        concat_ch,
                        layer_per_block,
                        module_name))
        for i in range(block_per_stage-1):
            module_name = 'OSA{}_{}'.format(stage_num, i+2)
            self.add_module(module_name,
                _OSA_module(concat_ch,
                            stage_ch,
                            concat_ch,
                            layer_per_block,
                            module_name,
                            identity=True))


class VoVNet(nn.Module):
    def __init__(self, 
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 num_classes=1000):
        super(VoVNet, self).__init__()

        # Stem module
        stem = conv3x3(3,   64, 'stem', '1', 2)
        stem += conv3x3(64,  64, 'stem', '2', 1)
        stem += conv3x3(64, 128, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential(OrderedDict(stem)))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4): #num_stages
            name = 'stage%d' % (i+2)
            self.stage_names.append(name)
            self.add_module(name,
                            _OSA_stage(in_ch_list[i],
                                       config_stage_ch[i],
                                       config_concat_ch[i],
                                       block_per_stage[i],
                                       layer_per_block,
                                       i+2))

        self.classifier = nn.Linear(config_concat_ch[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x


def _vovnet(arch,
            config_stage_ch,
            config_concat_ch,
            block_per_stage,
            layer_per_block,
            pretrained,
            progress,
            **kwargs):
    model = VoVNet(config_stage_ch, config_concat_ch,
                   block_per_stage, layer_per_block,
                   **kwargs)
    return model


def oth_vovnet57(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-57 model as described in 
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet57', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,4,3], 5, pretrained, progress, **kwargs)


def oth_vovnet39(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet39', [128, 160, 192, 224], [256, 512, 768, 1024],
                    [1,1,2,2], 5, pretrained, progress, **kwargs)


def oth_vovnet27_slim(pretrained=False, progress=True, **kwargs):
    r"""Constructs a VoVNet-39 model as described in
    `"An Energy and GPU-Computation Efficient Backbone Networks"
    <https://arxiv.org/abs/1904.09730>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vovnet('vovnet27_slim', [64, 80, 96, 112], [128, 256, 384, 512],
                    [1,1,1,1], 5, pretrained, progress, **kwargs)


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
