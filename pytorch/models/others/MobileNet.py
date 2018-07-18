import torch
import torch.nn as nn
import torch.nn.init as init
from collections import OrderedDict

__all__ = [
    'mobilenet',
    'shallow_mobilenet',
    'fast_downsampling_mobilenet',
    'oth_mobilenet1_0',
    'oth_mobilenet0_75',
    'oth_mobilenet0_5',
    'oth_mobilenet0_25',
    'oth_fd_mobilenet1_0',
    'oth_fd_mobilenet0_75',
    'oth_fd_mobilenet0_5',
    'oth_fd_mobilenet0_25'
]

_mobilenet_shallow_channels = [64, 128, 128, 256, 256, 512, 1024, 1024]
_mobilenet_shallow_strides = [1, 2, 1, 2, 1, 2, 2, 1]
_mobilenet_channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
_mobilenet_strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1]
_mobilenet_fast_downsampling_channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 1024]
_mobilenet_fast_downsampling_strides = [2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1]

class MobileNet(nn.Module):
    def __init__(self, init_features, channels, strides):
        super(MobileNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv_0', nn.Conv2d(3, init_features, 3, stride=2, padding=1, bias=False)),
            ('norm_0', nn.BatchNorm2d(init_features)),
            ('relu_0', nn.ReLU(inplace=True)),
        ]))
        in_c = init_features
        for _, (out_c, stride) in enumerate(zip(channels, strides)):
            self.features.add_module('dw_conv_{}'.format(_), nn.Conv2d(in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False))
            self.features.add_module('dw_norm_{}'.format(_), nn.BatchNorm2d(in_c))
            self.features.add_module('dw_relu_{}'.format(_), nn.ReLU(inplace=True))
            self.features.add_module('pw_conv_{}'.format(_), nn.Conv2d(in_c, out_c, 1, bias=False))
            self.features.add_module('pw_norm_{}'.format(_), nn.BatchNorm2d(out_c))
            self.features.add_module('pw_relu_{}'.format(_), nn.ReLU(inplace=True))
            in_c = out_c
        self.pool = nn.AvgPool2d(7)
        self.classifier = nn.Linear(in_c, 1000)
        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if 'dw_conv' in name:
                init.kaiming_normal(module.weight, mode='fan_in')
            elif name == 'conv_0' or 'pw_conv' in name:
                init.kaiming_normal(module.weight, mode='fan_out')
            elif 'norm' in name:
                init.constant(module.weight, 1)
                init.constant(module.bias, 0)
            elif 'classifier' in name:
                init.kaiming_normal(module.weight, mode='fan_out')
                init.constant(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def mobilenet(model_config):
    width_mul = model_config['width_mul']
    init_features = int(32 * width_mul)
    channels = [int(x * width_mul) for x in _mobilenet_channels]
    return MobileNet(init_features, channels, _mobilenet_strides)

def shallow_mobilenet(model_config):
    width_mul = model_config['width_mul']
    init_features = int(32 * width_mul)
    channels = [int(x * width_mul) for x in _mobilenet_shallow_channels]
    return MobileNet(init_features, channels, _mobilenet_shallow_strides)

def fast_downsampling_mobilenet(model_config):
    width_mul = model_config['width_mul']
    init_features = int(32 * width_mul)
    channels = [int(x * width_mul) for x in _mobilenet_fast_downsampling_channels]
    return MobileNet(init_features, channels, _mobilenet_fast_downsampling_strides)


def oth_mobilenet1_0(**kwargs):
    return mobilenet({'width_mul': 1})


def oth_mobilenet0_75(**kwargs):
    return mobilenet({'width_mul': 0.75})


def oth_mobilenet0_5(**kwargs):
    return mobilenet({'width_mul': 0.5})


def oth_mobilenet0_25(**kwargs):
    return mobilenet({'width_mul': 0.25})


def oth_fd_mobilenet1_0(**kwargs):
    return fast_downsampling_mobilenet({'width_mul': 1})


def oth_fd_mobilenet0_75(**kwargs):
    return fast_downsampling_mobilenet({'width_mul': 0.75})


def oth_fd_mobilenet0_5(**kwargs):
    return fast_downsampling_mobilenet({'width_mul': 0.5})


def oth_fd_mobilenet0_25(**kwargs):
    return fast_downsampling_mobilenet({'width_mul': 0.25})


