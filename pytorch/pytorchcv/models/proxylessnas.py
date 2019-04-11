import torch.nn as nn
from common import ConvBlock


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        IdentityLayer.__name__: IdentityLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        ZeroLayer.__name__: ZeroLayer,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class BasicUnit(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


class IdentityLayer(BasicUnit):

    def __init__(self,
                 **kwargs):
        super(IdentityLayer, self).__init__()

    def forward(self, x):
        return x

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False


class ZeroLayer(BasicUnit):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        return None

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    @staticmethod
    def is_zero_layer():
        return True


class ConvLayer(BasicUnit):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 **kwargs):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
            bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False


class MBInvertedConvLayer(BasicUnit):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 expand_ratio=6):
        super(MBInvertedConvLayer, self).__init__()
        assert (expand_ratio >= 1)
        mid_channels = round(in_channels * expand_ratio)

        if expand_ratio > 1:
            self.inverted_bottleneck = ConvBlock(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                activation="relu6")
        else:
            self.inverted_bottleneck = None

        pad = get_same_padding(kernel_size)
        self.depth_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=mid_channels,
            activation="relu6")

        self.point_linear = ConvBlock(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            activation=None,
            activate=False)

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x

    @staticmethod
    def build_from_config(config):
        return MBInvertedConvLayer(**config)

    @staticmethod
    def is_zero_layer():
        return False


class MobileInvertedResidualBlock(BasicUnit):
    def __init__(self,
                 mobile_inverted_conv,
                 shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        assert (type(mobile_inverted_conv) in [MBInvertedConvLayer, ZeroLayer])
        assert (shortcut is None) or (type(shortcut) == IdentityLayer)

        if type(mobile_inverted_conv) == ZeroLayer:
            self.mobile_inverted_conv = None
        else:
            self.mobile_inverted_conv = mobile_inverted_conv

        # self.shortcut = shortcut
        self.use_shortcut = (shortcut is not None)

    def forward(self, x):
        if self.mobile_inverted_conv is None:
            res = x
        elif not self.use_shortcut:
            assert (type(self.mobile_inverted_conv) == MBInvertedConvLayer)
            res = self.mobile_inverted_conv(x)
        else:
            assert (type(self.mobile_inverted_conv) == MBInvertedConvLayer)
            # assert (type(self.shortcut) == IdentityLayer)
            conv_x = self.mobile_inverted_conv(x)
            # skip_x = self.shortcut(x)
            skip_x = x
            res = skip_x + conv_x
        return res

    @staticmethod
    def build_from_config(config):
        mobile_inverted_conv = set_layer_from_config(config['mobile_inverted_conv'])
        shortcut = set_layer_from_config(config['shortcut'])
        return MobileInvertedResidualBlock(mobile_inverted_conv, shortcut)


class ProxylessNASNets(BasicUnit):

    def __init__(self,
                 first_conv,
                 blocks,
                 feature_mix_layer,
                 classifier_in_features,
                 num_classes=1000):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(
            in_features=classifier_in_features,
            out_features=num_classes,
            bias=True)

    def forward(self, x):
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        if self.feature_mix_layer:
            x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    @staticmethod
    def build_from_config(config):

        first_conv = set_layer_from_config(config['first_conv'])
        feature_mix_layer = set_layer_from_config(config['feature_mix_layer'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        return ProxylessNASNets(
            first_conv,
            blocks,
            feature_mix_layer,
            classifier_in_features=config['classifier']['in_features'])


def proxylessnas_cpu(pretrained=False):
    import json
    net_config_path = "../imgclsmob_data/proxyless/proxyless_cpu.config"
    net_config_json = json.load(open(net_config_path, 'r'))
    return ProxylessNASNets.build_from_config(net_config_json)


def proxylessnas_gpu(pretrained=False):
    import json
    net_config_path = "../imgclsmob_data/proxyless/proxyless_gpu.config"
    net_config_json = json.load(open(net_config_path, 'r'))
    return ProxylessNASNets.build_from_config(net_config_json)


def proxylessnas_mobile(pretrained=False):
    import json
    net_config_path = "../imgclsmob_data/proxyless/proxyless_mobile.config"
    net_config_json = json.load(open(net_config_path, 'r'))
    return ProxylessNASNets.build_from_config(net_config_json)


def proxylessnas_mobile14(pretrained=False):
    import json
    net_config_path = "../imgclsmob_data/proxyless/proxyless_mobile_14.config"
    net_config_json = json.load(open(net_config_path, 'r'))
    return ProxylessNASNets.build_from_config(net_config_json)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        (proxylessnas_cpu, 1000),
        (proxylessnas_gpu, 1000),
        (proxylessnas_mobile, 1000),
        (proxylessnas_mobile14, 1000),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != proxylessnas_cpu or weight_count == 4361648)
        assert (model != proxylessnas_gpu or weight_count == 7119848)
        assert (model != proxylessnas_mobile or weight_count == 4080512)
        assert (model != proxylessnas_mobile14 or weight_count == 6857568)

        x = Variable(torch.randn(14, 3, 224, 224))
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, num_classes))


if __name__ == "__main__":
    _test()
