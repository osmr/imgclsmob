import torch.nn as nn
from .common import ConvBlock, conv1x1_block, conv3x3_block


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class MBInvertedConvLayer(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 expand_ratio=6,
                 **kwargs):
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


class ProxylessUnit(nn.Module):
    def __init__(self,
                 mobile_inverted_conv,
                 use_shortcut):
        super(ProxylessUnit, self).__init__()
        self.mobile_inverted_conv = mobile_inverted_conv
        self.use_shortcut = use_shortcut

    def forward(self, x):
        if self.mobile_inverted_conv is None:
            res = x
        elif not self.use_shortcut:
            assert (type(self.mobile_inverted_conv) == MBInvertedConvLayer)
            res = self.mobile_inverted_conv(x)
        else:
            assert (type(self.mobile_inverted_conv) == MBInvertedConvLayer)
            conv_x = self.mobile_inverted_conv(x)
            skip_x = x
            res = skip_x + conv_x
        return res


class ProxylessNAS(nn.Module):

    def __init__(self,
                 blocks_config,
                 channels,
                 init_block_channels,
                 final_block_channels,
                 shortcut_list,
                 in_channels=3,
                 num_classes=1000):
        super(ProxylessNAS, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        in_channels = init_block_channels

        for i, block_config in enumerate(blocks_config):
            in_channels = -1
            out_channels = -1
            kernel_size = -1
            stride = -1
            expand_ratio = -1

            mobile_inverted_conv_config = block_config['mobile_inverted_conv']
            mobile_inverted_conv_layer_name = mobile_inverted_conv_config['name']
            if mobile_inverted_conv_layer_name == "ZeroLayer":
                mobile_inverted_conv = None
            else:
                in_channels = mobile_inverted_conv_config['in_channels']
                out_channels = mobile_inverted_conv_config['out_channels']
                kernel_size = mobile_inverted_conv_config['kernel_size']
                stride = mobile_inverted_conv_config['stride']
                expand_ratio = mobile_inverted_conv_config['expand_ratio']
                mobile_inverted_conv = MBInvertedConvLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    expand_ratio=expand_ratio)

            print("i={}, in_channels={}, out_channels={}, stride={}".format(i, in_channels, out_channels, stride))

            use_shortcut = (shortcut_list[i] == 1)

            self.features.add_module("unit{}".format(i + 1), ProxylessUnit(
                mobile_inverted_conv=mobile_inverted_conv,
                use_shortcut=use_shortcut))

        self.features.add_module("final_block", conv1x1_block(
            in_channels=channels[-1],
            out_channels=final_block_channels))
        in_channels = final_block_channels
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_proxylessnas(version):
    if version == 'cpu':
        net_config_path = "../imgclsmob_data/proxyless/proxyless_cpu.config"
        channels = [360]
        init_block_channels = 40
        final_block_channels = 1432
    elif version == 'gpu':
        net_config_path = "../imgclsmob_data/proxyless/proxyless_gpu.config"
        channels = [432]
        init_block_channels = 40
        final_block_channels = 1728
    elif version == 'mobile':
        net_config_path = "../imgclsmob_data/proxyless/proxyless_mobile.config"
        channels = [320]
        init_block_channels = 32
        final_block_channels = 1280
    elif version == 'mobile14':
        net_config_path = "../imgclsmob_data/proxyless/proxyless_mobile_14.config"
        channels = [448]
        init_block_channels = 48
        final_block_channels = 1792
    else:
        raise ValueError("Unsupported proxylessnas version {}".format(version))

    shortcut_list = [0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0]

    import json
    config = json.load(open(net_config_path, 'r'))
    net = ProxylessNAS(
        blocks_config=config['blocks'],
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        shortcut_list=shortcut_list)

    return net


def proxylessnas_cpu(pretrained=False):
    return get_proxylessnas(version="cpu")


def proxylessnas_gpu(pretrained=False):
    return get_proxylessnas(version="gpu")


def proxylessnas_mobile(pretrained=False):
    return get_proxylessnas(version="mobile")


def proxylessnas_mobile14(pretrained=False):
    return get_proxylessnas(version="mobile14")


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
        # y.sum().backward()
        assert (tuple(y.size()) == (14, num_classes))


if __name__ == "__main__":
    _test()
