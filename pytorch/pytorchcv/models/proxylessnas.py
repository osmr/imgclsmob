import torch.nn as nn
from .common import ConvBlock, conv1x1_block, conv3x3_block

__all__ = ['proxylessnas_cpu', 'proxylessnas_gpu', 'proxylessnas_mobile', 'proxylessnas_mobile14']


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
        assert (type(kernel_size) == int)
        assert (expand_ratio >= 1)
        assert (kernel_size in [3, 5, 7])
        mid_channels = round(in_channels * expand_ratio)

        if expand_ratio > 1:
            self.inverted_bottleneck = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                activation="relu6")
        else:
            self.inverted_bottleneck = None

        padding = (kernel_size - 1) // 2
        self.depth_conv = ConvBlock(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=mid_channels,
            activation="relu6")

        self.point_linear = conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels,
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
                 channels,
                 init_block_channels,
                 final_block_channels,
                 residuals,
                 shortcuts,
                 kernel_sizes,
                 expand_ratios,
                 in_channels=3,
                 num_classes=1000):
        super(ProxylessNAS, self).__init__()

        self.features = nn.Sequential()
        self.features.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2))
        in_channels = init_block_channels

        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            residuals_per_stage = residuals[i]
            shortcuts_per_stage = shortcuts[i]
            kernel_sizes_per_stage = kernel_sizes[i]
            expand_ratios_per_stage = expand_ratios[i]
            for j, out_channels in enumerate(channels_per_stage):
                residual = residuals_per_stage[j]
                shortcut = shortcuts_per_stage[j]
                if residual == 1:
                    kernel_size = kernel_sizes_per_stage[j]
                    stride = 2 if (j == 0) and (i != 0) else 1
                    expand_ratio = expand_ratios_per_stage[j]
                    mobile_inverted_conv = MBInvertedConvLayer(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        expand_ratio=expand_ratio)
                else:
                    mobile_inverted_conv = None
                stage.add_module("unit{}".format(j + 1), ProxylessUnit(
                    mobile_inverted_conv=mobile_inverted_conv,
                    use_shortcut=shortcut))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)

        self.features.add_module("final_block", conv1x1_block(
            in_channels=channels[-1][-1],
            out_channels=final_block_channels))
        in_channels = final_block_channels
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self.set_bn_param(bn_momentum=0.1, bn_eps=1e-3)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

    def set_bn_param(self, bn_momentum, bn_eps):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = bn_momentum
                m.eps = bn_eps
        return


def get_proxylessnas(version):
    if version == 'cpu':
        # net_config_path = "../imgclsmob_data/proxyless/proxyless_cpu.config"
        residuals = [[1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [32, 32, 32, 32], [48, 48, 48, 48], [88, 88, 88, 88, 104, 104, 104, 104],
                    [216, 216, 216, 216, 360]]
        kernel_sizes = [[3], [3, 3, 3, 3], [3, 3, 3, 5], [3, 3, 3, 3, 5, 3, 3, 3], [5, 5, 5, 3, 5]]
        expand_ratios = [[1], [6, 3, 3, 3], [6, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 3, 3, 3, 6]]
        init_block_channels = 40
        final_block_channels = 1432
    elif version == 'gpu':
        # net_config_path = "../imgclsmob_data/proxyless/proxyless_gpu.config"
        residuals = [[1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [32, 32, 32, 32], [56, 56, 56, 56], [112, 112, 112, 112, 128, 128, 128, 128],
                    [256, 256, 256, 256, 432]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 3, 3], [7, 5, 5, 5, 5, 3, 3, 5], [7, 7, 7, 5, 7]]
        expand_ratios = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 6, 6, 6]]
        init_block_channels = 40
        final_block_channels = 1728
    elif version == 'mobile':
        # net_config_path = "../imgclsmob_data/proxyless/proxyless_mobile.config"
        residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[16], [32, 32, 32, 32], [40, 40, 40, 40], [80, 80, 80, 80, 96, 96, 96, 96],
                    [192, 192, 192, 192, 320]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
        expand_ratios = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
        init_block_channels = 32
        final_block_channels = 1280
    elif version == 'mobile14':
        # net_config_path = "../imgclsmob_data/proxyless/proxyless_mobile_14.config"
        residuals = [[1], [1, 1, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
        channels = [[24], [40, 40, 40, 40], [56, 56, 56, 56], [112, 112, 112, 112, 136, 136, 136, 136],
                    [256, 256, 256, 256, 448]]
        kernel_sizes = [[3], [5, 3, 3, 3], [7, 3, 5, 5], [7, 5, 5, 5, 5, 5, 5, 5], [7, 7, 7, 7, 7]]
        expand_ratios = [[1], [3, 3, 3, 3], [3, 3, 3, 3], [6, 3, 3, 3, 6, 3, 3, 3], [6, 6, 3, 3, 6]]
        init_block_channels = 48
        final_block_channels = 1792
    else:
        raise ValueError("Unsupported proxylessnas version {}".format(version))

    shortcuts = [[0], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 1, 0, 1, 1, 1], [0, 1, 1, 1, 0]]

    # import json
    # config = json.load(open(net_config_path, 'r'))
    net = ProxylessNAS(
        channels=channels,
        init_block_channels=init_block_channels,
        final_block_channels=final_block_channels,
        residuals=residuals,
        shortcuts=shortcuts,
        kernel_sizes=kernel_sizes,
        expand_ratios=expand_ratios)

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
