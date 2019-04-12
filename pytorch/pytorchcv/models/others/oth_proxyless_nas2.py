from collections import OrderedDict
import torch
import torch.nn as nn

__all__ = ['oth_proxyless_nas_cpu', 'oth_proxyless_nas_gpu', 'oth_proxyless_nas_mobile', 'oth_proxyless_nas_mobile_14']


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class BasicUnit(nn.Module):

    def forward(self, x):
        raise NotImplementedError

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError


def set_layer_from_config(layer_config):
    if layer_config is None:
        return None

    name2layer = {
        ConvLayer.__name__: ConvLayer,
        PoolingLayer.__name__: PoolingLayer,
        IdentityLayer.__name__: IdentityLayer,
        LinearLayer.__name__: LinearLayer,
        MBInvertedConvLayer.__name__: MBInvertedConvLayer,
        ZeroLayer.__name__: ZeroLayer,
    }

    layer_name = layer_config.pop('name')
    layer = name2layer[layer_name]
    return layer.build_from_config(layer_config)


class BasicLayer(BasicUnit):

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn=True,
                 act_func='relu6',
                 dropout_rate=0,
                 ops_order='weight_bn_act'):
        super(BasicLayer, self).__init__()

        assert (act_func is None) or (act_func == 'relu6')
        assert (dropout_rate == 0.0)
        assert (ops_order == 'weight_bn_act')

        self.ops_order = ops_order

        """ add modules """
        # batch norm
        if use_bn:
            if self.bn_before_weight:
                self.bn = nn.BatchNorm2d(in_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        # activation
        if act_func == 'relu6':
            if self.ops_list[0] == 'act':
                self.activation = nn.ReLU6(inplace=False)
            else:
                self.activation = nn.ReLU6(inplace=True)
        else:
            self.activation = None

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError('Invalid ops_order: %s' % self.ops_order)

    def weight_call(self, x):
        raise NotImplementedError

    def forward(self, x):
        for op in self.ops_list:
            if op == 'weight':
                x = self.weight_call(x)
            elif op == 'bn':
                if self.bn is not None:
                    x = self.bn(x)
            elif op == 'act':
                if self.activation is not None:
                    x = self.activation(x)
            else:
                raise ValueError('Unrecognized op: %s' % op)
        return x

    @staticmethod
    def build_from_config(config):
        raise NotImplementedError

    @staticmethod
    def is_zero_layer():
        return False


class ConvLayer(BasicLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=False,
                 has_shuffle=False,
                 use_bn=True,
                 act_func='relu6',
                 dropout_rate=0,
                 ops_order='weight_bn_act'):
        super(ConvLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        assert (act_func is None) or (act_func == 'relu6')
        assert (not has_shuffle)
        assert (dropout_rate == 0.0)
        assert (ops_order == 'weight_bn_act')

        padding = get_same_padding(kernel_size)
        if isinstance(padding, int):
            padding *= dilation
        else:
            padding[0] *= dilation
            padding[1] *= dilation

        # `kernel_size`, `stride`, `padding`, `dilation` can either be `int` or `tuple` of int
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def weight_call(self, x):
        x = self.conv(x)
        return x

    @staticmethod
    def build_from_config(config):
        return ConvLayer(**config)


class PoolingLayer(BasicLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 pool_type,
                 kernel_size=2,
                 stride=2,
                 use_bn=False,
                 act_func=None,
                 dropout_rate=0,
                 ops_order='weight_bn_act'):
        super(PoolingLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        assert (act_func is None)
        assert (not use_bn)
        assert (dropout_rate == 0.0)
        assert (ops_order == 'weight_bn_act')

        if stride == 1:
            # same padding if `stride == 1`
            padding = get_same_padding(kernel_size)
        else:
            padding = 0

        if pool_type == 'avg':
            self.pool = nn.AvgPool2d(
                kernel_size,
                stride=stride,
                padding=padding,
                count_include_pad=False)
        elif pool_type == 'max':
            self.pool = nn.MaxPool2d(
                kernel_size,
                stride=stride,
                padding=padding)
        else:
            raise NotImplementedError

    def weight_call(self, x):
        return self.pool(x)

    @staticmethod
    def build_from_config(config):
        return PoolingLayer(**config)


class IdentityLayer(BasicLayer):

    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bn=False,
                 act_func=None,
                 dropout_rate=0,
                 ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels, use_bn, act_func, dropout_rate, ops_order)

        assert (act_func is None)
        assert (not use_bn)
        assert (dropout_rate == 0.0)
        assert (ops_order == 'weight_bn_act')

    def weight_call(self, x):
        return x

    @staticmethod
    def build_from_config(config):
        return IdentityLayer(**config)


class LinearLayer(BasicUnit):

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 use_bn=False,
                 act_func=None,
                 dropout_rate=0,
                 ops_order='weight_bn_act'):
        super(LinearLayer, self).__init__()

        assert (act_func is None)
        assert (dropout_rate == 0.0)
        assert (not use_bn)
        assert (ops_order == 'weight_bn_act')

        self.ops_order = ops_order

        self.linear = nn.Linear(in_features, out_features, bias)

    def forward(self, x):
        x = self.linear(x)
        return x

    @staticmethod
    def build_from_config(config):
        return LinearLayer(**config)

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
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(mid_channels)),
                ('relu', nn.ReLU6(inplace=True)),
            ]))
        else:
            self.inverted_bottleneck = None

        pad = get_same_padding(kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size,
                stride,
                pad,
                groups=mid_channels,
                bias=False)),
            ('bn', nn.BatchNorm2d(mid_channels)),
            ('relu', nn.ReLU6(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
        ]))

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


class ZeroLayer(BasicUnit):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        if x.is_cuda:
            with torch.cuda.device(x.get_device()):
                padding = torch.cuda.FloatTensor(n, c, h, w).fill_(0)
        else:
            padding = torch.zeros(n, c, h, w)
        padding = torch.autograd.Variable(padding, requires_grad=False)
        return padding

    @staticmethod
    def build_from_config(config):
        return ZeroLayer(**config)

    @staticmethod
    def is_zero_layer():
        return True


class MobileInvertedResidualBlock(BasicUnit):
    def __init__(self,
                 mobile_inverted_conv,
                 shortcut):
        super(MobileInvertedResidualBlock, self).__init__()

        assert (type(mobile_inverted_conv) in [MBInvertedConvLayer, ZeroLayer])
        assert (shortcut is None) or (type(shortcut) == IdentityLayer)

        self.mobile_inverted_conv = mobile_inverted_conv
        self.shortcut = shortcut

    def forward(self, x):
        if self.mobile_inverted_conv.is_zero_layer():
            res = x
        elif self.shortcut is None or self.shortcut.is_zero_layer():
            res = self.mobile_inverted_conv(x)
        else:
            conv_x = self.mobile_inverted_conv(x)
            skip_x = self.shortcut(x)
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
                 classifier):
        super(ProxylessNASNets, self).__init__()

        self.first_conv = first_conv
        self.blocks = nn.ModuleList(blocks)
        self.feature_mix_layer = feature_mix_layer
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = classifier

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
        classifier = set_layer_from_config(config['classifier'])
        blocks = []
        for block_config in config['blocks']:
            blocks.append(MobileInvertedResidualBlock.build_from_config(block_config))

        return ProxylessNASNets(first_conv, blocks, feature_mix_layer, classifier)


def oth_proxyless_nas_cpu(pretrained=False):
    import json
    net_config_path = "../imgclsmob_data/proxyless/proxyless_cpu.config"
    net_config_json = json.load(open(net_config_path, 'r'))
    return ProxylessNASNets.build_from_config(net_config_json)


def oth_proxyless_nas_gpu(pretrained=False):
    import json
    net_config_path = "../imgclsmob_data/proxyless/proxyless_gpu.config"
    net_config_json = json.load(open(net_config_path, 'r'))
    return ProxylessNASNets.build_from_config(net_config_json)


def oth_proxyless_nas_mobile(pretrained=False):
    import json
    net_config_path = "../imgclsmob_data/proxyless/proxyless_mobile.config"
    net_config_json = json.load(open(net_config_path, 'r'))
    return ProxylessNASNets.build_from_config(net_config_json)


def oth_proxyless_nas_mobile_14(pretrained=False):
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
        (oth_proxyless_nas_cpu, 1000),
        (oth_proxyless_nas_gpu, 1000),
        (oth_proxyless_nas_mobile, 1000),
        (oth_proxyless_nas_mobile_14, 1000),
    ]

    for model, num_classes in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_proxyless_nas_cpu or weight_count == 4361648)
        assert (model != oth_proxyless_nas_gpu or weight_count == 7119848)
        assert (model != oth_proxyless_nas_mobile or weight_count == 4080512)
        assert (model != oth_proxyless_nas_mobile_14 or weight_count == 6857568)

        x = Variable(torch.randn(14, 3, 224, 224))
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (14, num_classes))


if __name__ == "__main__":
    _test()
