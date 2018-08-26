"""
    DPN, implemented in PyTorch.
    Original paper: 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.
"""

__all__ = ['DPN', 'dpn68']

import os
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .common import conv1x1


def dpn_batch_norm(channels):
    """
    DPN specific Batch normalization layer.
    """
    return nn.BatchNorm2d(
        num_features=channels,
        eps=0.001)


class CatBnActivation(nn.Module):
    """
    DPN final block, which performs the preactivation with cutting.

    Parameters:
    ----------
    channels : int
        Number of channels.
    """
    def __init__(self,
                 channels):
        super(CatBnActivation, self).__init__()
        self.bn = dpn_batch_norm(channels=channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        x = self.bn(x)
        x = self.activ(x)
        return x


class DPNConv(nn.Module):
    """
    DPN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups):
        super(DPNConv, self).__init__()
        self.bn = dpn_batch_norm(channels=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        x = self.conv(x)
        return x


def dpn_conv1x1(in_channels,
                out_channels,
                stride=1):
    """
    1x1 version of the DPN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return DPNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        groups=1)


def dpn_conv3x3(in_channels,
                out_channels,
                stride,
                groups):
    """
    3x3 version of the DPN specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    return DPNConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        groups=groups)


class DPNUnit(nn.Module):
    """
    DPN unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout_rate : bool
        Parameter of Dropout layer. Faction of the input units to drop.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 out_channels_1x1a,
                 out_channels_3x3b,
                 out_channels_1x1c,
                 inc,
                 groups,
                 block_type='normal',
                 b=False):
        super(DPNUnit, self).__init__()

        self.num_1x1_c = out_channels_1x1c
        self.inc = inc
        self.b = b
        if block_type is 'proj':
            self.key_stride = 1
            self.has_proj = True
        elif block_type is 'down':
            self.key_stride = 2
            self.has_proj = True
        else:
            assert block_type is 'normal'
            self.key_stride = 1
            self.has_proj = False

        if self.has_proj:
            if self.key_stride == 2:
                self.conv1x1_w_s2 = dpn_conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels_1x1c + 2 * inc,
                    stride=2)
            else:
                self.conv1x1_w_s1 = dpn_conv1x1(
                    in_channels=in_channels,
                    out_channels=out_channels_1x1c + 2 * inc)
        self.conv1x1a = dpn_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels_1x1a)
        self.conv3x3b = dpn_conv3x3(
            in_channels=out_channels_1x1a,
            out_channels=out_channels_3x3b,
            stride=self.key_stride,
            groups=groups)
        if b:
            self.conv1x1c = CatBnActivation(channels=out_channels_3x3b)
            self.conv1x1c1 = conv1x1(
                in_channels=out_channels_3x3b,
                out_channels=out_channels_1x1c)
            self.conv1x1c2 = conv1x1(
                in_channels=out_channels_3x3b,
                out_channels=inc)
        else:
            self.conv1x1c = dpn_conv1x1(
                in_channels=out_channels_3x3b,
                out_channels=out_channels_1x1c + inc)

    def forward(self, x):
        x_in = torch.cat(x, dim=1) if isinstance(x, tuple) else x
        if self.has_proj:
            if self.key_stride == 2:
                x_s = self.conv1x1_w_s2(x_in)
            else:
                x_s = self.conv1x1_w_s1(x_in)
            x_s1 = x_s[:, :self.num_1x1_c, :, :]
            x_s2 = x_s[:, self.num_1x1_c:, :, :]
        else:
            x_s1 = x[0]
            x_s2 = x[1]
        x_in = self.conv1x1a(x_in)
        x_in = self.conv3x3b(x_in)
        if self.b:
            x_in = self.conv1x1c(x_in)
            out1 = self.conv1x1c1(x_in)
            out2 = self.conv1x1c2(x_in)
        else:
            x_in = self.conv1x1c(x_in)
            out1 = x_in[:, :self.num_1x1_c, :, :]
            out2 = x_in[:, self.num_1x1_c:, :, :]
        resid = x_s1 + out1
        dense = torch.cat((x_s2, out2), dim=1)
        return resid, dense


class DPNInitBlock(nn.Module):
    """
    DPN specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 padding):
        super(DPNInitBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=False)
        self.bn = dpn_batch_norm(channels=out_channels)
        self.activ = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activ(x)
        x = self.pool(x)
        return x


def adaptive_avgmax_pool2d(x, pool_type='avg', padding=0, count_include_pad=False):
    """Selectable global pooling function with dynamic input kernel size
    """
    if pool_type == 'avgmaxc':
        x = torch.cat([
            F.avg_pool2d(
                x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad),
            F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        ], dim=1)
    elif pool_type == 'avgmax':
        x_avg = F.avg_pool2d(
                x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
        x_max = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
        x = 0.5 * (x_avg + x_max)
    elif pool_type == 'max':
        x = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=padding)
    else:
        if pool_type != 'avg':
            print('Invalid pool type %s specified. Defaulting to average pooling.' % pool_type)
        x = F.avg_pool2d(
            x, kernel_size=(x.size(2), x.size(3)), padding=padding, count_include_pad=count_include_pad)
    return x


class AdaptiveAvgMaxPool2d(torch.nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg'):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size
        self.pool_type = pool_type
        if pool_type == 'avgmaxc' or pool_type == 'avgmax':
            self.pool = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size), nn.AdaptiveMaxPool2d(output_size)])
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            if pool_type != 'avg':
                print('Invalid pool type %s specified. Defaulting to average pooling.' % pool_type)
            self.pool = nn.AdaptiveAvgPool2d(output_size)

    def forward(self, x):
        if self.pool_type == 'avgmaxc':
            x = torch.cat([p(x) for p in self.pool], dim=1)
        elif self.pool_type == 'avgmax':
            x = 0.5 * torch.sum(torch.stack([p(x) for p in self.pool]), 0).squeeze(dim=0)
        else:
            x = self.pool(x)
        return x

    def factor(self):
        return self._pooling_factor(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'output_size=' + str(self.output_size) \
               + ', pool_type=' + self.pool_type + ')'

    @staticmethod
    def _pooling_factor(pool_type='avg'):
        return 2 if pool_type == 'avgmaxc' else 1


class DPN(nn.Module):
    """
    DPN model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    in_channels : int, default 3
        Number of input channels.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 init_block_kernel_size,
                 init_block_padding,
                 bw_factor,
                 inc_sec,
                 k_r,
                 groups,
                 b,
                 test_time_pool,
                 in_channels=3,
                 num_classes=1000):
        super(DPN, self).__init__()
        self.test_time_pool = test_time_pool
        self.b = b

        self.features = nn.Sequential()
        self.features.add_module("init_block", DPNInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            kernel_size=init_block_kernel_size,
            padding=init_block_padding))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            stage = nn.Sequential()
            r = (2 ** i) * k_r
            bw = (2 ** i) * 64 * bw_factor
            inc = inc_sec[i]
            for j, out_channels in enumerate(channels_per_stage):
                if j == 0:
                    if i == 0:
                        block_type = "proj"
                    else:
                        block_type = "down"
                else:
                    block_type = "normal"
                stage.add_module("unit{}".format(j + 1), DPNUnit(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    out_channels_1x1a=r,
                    out_channels_3x3b=r,
                    out_channels_1x1c=bw,
                    inc=inc,
                    groups=groups,
                    block_type=block_type,
                    b=b))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('post_activ', CatBnActivation(channels=in_channels))

        # self.output = nn.Sequential()
        # if not self.training and self.test_time_pool:
        #     self.output.add_module('avg_pool', nn.AvgPool2d(
        #         kernel_size=7,
        #         stride=1))
        #     self.output.add_module('classifier', conv1x1(
        #         in_channels=in_channels,
        #         out_channels=num_classes,
        #         bias=True))
        #     self.output.add_module('avgmax_pool', AdaptiveAvgMaxPool2d(
        #         pool_type='avgmax'))
        # else:
        #     self.output.add_module('avg_pool', AdaptiveAvgMaxPool2d(
        #         pool_type='avg'))
        #     self.output.add_module('classifier', conv1x1(
        #         in_channels=in_channels,
        #         out_channels=num_classes,
        #         bias=True))

        self.output = conv1x1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def logits(self, features):
        if not self.training and self.test_time_pool:
            x = F.avg_pool2d(features, kernel_size=7, stride=1)
            out = self.output(x)
            # The extra test time pool should be pooling an img_size//32 - 6 size patch
            out = adaptive_avgmax_pool2d(out, pool_type='avgmax')
        else:
            x = adaptive_avgmax_pool2d(features, pool_type='avg')
            out = self.output(x)
        return out.view(out.size(0), -1)

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x


def get_dpn(num_layers,
            model_name=None,
            pretrained=False,
            root=os.path.join('~', '.torch', 'models'),
            **kwargs):
    """
    Create DPN model with specific parameters.

    Parameters:
    ----------
    num_layers : int
        Number of layers.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """

    if num_layers == 68:
        small = True
        #num_init_features = 10
        init_block_channels = 10
        init_block_kernel_size = 3
        init_block_padding = 1
        k_r = 128
        groups = 32
        k_sec = (3, 4, 12, 3)
        inc_sec = (16, 32, 32, 64)
        test_time_pool = True
        b = False
    else:
        raise ValueError("Unsupported DPN version with number of layers {}".format(num_layers))

    bw_factor = 1 if small else 4

    channels = [[0] * li for li in k_sec]
    for i in range(len(k_sec)):
        bw = (2 ** i) * 64 * bw_factor
        inc = inc_sec[i]
        channels[i][0] = bw + 3 * inc
        for j in range(1, k_sec[i]):
            channels[i][j] = channels[i][j-1] + inc

    net = DPN(
        channels=channels,
        init_block_channels=init_block_channels,
        init_block_kernel_size=init_block_kernel_size,
        init_block_padding=init_block_padding,
        bw_factor=bw_factor,
        inc_sec=inc_sec,
        k_r=k_r,
        groups=groups,
        b=b,
        test_time_pool=test_time_pool,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        import torch
        from .model_store import get_model_file
        net.load_state_dict(torch.load(get_model_file(
            model_name=model_name,
            local_model_store_dir_path=root)))

    return net


def dpn68(**kwargs):
    """
    DPN-68 model from 'Dual Path Networks,' https://arxiv.org/abs/1707.01629.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_dpn(num_layers=68, model_name="dpn68", **kwargs)


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        dpn68,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        net.train()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dpn68 or weight_count == 12611602)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()

