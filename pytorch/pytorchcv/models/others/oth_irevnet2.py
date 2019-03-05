"""
    i-RevNet, implemented in PyTorch.
    Original paper: 'i-RevNet: Deep Invertible Networks,' https://arxiv.org/abs/1802.07088.
"""

__all__ = ['oth_irevnet301']

import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualPathSequential(nn.Sequential):
    """
    A sequential container for modules with dual inputs/outputs.
    Modules will be executed in the order they are added.

    Parameters:
    ----------
    return_two : bool, default True
        Whether to return two output after execution.
    first_ordinals : int, default 0
        Number of the first modules with single input/output.
    last_ordinals : int, default 0
        Number of the final modules with single input/output.
    dual_path_scheme : function
        Scheme of dual path response for a module.
    dual_path_scheme_ordinal : function
        Scheme of dual path response for an ordinal module.
    """
    def __init__(self,
                 return_two=True,
                 first_ordinals=0,
                 last_ordinals=0,
                 dual_path_scheme=(lambda module, x1, x2: module(x1, x2)),
                 dual_path_scheme_ordinal=(lambda module, x1, x2: (module(x1), x2))):
        super(DualPathSequential, self).__init__()
        self.return_two = return_two
        self.first_ordinals = first_ordinals
        self.last_ordinals = last_ordinals
        self.dual_path_scheme = dual_path_scheme
        self.dual_path_scheme_ordinal = dual_path_scheme_ordinal

    def forward(self, x1, x2=None):
        length = len(self._modules.values())
        for i, module in enumerate(self._modules.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(module, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(module, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1

    def inverse(self, x1, x2=None):
        length = len(self._modules.values())
        for i, module in enumerate(self._modules.values()):
            if (i < self.first_ordinals) or (i >= length - self.last_ordinals):
                x1, x2 = self.dual_path_scheme_ordinal(module, x1, x2)
            else:
                x1, x2 = self.dual_path_scheme(module, x1, x2)
        if self.return_two:
            return x1, x2
        else:
            return x1


class RevInjectivePad(nn.Module):
    def __init__(self, pad_size):
        super(RevInjectivePad, self).__init__()
        self.pad_size = pad_size
        self.pad = nn.ZeroPad2d((0, 0, 0, pad_size))

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = self.pad(x)
        return x.permute(0, 2, 1, 3)

    def inverse(self, x):
        return x[:, :x.size(1) - self.pad_size, :, :]


class RevDownscale(nn.Module):
    def __init__(self, scale):
        super(RevDownscale, self).__init__()
        self.scale = scale

    def forward(self, x):
        batch, x_channels, x_height, x_width = x.size()
        y_channels = x_channels * self.scale * self.scale
        assert (x_height % self.scale == 0)
        y_height = x_height // self.scale

        y = x.permute(0, 2, 3, 1)
        d2_split_seq = y.split(split_size=self.scale, dim=2)
        d2_split_seq = [t.contiguous().view(batch, y_height, y_channels) for t in d2_split_seq]
        y = torch.stack(d2_split_seq, dim=1)
        y = y.permute(0, 3, 2, 1)
        return y.contiguous()

    def inverse(self, y):
        scale_sqr = self.scale * self.scale
        batch, y_channels, y_height, y_width = y.size()
        assert (y_channels % scale_sqr == 0)
        x_channels = y_channels // scale_sqr
        x_height = y_height * self.scale
        x_width = y_width * self.scale

        x = y.permute(0, 2, 3, 1)
        x = x.contiguous().view(batch, y_height, y_width, scale_sqr, x_channels)
        d3_split_seq = x.split(split_size=self.scale, dim=3)
        d3_split_seq = [t.contiguous().view(batch, y_height, x_width, x_channels) for t in d3_split_seq]
        x = torch.stack(d3_split_seq, dim=0)
        x = x.transpose(0, 1).permute(0, 2, 1, 3, 4).contiguous().view(batch, x_height, x_width, x_channels)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class RevBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 first=False):
        super(RevBottleneck, self).__init__()
        mid_channels = out_channels // 4
        self.pad = 2 * out_channels - in_channels
        self.stride = stride
        self.inj_pad = RevInjectivePad(self.pad)
        self.psi = RevDownscale(stride)
        if self.pad != 0 and stride == 1:
            in_channels = out_channels * 2
        in_channels2 = in_channels // 2

        layers = []
        if not first:
            layers.append(nn.BatchNorm2d(in_channels2))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(
            in_channels2,
            mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False))
        layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            padding=1,
            bias=False))
        layers.append(nn.BatchNorm2d(mid_channels))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(
            mid_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False))
        self.bottleneck_block = nn.Sequential(*layers)

    def forward(self, x1, x2):
        if self.pad != 0 and self.stride == 1:
            x = torch.cat((x1, x2), dim=1)
            x = self.inj_pad.forward(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=1)

        fx2 = self.bottleneck_block(x2)
        if self.stride == 2:
            x1 = self.psi(x1)
            x2 = self.psi(x2)
        y1 = fx2 + x1
        return x2, y1

    def inverse(self, x):
        x2, y1 = x[0], x[1]
        if self.stride == 2:
            x2 = self.psi.inverse(x2)
        fx2 = - self.bottleneck_block(x2)
        x1 = fx2 + y1
        if self.stride == 2:
            x1 = self.psi.inverse(x1)
        if self.pad != 0 and self.stride == 1:
            x = torch.cat((x1, x2), dim=1)
            x = self.inj_pad.inverse(x)
            x1, x2 = torch.chunk(x, chunks=2, dim=1)
            x = (x1, x2)
        else:
            x = (x1, x2)
        return x


class iRevNet(nn.Module):
    def __init__(self,
                 channels,
                 init_block_channels,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(iRevNet, self).__init__()
        assert (in_channels > 0)
        self.in_size = in_size
        self.num_classes = num_classes

        self.init_psi = RevDownscale(scale=2)
        in_channels = init_block_channels

        block_list = nn.ModuleList()
        first = True
        for i, channels_per_stage in enumerate(channels):
            for j, out_channels in enumerate(channels_per_stage):
                stride = 2 if (j == 0) else 1
                block_list.append(RevBottleneck(
                    in_channels,
                    out_channels,
                    stride,
                    first=first))
                in_channels = 2 * out_channels
                first = False
        self.stack = block_list

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.9)

        self.features = nn.Sequential()
        self.features.add_module("final_pool", nn.AvgPool2d(
            kernel_size=7,
            stride=1))
        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

    def forward(self, x, return_out_bij=False):
        """ irevnet forward """
        x = self.init_psi(x)

        x1, x2 = torch.chunk(x, chunks=2, dim=1)
        for block in self.stack:
            x1, x2 = block(x1, x2)
        x = torch.cat((x1, x2), dim=1)
        if return_out_bij:
            out_bij = x.clone()

        x = F.relu(self.bn1(x))

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)

        if return_out_bij:
            return x, out_bij
        else:
            return x

    def inverse(self, out_bij):
        """ irevnet inverse """
        out = torch.chunk(out_bij, chunks=2, dim=1)
        for i in range(len(self.stack)):
            out = self.stack[-1-i].inverse(out)
        out = torch.cat((out[0], out[1]), dim=1)
        x = self.init_psi.inverse(out)
        return x


def get_irevnet(blocks,
                model_name=None,
                pretrained=False,
                root=os.path.join('~', '.torch', 'models'),
                **kwargs):
    """
    Create i-RevNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if blocks == 301:
        layers = [6, 16, 72, 6]
    else:
        raise ValueError("Unsupported i-RevNet with number of blocks: {}".format(blocks))

    assert (sum(layers) * 3 + 1 == blocks)

    channels_per_layers = [24, 96, 384, 1536]
    init_block_channels = 12

    channels = [[ci] * li for (ci, li) in zip(channels_per_layers, layers)]

    net = iRevNet(
        channels=channels,
        init_block_channels=init_block_channels,
        **kwargs)

    return net


def oth_irevnet301(pretrained=False, **kwargs):
    return get_irevnet(blocks=301)


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
        oth_irevnet301,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_irevnet301 or weight_count == 125120356)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
