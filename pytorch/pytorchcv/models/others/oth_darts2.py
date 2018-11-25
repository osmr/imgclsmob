import torch
import torch.nn as nn


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DARTSConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(DARTSConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=False),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0,
                bias=False),
            nn.BatchNorm2d(
                out_channels,
                affine=True),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels, affine=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=1, padding=padding, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
        )

    def forward(self, x):
        return self.op(x)


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:, :, ::self.stride, ::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels):
        super(FactorizedReduce, self).__init__()
        assert out_channels % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(in_channels, out_channels // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


OPS = {
    'max_pool_3x3': lambda channels, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda channels, stride: Identity() if stride == 1 else FactorizedReduce(channels, channels),
    'sep_conv_3x3': lambda channels, stride: SepConv(channels, channels, 3, stride, 1),
    'dil_conv_3x3': lambda channels, stride: DilConv(channels, channels, 3, stride, 2, 2),
}


class Cell(nn.Module):

    def __init__(self,
                 genotype,
                 prev_in_channels,
                 in_channels,
                 out_channels,
                 reduction,
                 reduction_prev):
        super(Cell, self).__init__()

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(prev_in_channels, out_channels)
        else:
            self.preprocess0 = DARTSConv(
                prev_in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0)
        self.preprocess1 = DARTSConv(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(
            out_channels,
            op_names,
            indices,
            concat,
            reduction)

    def _compile(self,
                 out_channels,
                 op_names,
                 indices,
                 concat,
                 reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](out_channels, stride)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            s = h1 + h2
            states += [s]
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkImageNet(nn.Module):

    def __init__(self,
                 init_block_channels,
                 num_classes,
                 layers,
                 genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, init_block_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_block_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_block_channels // 2, init_block_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_block_channels),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(init_block_channels, init_block_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(init_block_channels),
        )

        prev_in_channels, in_channels, out_channels = init_block_channels, init_block_channels, init_block_channels

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                out_channels *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                genotype,
                prev_in_channels,
                in_channels,
                out_channels,
                reduction,
                reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            prev_in_channels, in_channels = in_channels, cell.multiplier * out_channels

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, input):
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits


def oth_darts(num_classes=1000, pretrained='imagenet'):

    from collections import namedtuple
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

    net = NetworkImageNet(
        init_block_channels=48,
        num_classes=num_classes,
        layers=14,
        genotype=Genotype(
            normal=[
                ('sep_conv_3x3', 0),
                ('sep_conv_3x3', 1),
                ('sep_conv_3x3', 0),
                ('sep_conv_3x3', 1),
                ('sep_conv_3x3', 1),
                ('skip_connect', 0),
                ('skip_connect', 0),
                ('dil_conv_3x3', 2)],
            normal_concat=[2, 3, 4, 5],
            reduce=[
                ('max_pool_3x3', 0),
                ('max_pool_3x3', 1),
                ('skip_connect', 2),
                ('max_pool_3x3', 1),
                ('max_pool_3x3', 0),
                ('skip_connect', 2),
                ('skip_connect', 2),
                ('max_pool_3x3', 1)],
            reduce_concat=[2, 3, 4, 5])
    )
    return net


def load_model(net,
               file_path,
               ignore_extra=True):
    """
  Load model state dictionary from a file.

  Parameters
  ----------
  net : Module
      Network in which weights are loaded.
  file_path : str
      Path to the file.
  ignore_extra : bool, default True
      Whether to silently ignore parameters from the file that are not present in this Module.
  """
    import torch

    if ignore_extra:
        pretrained_state = torch.load(file_path)
        model_dict = net.state_dict()
        pretrained_state = {k: v for k, v in pretrained_state.items() if k in model_dict}
        net.load_state_dict(pretrained_state)
    else:
        net.load_state_dict(torch.load(file_path))


def _test():
    import numpy as np
    import torch
    from torch.autograd import Variable

    pretrained = False

    models = [
        oth_darts,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        net_params = filter(lambda p: p.requires_grad, net.parameters())
        weight_count = 0
        for param in net_params:
            weight_count += np.prod(param.size())
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_darts or weight_count == 4718752)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
