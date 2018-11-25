import torch
import torch.nn as nn


class Identity(nn.Module):
    """
    Identity block.
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def conv1x1(in_channels,
            out_channels,
            stride=1,
            bias=False):
    """
    Convolution 1x1 layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        bias=bias)


class DwsConv(nn.Module):
    """
    Standard dilated depthwise separable convolution block with.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layers use a bias vector.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 bias=False):
        super(DwsConv, self).__init__()
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)
        self.pw_conv = conv1x1(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias)

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class DartsConv(nn.Module):
    """
    DARTS specific convolution block.

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
    activate : bool, default True
        Whether activate the convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activate=True):
        super(DartsConv, self).__init__()
        self.activate = activate

        if self.activate:
            self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        if self.activate:
            x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


def darts_conv1x1(in_channels,
                  out_channels,
                  activate=True):
    """
    1x1 version of the DARTS specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return DartsConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        activate=activate)


def darts_conv3x3(in_channels,
                  out_channels,
                  activate=True):
    """
    3x3 version of the DARTS specific convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return DartsConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
        activate=activate)


class DartsDwsConv(nn.Module):
    """
    DARTS specific dilated convolution block.

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
    dilation : int or tuple/list of 2 int
        Dilation value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation):
        super(DartsDwsConv, self).__init__()
        self.activ = nn.ReLU(inplace=True)
        self.conv = DwsConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.activ(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DartsDwsBranch(nn.Module):
    """
    DARTS specific block with depthwise separable convolution layers.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 2 int
        Convolution window size.
    stride : int or tuple/list of 2 int
        Stride of the convolution.
    padding : int or tuple/list of 2 int
        Padding value for convolution layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding):
        super(DartsDwsBranch, self).__init__()
        mid_channels = in_channels

        self.conv1 = DartsDwsConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=1)
        self.conv2 = DartsDwsConv(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DartsReduceBranch(nn.Module):
    """
    DARTS specific factorized reduce block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 2
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=2):
        super(DartsReduceBranch, self).__init__()
        assert (out_channels % 2 == 0)
        mid_channels = out_channels // 2

        self.activ = nn.ReLU(inplace=True)
        self.conv1 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv2 = conv1x1(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=stride)
        self.bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, x):
        x = self.activ(x)
        x1 = self.conv1(x)
        x = x[:, :, 1:, 1:].contiguous()
        x2 = self.conv2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.bn(x)
        return x


def darts_maxpool3x3(channels,
                     stride):
    """
    DARTS specific 3x3 Max pooling layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels. Unused parameter.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    assert (channels > 0)
    return nn.MaxPool2d(
        kernel_size=3,
        stride=stride,
        padding=1)


def darts_skip_connection(channels,
                          stride):
    """
    DARTS specific skip connection layer.

    Parameters:
    ----------
    channels : int
        Number of input/output channels. Unused parameter.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    assert (channels > 0)
    if stride == 1:
        return Identity()
    else:
        assert (stride == 2)
        return DartsReduceBranch(
            in_channels=channels,
            out_channels=channels,
            stride=stride)


def darts_dws_conv3x3(channels,
                      stride):
    """
    3x3 version of DARTS specific dilated convolution block.

    Parameters:
    ----------
    channels : int
        Number of input/output channels. Unused parameter.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return DartsDwsConv(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=2,
        dilation=2)


def darts_dws_branch3x3(channels,
                        stride):
    """
    3x3 version of DARTS specific dilated convolution branch.

    Parameters:
    ----------
    channels : int
        Number of input/output channels. Unused parameter.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    return DartsDwsBranch(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        stride=stride,
        padding=1)


OPS = {
    'max_pool_3x3': darts_maxpool3x3,
    'skip_connect': darts_skip_connection,
    'dil_conv_3x3': darts_dws_conv3x3,
    'sep_conv_3x3': darts_dws_branch3x3,
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
            self.preprocess0 = DartsReduceBranch(
                in_channels=prev_in_channels,
                out_channels=out_channels)
        else:
            self.preprocess0 = darts_conv1x1(
                in_channels=prev_in_channels,
                out_channels=out_channels)
        self.preprocess1 = darts_conv1x1(
            in_channels=in_channels,
            out_channels=out_channels)

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
            darts_conv3x3(
                in_channels=3,
                out_channels=init_block_channels // 2,
                activate=False),
            darts_conv3x3(
                in_channels=init_block_channels // 2,
                out_channels=init_block_channels,
                activate=True))

        self.stem1 = darts_conv3x3(
            in_channels=init_block_channels,
            out_channels=init_block_channels,
            activate=True)

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

        self.final_pool = nn.AvgPool2d(
            kernel_size=7,
            stride=1)
        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

    def forward(self, x):
        x_prev = self.stem0(x)
        x = self.stem1(x_prev)
        for i, cell in enumerate(self.cells):
            x_prev, x = x, cell(x_prev, x)
        x = self.final_pool(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


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
