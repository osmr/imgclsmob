import torch.nn as nn
import math
import torch

__all__ = ['dia_resnet20_cifar10']


class FirstAmp(nn.Module):

    def __init__(self,
                 in_features,
                 out_features):
        super(FirstAmp, self).__init__()
        mid_features = in_features // 4

        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=mid_features)
        self.activ = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(
            in_features=mid_features,
            out_features=out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)
        return x


class DIALSTMCell(nn.Module):

    def __init__(self,
                 in_x_features,
                 in_h_features,
                 num_layers,
                 dropout=0.1):
        super(DIALSTMCell, self).__init__()
        out_features = 4 * in_h_features

        self.x_amps = nn.Sequential()
        self.h_amps = nn.Sequential()
        for i in range(num_layers):
            cell_class = FirstAmp if i == 0 else nn.Linear
            self.x_amps.add_module("amp{}".format(i + 1), cell_class(
                in_features=in_x_features,
                out_features=out_features))
            self.h_amps.add_module("amp{}".format(i + 1), cell_class(
                in_features=in_h_features,
                out_features=out_features))
            in_x_features = in_h_features
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, hx, cx):
        hy = []
        cy = []
        for i in range(len(self.x_amps._modules)):
            hx_i = hx[i]
            cx_i = cx[i]
            gates = self.x_amps[i](x) + self.h_amps[i](hx_i)
            i_gate, f_gate, c_gate, o_gate = gates.chunk(chunks=4, dim=1)
            i_gate = torch.sigmoid(i_gate)
            f_gate = torch.sigmoid(f_gate)
            c_gate = torch.tanh(c_gate)
            o_gate = torch.sigmoid(o_gate)
            ncx = (f_gate * cx_i) + (i_gate * c_gate)
            nhx = o_gate * torch.sigmoid(ncx)
            cy.append(ncx)
            hy.append(nhx)
            x = self.dropout(nhx)
        hy = torch.stack(hy, dim=0)
        cy = torch.stack(cy, dim=0)
        return hy, cy


def conv3x3(in_planes,
            out_planes,
            stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        y = self.pool(x)

        x += identity
        x = self.relu(x)

        return x, y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        return x, identity


class Attention(nn.Module):
    def __init__(self,
                 module_list,
                 input_size,
                 hidden_size):
        super(Attention, self).__init__()

        self.module_list = module_list
        self.lstm = DIALSTMCell(
            in_x_features=input_size,
            in_h_features=hidden_size,
            num_layers=1)
        self.pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        for i, unit in enumerate(self.module_list):
            x, identity = unit(x)
            w = self.pool(x)
            w = w.view(w.size(0), -1)
            if i == 0:
                hi = torch.zeros_like(w).unsqueeze(dim=0)
                ci = torch.zeros_like(w).unsqueeze(dim=0)
            hi, ci = self.lstm(w, hi, ci)
            x = x * hi[-1].unsqueeze(dim=2).unsqueeze(dim=3)
            x += identity
            x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = Attention(
            module_list=self._make_layer(block, 16, n),
            input_size=64,
            hidden_size=64)
        self.layer2 = Attention(
            module_list=self._make_layer(block, 32, n, stride=2),
            input_size=128,
            hidden_size=128)
        self.layer3 = Attention(
            module_list=self._make_layer(block, 64, n, stride=2),
            input_size=256,
            hidden_size=256)

        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = nn.ModuleList([])

        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return layers

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)    # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def dia_resnet20_cifar10(pretrained=False, **kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(20, 10, 'Bottleneck', **kwargs)


def dia_resnet56_cifar10(pretrained=False, **kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(56, 10, 'Bottleneck', **kwargs)


def dia_resnet110_cifar10(pretrained=False, **kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(110, 10, 'Bottleneck', **kwargs)


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
        dia_resnet20_cifar10,
        dia_resnet56_cifar10,
        dia_resnet110_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained).cuda()

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != dia_resnet20_cifar10 or weight_count == 439226)
        assert (model != dia_resnet56_cifar10 or weight_count == 810170)
        assert (model != dia_resnet110_cifar10 or weight_count == 1366586)

        x = torch.randn(1, 3, 32, 32).cuda()
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 10))


if __name__ == "__main__":
    _test()
