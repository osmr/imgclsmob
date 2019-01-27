import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, dropRate=0.0):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, dropRate)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class DenseNet3(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=12,
                 reduction=0.5, bottleneck=True, dropRate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n/2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes*reduction)), dropRate=dropRate)
        in_planes = int(math.floor(in_planes*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, dropRate)
        in_planes = int(in_planes+n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_classes)
        self.in_planes = in_planes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)


def oth_densenet40_k12_cifar10(pretrained=False, **kwargs):
    model = DenseNet3(depth=40, num_classes=10, growth_rate=12, bottleneck=False, **kwargs)
    return model


def oth_densenet40_k12_cifar100(pretrained=False, **kwargs):
    model = DenseNet3(depth=40, num_classes=100, growth_rate=12, bottleneck=False, **kwargs)
    return model


def oth_densenet100_k12_cifar10(pretrained=False, **kwargs):
    model = DenseNet3(depth=100, num_classes=10, growth_rate=12, bottleneck=False, **kwargs)
    return model


def oth_densenet100_k12_cifar100(pretrained=False, **kwargs):
    model = DenseNet3(depth=100, num_classes=100, growth_rate=12, bottleneck=False, **kwargs)
    return model


def oth_densenet100_k12_bc_cifar10(pretrained=False, **kwargs):
    model = DenseNet3(depth=100, num_classes=10, growth_rate=12, bottleneck=True, **kwargs)
    return model


def oth_densenet100_k12_bc_cifar100(pretrained=False, **kwargs):
    model = DenseNet3(depth=100, num_classes=100, growth_rate=12, bottleneck=True, **kwargs)
    return model


def oth_densenet250_k24_bc_cifar10(pretrained=False, **kwargs):
    model = DenseNet3(depth=250, num_classes=10, growth_rate=24, bottleneck=True, **kwargs)
    return model


def oth_densenet250_k24_bc_cifar100(pretrained=False, **kwargs):
    model = DenseNet3(depth=250, num_classes=100, growth_rate=24, bottleneck=True, **kwargs)
    return model


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
        oth_densenet40_k12_cifar10,
        oth_densenet40_k12_cifar100,
        oth_densenet100_k12_cifar10,
        oth_densenet100_k12_cifar100,
        oth_densenet100_k12_bc_cifar10,
        oth_densenet100_k12_bc_cifar100,
        oth_densenet250_k24_bc_cifar10,
        oth_densenet250_k24_bc_cifar100,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_densenet40_k12_cifar10 or weight_count == 599050)
        assert (model != oth_densenet40_k12_cifar100 or weight_count == 622360)
        assert (model != oth_densenet100_k12_bc_cifar10 or weight_count == 769162)
        assert (model != oth_densenet100_k12_bc_cifar100 or weight_count == 800032)
        assert (model != oth_densenet250_k24_bc_cifar10 or weight_count == 15324406)
        assert (model != oth_densenet250_k24_bc_cifar100 or weight_count == 15480556)

        x = Variable(torch.randn(1, 3, 32, 32))
        y = net(x)
        #assert (tuple(y.size()) == (1, 10))


if __name__ == "__main__":
    _test()
