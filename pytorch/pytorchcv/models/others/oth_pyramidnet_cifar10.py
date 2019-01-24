import torch.nn as nn
import torch.nn.functional as F


# code from https://github.com/KellerJordan/ResNet-PyTorch-CIFAR10/blob/master/model.py
class IdentityPadding(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(IdentityPadding, self).__init__()

        if stride == 2:
            self.pooling = nn.AvgPool2d(kernel_size=2, stride=2, ceil_mode=True)
        else:
            self.pooling = None
            
        self.add_channels = out_channels - in_channels
    
    def forward(self, x):
        out = F.pad(x, (0, 0, 0, 0, 0, self.add_channels))
        if self.pooling is not None:
            out = self.pooling(out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                                stride=stride, padding=1, bias=False)      
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                                stride=1, padding=1, bias=False)    
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.down_sample = IdentityPadding(in_channels, out_channels, stride)
            
        self.stride = stride

    def forward(self, x):
        shortcut = self.down_sample(x)
        out = self.bn1(x)
        out = self.conv1(out)        
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
       
        out += shortcut
        return out


class PyramidNet(nn.Module):
    def __init__(self, num_layers, alpha, block, num_classes=10):
        super(PyramidNet, self).__init__()   	
        self.in_channels = 16
        
        # num_layers = (110 - 2)/6 = 18
        self.num_layers = num_layers
        self.addrate = alpha / (3*self.num_layers*1.0)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # feature map size = 32x32
        self.layer1 = self.get_layers(block, stride=1)
        # feature map size = 16x16
        self.layer2 = self.get_layers(block, stride=2)
        # feature map size = 8x8
        self.layer3 = self.get_layers(block, stride=2)

        self.out_channels = int(round(self.out_channels))
        self.bn_out= nn.BatchNorm2d(self.out_channels)
        self.relu_out = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc_out = nn.Linear(self.out_channels, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_layers(self, block, stride):
        layers_list = []
        for _ in range(self.num_layers - 1):
            self.out_channels = self.in_channels + self.addrate
            layers_list.append(block(int(round(self.in_channels)), 
                                     int(round(self.out_channels)), 
                                     stride))
            self.in_channels = self.out_channels
            stride=1

        return nn.Sequential(*layers_list)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn_out(x)
        x = self.relu_out(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x


def oth_pyramidnet_cifar10(pretrained=False):
    block = ResidualBlock
    model = PyramidNet(num_layers=18, alpha=48, block=block)
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
        oth_pyramidnet_cifar10,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_pyramidnet_cifar10 or weight_count == 1558881)

        x = Variable(torch.randn(1, 3, 32, 32))
        y = net(x)
        assert (tuple(y.size()) == (1, 10))


if __name__ == "__main__":
    _test()
