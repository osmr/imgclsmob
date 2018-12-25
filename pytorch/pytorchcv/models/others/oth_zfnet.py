import torch.nn as nn


class ZFNet(nn.Module):

    """Original ZFNET Architecture"""

    def __init__(self, channels=3, num_classes=1000):
        super(ZFNet, self).__init__()
        self.channels = channels
        self.class_count = num_classes

        self.conv_net = self.get_conv_net()
        self.fc_net = self.get_fc_net()

    def get_conv_net(self):
        layers = []

        # in_channels = self.channels, out_channels = 96
        # kernel_size = 7x7, stride = 2
        layer = nn.Conv2d(
            self.channels, 96, kernel_size=7, stride=2, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))

        # in_channels = 96, out_channels = 256
        # kernel_size = 5x5, stride = 2
        layer = nn.Conv2d(96, 256, kernel_size=5, stride=2)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(nn.LocalResponseNorm(5))

        # in_channels = 256, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))

        # in_channels = 384, out_channels = 384
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))

        # in_channels = 384, out_channels = 256
        # kernel_size = 3x3, stride = 1
        layer = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        return nn.Sequential(*layers)

    def get_fc_net(self):
        layers = []

        # in_channels = 9216 -> output of self.conv_net
        # out_channels = 4096
        layer = nn.Linear(9216, 4096)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())

        # in_channels = 4096
        # out_channels = self.class_count
        layer = nn.Linear(4096, self.class_count)
        nn.init.normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Dropout())

        return nn.Sequential(*layers)

    def forward(self, x):
        y = self.conv_net(x)
        y = y.view(-1, 9216)
        y = self.fc_net(y)
        return y


def oth_zfnet(pretrained=False, **kwargs):
    return ZFNet(**kwargs)


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
        oth_zfnet,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_zfnet or weight_count == 45576296)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
