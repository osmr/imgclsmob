"""
    SegNet for image segmentation, implemented in PyTorch.
    Original paper: 'SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation,'
    https://arxiv.org/abs/1511.00561.
"""

import torch
import torch.nn as nn
from common import conv3x3, conv3x3_block, DualPathSequential

__all__ = ["SegNet"]


class SegNet(nn.Module):
    def __init__(self,
                 bn_eps=1e-5,
                 aux=False,
                 fixed_size=False,
                 in_channels=3,
                 in_size=(1024, 2048),
                 num_classes=19):
        super(SegNet, self).__init__()
        bias = True

        layers = [[3, 3, 4, 4, 4], [4, 4, 4, 3, 2]]
        channels = [[64, 128, 256, 512, 512], [512, 256, 128, 64, 64]]

        for i, out_channels in enumerate(channels[0]):
            stage = nn.Sequential()
            for j in range(layers[0][i]):
                if j == layers[0][i] - 1:
                    unit = nn.MaxPool2d(
                        kernel_size=2,
                        stride=2,
                        return_indices=True)
                else:
                    unit = conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=bias)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            setattr(self, "down_stage{}".format(i + 1), stage)

        for i, channels_per_stage in enumerate(channels[1]):
            stage = DualPathSequential(
                return_two=False,
                last_ordinals=(layers[1][i] - 1),
                dual_path_scheme=(lambda module, x1, x2: (module(x1, x2), x2)))
            for j in range(layers[1][i]):
                if j == layers[1][i] - 1:
                    out_channels = channels_per_stage
                else:
                    out_channels = in_channels
                if j == 0:
                    unit = nn.MaxUnpool2d(
                        kernel_size=2,
                        stride=2)
                else:
                    unit = conv3x3_block(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        bias=bias)
                stage.add_module("unit{}".format(j + 1), unit)
                in_channels = out_channels
            setattr(self, "up_stage{}".format(i + 1), stage)

        self.head = conv3x3(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=bias)

    def forward(self, x):
        x, max_indices1 = self.down_stage1(x)
        x, max_indices2 = self.down_stage2(x)
        x, max_indices3 = self.down_stage3(x)
        x, max_indices4 = self.down_stage4(x)
        x, max_indices5 = self.down_stage5(x)

        x = self.up_stage1(x, max_indices5)
        x = self.up_stage2(x, max_indices4)
        x = self.up_stage3(x, max_indices3)
        x = self.up_stage4(x, max_indices2)
        x = self.up_stage5(x, max_indices1)

        x = self.head(x)
        return x

    def load_from_segnet(self, model_path):
        s_dict = self.state_dict()# create a copy of the state dict
        th = torch.load(model_path).state_dict() # load the weigths
        # for name in th:
            # s_dict[corresp_name[name]] = th[name]
        self.load_state_dict(th)


def oth_segnet_cityscapes(num_classes=19, pretrained=False, **kwargs):
    return SegNet(num_classes=num_classes, **kwargs)


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    pretrained = False
    # fixed_size = True
    in_size = (1024, 2048)
    classes = 19

    models = [
        oth_segnet_cityscapes,
    ]

    for model in models:

        # from torchsummary import summary
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # net = SegNet(num_classes=19).to(device)
        # summary(net, (3, 512, 1024))

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_segnet_cityscapes or weight_count == 29453971)

        batch = 4
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size()) == (batch, classes, in_size[0], in_size[1]))


if __name__ == "__main__":
    _test()
