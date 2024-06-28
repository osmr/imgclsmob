"""
    LFFD for face detection, implemented in PyTorch.
    Original paper: 'LFFD: A Light and Fast Face Detector for Edge Devices,' https://arxiv.org/abs/1904.10633.
"""

__all__ = ['LFFD', 'lffd20x5s320v2_widerface', 'lffd25x8s560v1_widerface']

import os
import torch.nn as nn
from .common import (conv3x3, conv1x1_block, conv3x3_block, Concurrent, MultiOutputSequential, ParallelConcurent,
                     calc_net_weights)
from .resnet import ResUnit
from .preresnet import PreResUnit


class LffdDetectionBranch(nn.Module):
    """
    LFFD specific detection branch.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias,
                 use_bn):
        super(LffdDetectionBranch, self).__init__()
        self.conv1 = conv1x1_block(
            in_channels=in_channels,
            out_channels=in_channels,
            bias=bias,
            use_bn=use_bn)
        self.conv2 = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            use_bn=use_bn,
            activation=None)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LffdDetectionBlock(nn.Module):
    """
    LFFD specific detection block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 bias,
                 use_bn):
        super(LffdDetectionBlock, self).__init__()
        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            bias=bias,
            use_bn=use_bn)
        self.branches = Concurrent()
        self.branches.add_module("bbox_branch", LffdDetectionBranch(
            in_channels=mid_channels,
            out_channels=4,
            bias=bias,
            use_bn=use_bn))
        self.branches.add_module("score_branch", LffdDetectionBranch(
            in_channels=mid_channels,
            out_channels=2,
            bias=bias,
            use_bn=use_bn))

    def forward(self, x):
        x = self.conv(x)
        x = self.branches(x)
        return x


class LFFD(nn.Module):
    """
    LFFD model from 'LFFD: A Light and Fast Face Detector for Edge Devices,' https://arxiv.org/abs/1904.10633.

    Parameters
    ----------
    enc_channels : list(int)
        Number of output channels for each encoder stage.
    dec_channels : int
        Number of output channels for each decoder stage.
    init_block_channels : int
        Number of output channels for the initial encoder unit.
    layers : list(int)
        Number of units in each encoder stage.
    int_bends : list(int)
        Number of internal bends for each encoder stage.
    use_preresnet : bool
        Whether to use PreResnet backbone instead of ResNet.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple(int, int), default (640, 640)
        Spatial size of the expected input image.
    """
    def __init__(self,
                 enc_channels,
                 dec_channels,
                 init_block_channels,
                 layers,
                 int_bends,
                 use_preresnet,
                 in_channels=3,
                 in_size=(640, 640)):
        super(LFFD, self).__init__()
        self.in_size = in_size
        unit_class = PreResUnit if use_preresnet else ResUnit
        bias = True
        use_bn = False

        self.encoder = MultiOutputSequential(return_last=False)
        self.encoder.add_module("init_block", conv3x3_block(
            in_channels=in_channels,
            out_channels=init_block_channels,
            stride=2,
            padding=0,
            bias=bias,
            use_bn=use_bn))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(enc_channels):
            layers_per_stage = layers[i]
            int_bends_per_stage = int_bends[i]
            stage = MultiOutputSequential(multi_output=False, dual_output=True)
            stage.add_module("trans{}".format(i + 1), conv3x3(
                in_channels=in_channels,
                out_channels=channels_per_stage,
                stride=2,
                padding=0,
                bias=bias))
            for j in range(layers_per_stage):
                unit = unit_class(
                    in_channels=channels_per_stage,
                    out_channels=channels_per_stage,
                    stride=1,
                    bias=bias,
                    use_bn=use_bn,
                    bottleneck=False)
                if layers_per_stage - j <= int_bends_per_stage:
                    unit.do_output = True
                stage.add_module("unit{}".format(j + 1), unit)
            final_activ = nn.ReLU(inplace=True)
            final_activ.do_output = True
            stage.add_module("final_activ", final_activ)
            stage.do_output2 = True
            in_channels = channels_per_stage
            self.encoder.add_module("stage{}".format(i + 1), stage)

        self.decoder = ParallelConcurent()
        k = 0
        for i, channels_per_stage in enumerate(enc_channels):
            layers_per_stage = layers[i]
            int_bends_per_stage = int_bends[i]
            for j in range(layers_per_stage):
                if layers_per_stage - j <= int_bends_per_stage:
                    self.decoder.add_module("unit{}".format(k + 1), LffdDetectionBlock(
                        in_channels=channels_per_stage,
                        mid_channels=dec_channels,
                        bias=bias,
                        use_bn=use_bn))
                    k += 1
            self.decoder.add_module("unit{}".format(k + 1), LffdDetectionBlock(
                in_channels=channels_per_stage,
                mid_channels=dec_channels,
                bias=bias,
                use_bn=use_bn))
            k += 1

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_lffd(blocks,
             use_preresnet,
             model_name=None,
             pretrained=False,
             root: str = os.path.join("~", ".torch", "models"),
             **kwargs):
    """
    Create LFFD model with specific parameters.

    Parameters
    ----------
    blocks : int
        Number of blocks.
    use_preresnet : bool
        Whether to use PreResnet backbone instead of ResNet.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if blocks == 20:
        layers = [3, 1, 1, 1, 1]
        enc_channels = [64, 64, 64, 128, 128]
        int_bends = [0, 0, 0, 0, 0]
    elif blocks == 25:
        layers = [4, 2, 1, 3]
        enc_channels = [64, 64, 128, 128]
        int_bends = [1, 1, 0, 2]
    else:
        raise ValueError("Unsupported LFFD with number of blocks: {}".format(blocks))

    dec_channels = 128
    init_block_channels = 64

    net = LFFD(
        enc_channels=enc_channels,
        dec_channels=dec_channels,
        init_block_channels=init_block_channels,
        layers=layers,
        int_bends=int_bends,
        use_preresnet=use_preresnet,
        **kwargs)

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError("Parameter `model_name` should be properly initialized for loading pretrained model.")
        from .model_store import download_model
        download_model(
            net=net,
            model_name=model_name,
            local_model_store_dir_path=root)

    return net


def lffd20x5s320v2_widerface(**kwargs):
    """
    LFFD-320-20L-5S-V2 model for WIDER FACE from 'LFFD: A Light and Fast Face Detector for Edge Devices,'
    https://arxiv.org/abs/1904.10633.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_lffd(blocks=20, use_preresnet=True, model_name="lffd20x5s320v2_widerface", **kwargs)


def lffd25x8s560v1_widerface(**kwargs):
    """
    LFFD-560-25L-8S-V1 model for WIDER FACE from 'LFFD: A Light and Fast Face Detector for Edge Devices,'
    https://arxiv.org/abs/1904.10633.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_lffd(blocks=25, use_preresnet=False, model_name="lffd25x8s560v1_widerface", **kwargs)


def _test():
    import torch

    in_size = (640, 640)
    pretrained = False

    models = [
        (lffd20x5s320v2_widerface, 5),
        (lffd25x8s560v1_widerface, 8),
    ]

    for model, num_outs in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = calc_net_weights(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lffd20x5s320v2_widerface or weight_count == 1520606)
        assert (model != lffd25x8s560v1_widerface or weight_count == 2290608)

        batch = 14
        x = torch.randn(batch, 3, in_size[0], in_size[1])
        y = net(x)
        assert (len(y) == num_outs)


if __name__ == "__main__":
    _test()
