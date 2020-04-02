"""
    LFFD for face detection, implemented in Chainer.
    Original paper: 'LFFD: A Light and Fast Face Detector for Edge Devices,' https://arxiv.org/abs/1904.10633.
"""

__all__ = ['LFFD', 'lffd20x5s320v2_widerface', 'lffd25x8s560v1_widerface']

import os
import chainer.functions as F
from chainer import Chain
from chainer.serializers import load_npz
from .common import conv3x3, conv1x1_block, conv3x3_block, Concurrent, MultiOutputSequential, ParallelConcurent
from .resnet import ResUnit
from .preresnet import PreResUnit


class LffdDetectionBranch(Chain):
    """
    LFFD specific detection branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_bias,
                 use_bn,
                 **kwargs):
        super(LffdDetectionBranch, self).__init__(**kwargs)
        with self.init_scope():
            self.conv1 = conv1x1_block(
                in_channels=in_channels,
                out_channels=in_channels,
                use_bias=use_bias,
                use_bn=use_bn)
            self.conv2 = conv1x1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                use_bias=use_bias,
                use_bn=use_bn,
                activation=None)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LffdDetectionBlock(Chain):
    """
    LFFD specific detection block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    mid_channels : int
        Number of middle channels.
    use_bias : bool
        Whether the layer uses a bias vector.
    use_bn : bool
        Whether to use BatchNorm layer.
    """
    def __init__(self,
                 in_channels,
                 mid_channels,
                 use_bias,
                 use_bn,
                 **kwargs):
        super(LffdDetectionBlock, self).__init__(**kwargs)
        with self.init_scope():
            self.conv = conv1x1_block(
                in_channels=in_channels,
                out_channels=mid_channels,
                use_bias=use_bias,
                use_bn=use_bn)
            self.branches = Concurrent()
            with self.branches.init_scope():
                setattr(self.branches, "bbox_branch", LffdDetectionBranch(
                    in_channels=mid_channels,
                    out_channels=4,
                    use_bias=use_bias,
                    use_bn=use_bn))
                setattr(self.branches, "score_branch", LffdDetectionBranch(
                    in_channels=mid_channels,
                    out_channels=2,
                    use_bias=use_bias,
                    use_bn=use_bn))

    def __call__(self, x):
        x = self.conv(x)
        x = self.branches(x)
        return x


class LFFD(Chain):
    """
    LFFD model from 'LFFD: A Light and Fast Face Detector for Edge Devices,' https://arxiv.org/abs/1904.10633.

    Parameters:
    ----------
    enc_channels : list of int
        Number of output channels for each encoder stage.
    dec_channels : int
        Number of output channels for each decoder stage.
    init_block_channels : int
        Number of output channels for the initial encoder unit.
    layers : list of int
        Number of units in each encoder stage.
    int_bends : list of int
        Number of internal bends for each encoder stage.
    use_preresnet : bool
        Whether to use PreResnet backbone instead of ResNet.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (640, 640)
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
                 in_size=(640, 640),
                 **kwargs):
        super(LFFD, self).__init__(**kwargs)
        self.in_size = in_size
        unit_class = PreResUnit if use_preresnet else ResUnit
        use_bias = True
        use_bn = False

        with self.init_scope():
            self.encoder = MultiOutputSequential(return_last=False)
            with self.encoder.init_scope():
                setattr(self.encoder, "init_block", conv3x3_block(
                    in_channels=in_channels,
                    out_channels=init_block_channels,
                    stride=2,
                    pad=0,
                    use_bias=use_bias,
                    use_bn=use_bn))
                in_channels = init_block_channels
                for i, channels_per_stage in enumerate(enc_channels):
                    layers_per_stage = layers[i]
                    int_bends_per_stage = int_bends[i]
                    stage = MultiOutputSequential(multi_output=False, dual_output=True)
                    with stage.init_scope():
                        setattr(stage, "trans{}".format(i + 1), conv3x3(
                            in_channels=in_channels,
                            out_channels=channels_per_stage,
                            stride=2,
                            pad=0,
                            use_bias=use_bias))
                        for j in range(layers_per_stage):
                            unit = unit_class(
                                in_channels=channels_per_stage,
                                out_channels=channels_per_stage,
                                stride=1,
                                use_bias=use_bias,
                                use_bn=use_bn,
                                bottleneck=False)
                            if layers_per_stage - j <= int_bends_per_stage:
                                unit.do_output = True
                            setattr(stage, "unit{}".format(j + 1), unit)
                        final_activ = F.relu
                        final_activ.do_output = True
                        setattr(stage, "final_activ", final_activ)
                        stage.do_output2 = True
                    in_channels = channels_per_stage
                    setattr(self.encoder, "stage{}".format(i + 1), stage)

            self.decoder = ParallelConcurent()
            with self.decoder.init_scope():
                k = 0
                for i, channels_per_stage in enumerate(enc_channels):
                    layers_per_stage = layers[i]
                    int_bends_per_stage = int_bends[i]
                    for j in range(layers_per_stage):
                        if layers_per_stage - j <= int_bends_per_stage:
                            setattr(self.decoder, "unit{}".format(k + 1), LffdDetectionBlock(
                                in_channels=channels_per_stage,
                                mid_channels=dec_channels,
                                use_bias=use_bias,
                                use_bn=use_bn))
                            k += 1
                    setattr(self.decoder, "unit{}".format(k + 1), LffdDetectionBlock(
                        in_channels=channels_per_stage,
                        mid_channels=dec_channels,
                        use_bias=use_bias,
                        use_bn=use_bn))
                    k += 1

    def __call__(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_lffd(blocks,
             use_preresnet,
             model_name=None,
             pretrained=False,
             root=os.path.join("~", ".chainer", "models"),
             **kwargs):
    """
    Create LFFD model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    use_preresnet : bool
        Whether to use PreResnet backbone instead of ResNet.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
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
        from .model_store import get_model_file
        load_npz(
            file=get_model_file(
                model_name=model_name,
                local_model_store_dir_path=root),
            obj=net)

    return net


def lffd20x5s320v2_widerface(**kwargs):
    """
    LFFD-320-20L-5S-V2 model for WIDER FACE from 'LFFD: A Light and Fast Face Detector for Edge Devices,'
    https://arxiv.org/abs/1904.10633.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_lffd(blocks=20, use_preresnet=True, model_name="lffd20x5s320v2_widerface", **kwargs)


def lffd25x8s560v1_widerface(**kwargs):
    """
    LFFD-560-25L-8S-V1 model for WIDER FACE from 'LFFD: A Light and Fast Face Detector for Edge Devices,'
    https://arxiv.org/abs/1904.10633.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.chainer/models'
        Location for keeping the model parameters.
    """
    return get_lffd(blocks=25, use_preresnet=False, model_name="lffd25x8s560v1_widerface", **kwargs)


def _test():
    import numpy as np
    import chainer

    chainer.global_config.train = False

    in_size = (640, 640)
    pretrained = False

    models = [
        (lffd20x5s320v2_widerface, 5),
        (lffd25x8s560v1_widerface, 8),
    ]

    for model, num_outs in models:

        net = model(pretrained=pretrained)
        weight_count = net.count_params()
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != lffd20x5s320v2_widerface or weight_count == 1520606)
        assert (model != lffd25x8s560v1_widerface or weight_count == 2290608)

        batch = 14
        x = np.zeros((batch, 3, in_size[0], in_size[1]), np.float32)
        y = net(x)
        assert (len(y) == num_outs)


if __name__ == "__main__":
    _test()
