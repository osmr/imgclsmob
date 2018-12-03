import os
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Hourglass(nn.Module):
    """
    A hourglass block.

    Parameters:
    ----------
    down_seq : nn.Sequential
        Down modules as sequential.
    up_seq : nn.Sequential
        Up modules as sequential.
    skip_seq : nn.Sequential
        Skip connection modules as sequential.
    merge_type : str, default 'add'
        Type of concatenation of up and skip outputs.
    return_first_skip : bool, default False
        Whether return the first skip connection output. Used in ResAttNet.
    """
    def __init__(self,
                 down_seq,
                 up_seq,
                 skip_seq,
                 merge_type="add",
                 return_first_skip=False):
        super(Hourglass, self).__init__()
        assert (len(up_seq) == len(down_seq))
        assert (len(skip_seq) == len(down_seq))
        assert (merge_type in ["add"])
        self.down_seq = down_seq
        self.up_seq = up_seq
        self.skip_seq = skip_seq
        self.merge_type = merge_type
        self.return_first_skip = return_first_skip
        self.depth = len(down_seq)

    def forward(self, x, **kwargs):
        y = None
        down_outs = [x]
        for down_module in self.down_seq._modules.values():
            x = down_module(x)
            down_outs.append(x)
        for i in range(len(down_outs)):
            if i != 0:
                y = down_outs[self.depth - i]
                skip_module = self.skip_seq[self.depth - i]
                y = skip_module(y)
                if (y is not None) and (self.merge_type == "add"):
                    x = x + y
            if i != len(down_outs) - 1:
                up_module = self.up_seq[self.depth - 1 - i]
                x = up_module(x)
        if self.return_first_skip:
            return x, y
        else:
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


class ConvBlock(nn.Module):
    """
    Standard convolution block with Batch normalization and ReLU/ReLU6 activation.

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
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 groups=1,
                 bias=False,
                 act_type="relu",
                 activate=True):
        super(ConvBlock, self).__init__()
        self.activate = activate

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        if self.activate:
            if act_type == "relu":
                self.activ = nn.ReLU(inplace=True)
            elif act_type == "relu6":
                self.activ = nn.ReLU6(inplace=True)
            else:
                raise NotImplementedError()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        return x


def conv7x7_block(in_channels,
                  out_channels,
                  stride=1,
                  padding=3,
                  bias=False,
                  act_type="relu",
                  activate=True):
    """
    7x7 version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 3
        Padding value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    act_type : str, default 'relu'
        Name of activation function to use.
    activate : bool, default True
        Whether activate the convolution block.
    """
    return ConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        bias=bias,
        act_type=act_type,
        activate=activate)


class PreConvBlock(nn.Module):
    """
    Convolution block with Batch normalization and ReLU pre-activation.

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
    return_preact : bool, default False
        Whether return pre-activation. It's used by PreResNet.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 return_preact=False):
        super(PreConvBlock, self).__init__()
        self.return_preact = return_preact

        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        if self.return_preact:
            x_pre_activ = x
        x = self.conv(x)
        if self.return_preact:
            return x, x_pre_activ
        else:
            return x


def pre_conv1x1_block(in_channels,
                      out_channels,
                      stride=1,
                      return_preact=False):
    """
    1x1 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    return_preact : bool, default False
        Whether return pre-activation.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=0,
        return_preact=return_preact)


def pre_conv3x3_block(in_channels,
                      out_channels,
                      stride,
                      return_preact=False):
    """
    3x3 version of the pre-activated convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    return_preact : bool, default False
        Whether return pre-activation.
    """
    return PreConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        return_preact=return_preact)


class PreResBottleneck(nn.Module):
    """
    PreResNet bottleneck block for residual path in PreResNet unit.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride):
        super(PreResBottleneck, self).__init__()
        mid_channels = out_channels // 4

        self.conv1 = pre_conv1x1_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            return_preact=True)
        self.conv2 = pre_conv3x3_block(
            in_channels=mid_channels,
            out_channels=mid_channels,
            stride=stride)
        self.conv3 = pre_conv1x1_block(
            in_channels=mid_channels,
            out_channels=out_channels)

    def forward(self, x):
        x, x_pre_activ = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x, x_pre_activ


class ResBlock(nn.Module):
    """
    Residual block with pre-activation.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        super(ResBlock, self).__init__()
        self.resize_identity = (in_channels != out_channels) or (stride != 1)

        self.body = PreResBottleneck(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride)
        if self.resize_identity:
            self.identity_conv = conv1x1(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride)

    def forward(self, x):
        identity = x
        x, x_pre_activ = self.body(x)
        if self.resize_identity:
            identity = self.identity_conv(x_pre_activ)
        x = x + identity
        return x


def interpolate(x, size):
    return F.interpolate(
        input=x,
        size=size,
        mode='bilinear',
        align_corners=True)


class InterpolationBlock(nn.Module):
    """
    Interpolation block.

    Parameters:
    ----------
    scale_factor : float
        Multiplier for spatial size.
    """
    def __init__(self,
                 scale_factor):
        super(InterpolationBlock, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x, **kwargs):
        return F.interpolate(
            input=x,
            scale_factor=self.scale_factor,
            mode='bilinear',
            align_corners=True)


class DoubleSkipBlock(nn.Module):
    """
    Double skip connection block.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(DoubleSkipBlock, self).__init__()
        self.skip1 = ResBlock(in_channels, out_channels)

    def forward(self, x, **kwargs):
        x = x + self.skip1(x)
        return x


class AttBlock(nn.Module):
    # input size is 56*56
    def __init__(self,
                 in_channels,
                 out_channels,
                 hourglass_depth,
                 att_scales):
        super(AttBlock, self).__init__()
        assert (len(att_scales) == 3)
        scale_factor = 2
        scale_p, scale_t, scale_r = att_scales

        self.init_blocks = nn.Sequential()
        for i in range(scale_p):
            self.init_blocks.add_module('block{}'.format(i + 1), ResBlock(in_channels, out_channels))

        down_seq = nn.Sequential()
        up_seq = nn.Sequential()
        skip_seq = nn.Sequential()
        for i in range(hourglass_depth):
            down_res_blocks = nn.Sequential()
            for j in range(scale_r):
                down_res_blocks.add_module('block{}'.format(j + 1), ResBlock(in_channels, out_channels))
            down_seq.add_module('down{}'.format(i + 1), nn.Sequential(
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                down_res_blocks))
            up_res_blocks = nn.Sequential()
            for j in range(scale_r):
                up_res_blocks.add_module('block{}'.format(j + 1), ResBlock(in_channels, out_channels))
            up_seq.add_module('up{}'.format(i + 1), nn.Sequential(
                up_res_blocks,
                InterpolationBlock(scale_factor)))
            if i == 0:
                skip_res_blocks = nn.Sequential()
                for j in range(scale_t):
                    skip_res_blocks.add_module('block{}'.format(j + 1), ResBlock(in_channels, out_channels))
                skip_seq.add_module('skip1', skip_res_blocks)
            else:
                skip_seq.add_module('skip{}'.format(i + 1), DoubleSkipBlock(in_channels, out_channels))
        self.hg = Hourglass(
            down_seq=down_seq,
            up_seq=up_seq,
            skip_seq=skip_seq,
            return_first_skip=True)

        self.middle_block = nn.Sequential(
            pre_conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels),
            pre_conv1x1_block(
                in_channels=out_channels,
                out_channels=out_channels),
            nn.Sigmoid()
        )

        self.final_block = ResBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.init_blocks(x)
        x, y = self.hg(x)
        x = self.middle_block(x)
        x = (1 + x) * y
        x = self.final_block(x)
        return x


class ResAttInitBlock(nn.Module):
    """
    ResAttNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels):
        super(ResAttInitBlock, self).__init__()
        self.conv = conv7x7_block(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=2)
        self.pool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class PreActivation(nn.Module):
    """
    Pre-activation block without convolution layer. It's used by itself as the final block in PreResNet.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self,
                 in_channels):
        super(PreActivation, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=in_channels)
        self.activ = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.activ(x)
        return x


class ResAttNet(nn.Module):
    """
    ResAttNet model.

    Parameters:
    ----------
    channels : list of list of int
        Number of output channels for each unit.
    init_block_channels : int
        Number of output channels for the initial unit.
    attentions : list of list of int
        Whether to use a attention unit or residual one.
    attentions : list of int
        Attention block specific scales.
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 channels,
                 init_block_channels,
                 attentions,
                 att_scales,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(ResAttNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes

        self.features = nn.Sequential()
        self.features.add_module("init_block", ResAttInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels))
        in_channels = init_block_channels
        for i, channels_per_stage in enumerate(channels):
            hourglass_depth = len(channels) - 1 - i
            stage = nn.Sequential()
            for j, out_channels in enumerate(channels_per_stage):
                stride = 1 if (i == 0) or (j != 0) else 2
                if attentions[i][j]:
                    stage.add_module("unit{}".format(j + 1), AttBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        hourglass_depth=hourglass_depth,
                        att_scales=att_scales))
                else:
                    stage.add_module("unit{}".format(j + 1), ResBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=stride))
                in_channels = out_channels
            self.features.add_module("stage{}".format(i + 1), stage)
        self.features.add_module('post_activ', PreActivation(in_channels=in_channels))
        self.features.add_module('final_pool', nn.AvgPool2d(
            kernel_size=7,
            stride=1))

        self.output = nn.Linear(
            in_features=in_channels,
            out_features=num_classes)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_resattnet(blocks,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join('~', '.torch', 'models'),
                  **kwargs):
    """
    Create ResAttNet model with specific parameters.

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
    if blocks == 56:
        att_layers = [1, 1, 1]
        att_scales = [1, 2, 1]
    elif blocks == 92:
        att_layers = [1, 2, 3]
        att_scales = [1, 2, 1]
    elif blocks == 128:
        att_layers = [2, 3, 4]
        att_scales = [1, 2, 1]
    elif blocks == 164:
        att_layers = [3, 4, 5]
        att_scales = [1, 2, 1]
    elif blocks == 200:
        att_layers = [4, 5, 6]
        att_scales = [1, 2, 1]
    elif blocks == 236:
        att_layers = [5, 6, 7]
        att_scales = [1, 2, 1]
    elif blocks == 452:
        att_layers = [5, 6, 7]
        att_scales = [2, 4, 3]
    else:
        raise ValueError("Unsupported ResAttNet with number of blocks: {}".format(blocks))

    init_block_channels = 64
    channels_per_layers = [256, 512, 1024, 2048]
    layers = att_layers + [2]
    channels = [[ci] * (li + 1) for (ci, li) in zip(channels_per_layers, layers)]
    attentions = [[0] + [1] * li for li in att_layers] + [[0] * 3]

    net = ResAttNet(
        channels=channels,
        init_block_channels=init_block_channels,
        attentions=attentions,
        att_scales=att_scales,
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


def oth_resattnet56(pretrained=False, **kwargs):
    return get_resattnet(56, **kwargs)


def oth_resattnet92(pretrained=False, **kwargs):
    return get_resattnet(92, **kwargs)


def oth_resattnet128(pretrained=False, **kwargs):
    return get_resattnet(128, **kwargs)


def oth_resattnet164(pretrained=False, **kwargs):
    return get_resattnet(164, **kwargs)


def oth_resattnet200(pretrained=False, **kwargs):
    return get_resattnet(200, **kwargs)


def oth_resattnet236(pretrained=False, **kwargs):
    return get_resattnet(236, **kwargs)


def oth_resattnet452(pretrained=False, **kwargs):
    return get_resattnet(452, **kwargs)


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
        oth_resattnet56,
        oth_resattnet92,
        oth_resattnet128,
        oth_resattnet164,
        oth_resattnet200,
        oth_resattnet236,
        oth_resattnet452,
    ]

    for model in models:

        net = model(pretrained=pretrained)
        # net = AttentionModule_stage1(256, 256)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != oth_resattnet56 or weight_count == 31810728)
        assert (model != oth_resattnet92 or weight_count == 52466344)
        assert (model != oth_resattnet128 or weight_count == 65294504)
        assert (model != oth_resattnet164 or weight_count == 78122664)
        assert (model != oth_resattnet200 or weight_count == 90950824)
        assert (model != oth_resattnet236 or weight_count == 103778984)
        assert (model != oth_resattnet452 or weight_count == 182285224)

        x = Variable(torch.randn(1, 3, 224, 224))
        y = net(x)
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()


