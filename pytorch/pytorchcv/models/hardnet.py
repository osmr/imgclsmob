"""
    HarDNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.
"""

__all__ = ['HarDNet', 'hardnet39ds', 'hardnet68ds', 'hardnet68', 'hardnet85']

import os
import torch
import torch.nn as nn
from common import conv1x1_block, conv3x3_block, dwconv3x3_block, dwconv_block


class InvDwsConvBlock(nn.Module):
    """
    Inverse Depthwise separable convolution block with BatchNorms and activations at each convolution layers.

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
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    pw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the pointwise convolution block.
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation=1,
                 bias=False,
                 use_bn=True,
                 bn_eps=1e-5,
                 pw_activation=(lambda: nn.ReLU(inplace=True)),
                 dw_activation=(lambda: nn.ReLU(inplace=True))):
        super(InvDwsConvBlock, self).__init__()
        self.pw_conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=pw_activation)
        self.dw_conv = dwconv_block(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            use_bn=use_bn,
            bn_eps=bn_eps,
            activation=dw_activation)

    def forward(self, x):
        x = self.pw_conv(x)
        x = self.dw_conv(x)
        return x


def invdwsconv3x3_block(in_channels,
                        out_channels,
                        stride=1,
                        padding=1,
                        dilation=1,
                        bias=False,
                        bn_eps=1e-5,
                        pw_activation=(lambda: nn.ReLU(inplace=True)),
                        dw_activation=(lambda: nn.ReLU(inplace=True))):
    """
    3x3 inverse depthwise separable version of the standard convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int or tuple/list of 2 int, default 1
        Strides of the convolution.
    padding : int or tuple/list of 2 int, default 1
        Padding value for convolution layer.
    dilation : int or tuple/list of 2 int, default 1
        Dilation value for convolution layer.
    bias : bool, default False
        Whether the layer uses a bias vector.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    pw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the pointwise convolution block.
    dw_activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function after the depthwise convolution block.
    """
    return InvDwsConvBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=bias,
        bn_eps=bn_eps,
        dw_activation=dw_activation,
        pw_activation=pw_activation)


class HarDStage(nn.Module):
    """
    HarDNet stage.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    growth_rate : int
        Growth rate.
    growth_factor : float
        Growth factor.
    num_layers : int
        Number of layers.
    use_deptwise : bool
        Whether to use depthwise downsampling.
    activation : str
        Name of activation function.
    """
    def __init__(self,
                 in_channels,
                 growth_rate,
                 growth_factor,
                 num_layers,
                 use_deptwise,
                 activation):
        super(HarDStage, self).__init__()
        keepBase = False
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0  # if upsample else in_channels
        for i in range(num_layers):
            out_channels_i, in_channels_i, link_i = self.get_link(
                layer_idx=(i + 1),
                base_channels=in_channels,
                growth_rate=growth_rate,
                growth_factor=growth_factor)
            # print("i={}, in_channels_i={}".format(i, in_channels_i))
            # print("i={}, link_i={}".format(i, link_i))
            self.links.append(link_i)
            if use_deptwise:
                layers_.append(invdwsconv3x3_block(
                    in_channels=in_channels_i,
                    out_channels=out_channels_i,
                    pw_activation=activation,
                    dw_activation=None))
            else:
                layers_.append(conv3x3_block(
                    in_channels=in_channels_i,
                    out_channels=out_channels_i))

            if (i % 2 == 0) or (i == num_layers - 1):
                self.out_channels += out_channels_i
        self.layers = nn.ModuleList(layers_)

    def forward(self, x):
        layers_ = [x]

        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = []
            for i in link:
                tin.append(layers_[i])
            if len(tin) > 1:
                x = torch.cat(tin, 1)
            else:
                x = tin[0]
            out = self.layers[layer](x)
            layers_.append(out)

        t = len(layers_)
        out_ = []
        for i in range(t):
            if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1):
                out_.append(layers_[i])
        out = torch.cat(out_, 1)
        return out

    def get_link(self,
                 layer_idx,
                 base_channels,
                 growth_rate,
                 growth_factor):
        if layer_idx == 0:
            return base_channels, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer_idx % dv == 0:
                k = layer_idx - dv
                link.append(k)
                if i > 0:
                    out_channels *= growth_factor
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(
                layer_idx=i,
                base_channels=base_channels,
                growth_rate=growth_rate,
                growth_factor=growth_factor)
            in_channels += ch
        return out_channels, in_channels, link

    def get_out_ch(self):
        return self.out_channels


class HarDTransitionBlock(nn.Module):
    """
    HarDNet transition block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_dropout : bool
        Whether to use dropout module.
    use_downsamples : bool
        Whether to use downsampling module.
    use_deptwise : bool
        Whether to use depthwise downsampling.
    activation : str
        Name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_dropout,
                 use_downsamples,
                 use_deptwise,
                 activation):
        super(HarDTransitionBlock, self).__init__()
        self.use_dropout = use_dropout
        self.use_downsamples = use_downsamples

        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.1)

        self.conv = conv1x1_block(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=activation)

        if use_downsamples:
            if use_deptwise:
                self.downsample = dwconv3x3_block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=2,
                    activation=None)
            else:
                self.downsample = nn.MaxPool2d(
                    kernel_size=2,
                    stride=2)

    def forward(self, x):
        if self.use_dropout:
            x = self.dropout(x)
        x = self.conv(x)
        if self.use_downsamples:
            x = self.downsample(x)
        return x


class HarDInitBlock(nn.Module):
    """
    HarDNet specific initial block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    use_deptwise : bool
        Whether to use depthwise downsampling.
    activation : str
        Name of activation function.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_deptwise,
                 activation):
        super(HarDInitBlock, self).__init__()
        mid_channels = out_channels // 2

        self.conv1 = conv3x3_block(
            in_channels=in_channels,
            out_channels=mid_channels,
            stride=2,
            activation=activation)
        conv2_block_class = conv1x1_block if use_deptwise else conv3x3_block
        self.conv2 = conv2_block_class(
            in_channels=mid_channels,
            out_channels=out_channels,
            activation=activation)
        if use_deptwise:
            self.downsample = dwconv3x3_block(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=2,
                activation=None)
        else:
            self.downsample = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.downsample(x)
        return x


class HarDNet(nn.Module):
    """
    HarDNet model from 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    in_channels : int, default 3
        Number of input channels.
    in_size : tuple of two ints, default (224, 224)
        Spatial size of the expected input image.
    num_classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self,
                 init_block_channels,
                 growth_factor,
                 dropout_rate,
                 use_deptwise,
                 use_dropout,
                 layers,
                 channels_per_layers,
                 downsamples,
                 growth_rates,
                 in_channels=3,
                 in_size=(224, 224),
                 num_classes=1000):
        super(HarDNet, self).__init__()
        self.in_size = in_size
        self.num_classes = num_classes
        activation = "relu6"

        self.features = nn.Sequential()
        self.features.add_module("init_block", HarDInitBlock(
            in_channels=in_channels,
            out_channels=init_block_channels,
            use_deptwise=use_deptwise,
            activation=activation))
        in_channels = init_block_channels
        for i, layers_per_stage in enumerate(layers):
            out_channels = channels_per_layers[i]
            stage = HarDStage(
                in_channels=in_channels,
                growth_rate=growth_rates[i],
                growth_factor=growth_factor,
                num_layers=layers_per_stage,
                use_deptwise=use_deptwise,
                activation=activation)
            self.features.add_module("stage{}".format(i + 1), stage)
            self.features.add_module("trans{}".format(i + 1), HarDTransitionBlock(
                in_channels=stage.get_out_ch(),
                out_channels=out_channels,
                use_dropout=(i == len(layers) - 1 and use_dropout),
                use_downsamples=(downsamples[i] == 1),
                use_deptwise=use_deptwise,
                activation=activation))
            in_channels = out_channels
            # print(stage.get_out_ch())

        self.features.add_module("final_pool", nn.AdaptiveAvgPool2d(output_size=1))

        self.output = nn.Sequential()
        self.output.add_module("dropout", nn.Dropout(p=dropout_rate))
        self.output.add_module("fc", nn.Linear(
            in_features=in_channels,
            out_features=num_classes))

        self._init_params()

    def _init_params(self):
        for module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x


def get_hardnet(blocks,
                use_deptwise=True,
                model_name=None,
                pretrained=False,
                root=os.path.join("~", ".torch", "models"),
                **kwargs):
    """
    Create HarDNet model with specific parameters.

    Parameters:
    ----------
    blocks : int
        Number of blocks.
    use_deepwise : bool, default True
        Whether to use depthwise separable version of the model.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    if blocks == 39:
        init_block_channels = 48
        growth_factor = 1.6
        dropout_rate = 0.05 if use_deptwise else 0.1
        layers = [4, 16, 8, 4]
        channels_per_layers = [96, 320, 640, 1024]
        downsamples = [1, 1, 1, 0]
        growth_rates = [16, 20, 64, 160]
        use_dropout = False
    elif blocks == 68:
        init_block_channels = 64
        growth_factor = 1.7
        dropout_rate = 0.05 if use_deptwise else 0.1
        layers = [8, 16, 16, 16, 4]
        channels_per_layers = [128, 256, 320, 640, 1024]
        downsamples = [1, 0, 1, 1, 0]
        growth_rates = [14, 16, 20, 40, 160]
        use_dropout = False
    elif blocks == 85:
        init_block_channels = 96
        growth_factor = 1.7
        dropout_rate = 0.05 if use_deptwise else 0.2
        layers = [8, 16, 16, 16, 16, 4]
        channels_per_layers = [192, 256, 320, 480, 720, 1280]
        downsamples = [1, 0, 1, 0, 1, 0]
        growth_rates = [24, 24, 28, 36, 48, 256]
        use_dropout = True
    else:
        raise ValueError("Unsupported HarDNet version with number of layers {}".format(blocks))

    # def calc_out_channels(
    #         growth_rate,
    #         growth_factor_,
    #         num_layers):
    #     out_channels_ = 0
    #     for k in range(num_layers):
    #         layer_idx = (k + 1)
    #         out_channels_k = growth_rate
    #         for i in range(10):
    #             dv = 2 ** i
    #             if layer_idx % dv == 0:
    #                 if i > 0:
    #                     out_channels_k *= growth_factor_
    #         out_channels_k = int(int(out_channels_k + 1) / 2) * 2
    #         if (k % 2 == 0) or (k == num_layers - 1):
    #             out_channels_ += out_channels_k
    #     return out_channels_
    #
    # out_channels = [calc_out_channels(growth_rate, growth_factor, num_layers) for growth_rate, num_layers in
    #                 zip(layers, growth_rates)]
    # print(out_channels)

    net = HarDNet(
        init_block_channels=init_block_channels,
        growth_factor=growth_factor,
        dropout_rate=dropout_rate,
        use_deptwise=use_deptwise,
        use_dropout=use_dropout,
        layers=layers,
        channels_per_layers=channels_per_layers,
        downsamples=downsamples,
        growth_rates=growth_rates,
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


def hardnet39ds(**kwargs):
    """
    HarDNet-39DS (Depthwise Separable) model from 'HarDNet: A Low Memory Traffic Network,'
    https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=39, use_deptwise=True, model_name="hardnet39ds", **kwargs)


def hardnet68ds(**kwargs):
    """
    HarDNet-68DS (Depthwise Separable) model from 'HarDNet: A Low Memory Traffic Network,'
    https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=68, use_deptwise=True, model_name="hardnet68ds", **kwargs)


def hardnet68(**kwargs):
    """
    HarDNet-68 model from 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=68, use_deptwise=False, model_name="hardnet68", **kwargs)


def hardnet85(**kwargs):
    """
    HarDNet-85 model from 'HarDNet: A Low Memory Traffic Network,' https://arxiv.org/abs/1909.00948.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_hardnet(blocks=85, use_deptwise=False, model_name="hardnet85", **kwargs)


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
        hardnet39ds,
        hardnet68ds,
        hardnet68,
        hardnet85,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != hardnet39ds or weight_count == 3488228)
        assert (model != hardnet68ds or weight_count == 4180602)
        assert (model != hardnet68 or weight_count == 17565348)
        assert (model != hardnet85 or weight_count == 36670212)

        x = torch.randn(1, 3, 224, 224)
        y = net(x)
        y.sum().backward()
        assert (tuple(y.size()) == (1, 1000))


if __name__ == "__main__":
    _test()
