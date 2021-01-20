import torch
from torch import nn


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    if dilation > 1:
        return (dilation * kernel_size) // 2 - 1
    return kernel_size // 2


class GroupShuffle(nn.Module):

    def __init__(self, groups, channels):
        super(GroupShuffle, self).__init__()
        self.groups = groups
        self.channels_per_group = channels // groups

    def forward(self, x):
        sh = x.shape
        x = x.view(-1, self.groups, self.channels_per_group, sh[-1])
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(-1, self.groups * self.channels_per_group, sh[-1])
        return x


def get_conv_bn_layer(in_channels,
                      out_channels,
                      kernel_size=11,
                      stride=1,
                      dilation=1,
                      padding=0,
                      bias=False,
                      groups=1,
                      separable=False):
    if separable:
        layers = [
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=in_channels),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                dilation=1,
                padding=0,
                bias=bias,
                groups=groups)
        ]
    else:
        layers = [
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
                bias=bias,
                groups=groups)
        ]

    layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))

    if groups > 1:
        layers.append(GroupShuffle(groups, out_channels))
    return nn.Sequential(*layers)


def get_act_dropout_layer(drop_prob=0.2,
                          activation='relu'):
    if activation is None or activation == 'tanh':
        activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
    elif activation == 'relu':
        activation = nn.ReLU()
    layers = [
        activation,
        nn.Dropout(p=drop_prob)
    ]
    return nn.Sequential(*layers)


class QuartzUnit(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 repeat=3,
                 kernel_size=11,
                 stride=1,
                 residual=True,
                 dilation=1,
                 dropout=0.2,
                 groups=1,
                 separable=False):
        super(QuartzUnit, self).__init__()
        padding_val = get_same_padding(kernel_size, stride, dilation)

        temp_planes = in_channels
        net = []
        for _ in range(repeat):
            net.append(
                get_conv_bn_layer(
                    temp_planes,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_val,
                    groups=groups,
                    separable=separable)
            )
            net.append(
                get_act_dropout_layer(dropout)
            )
            temp_planes = out_channels
        self.net = nn.Sequential(*net)

        self.residual = residual
        if self.residual:
            self.residual_layer = get_conv_bn_layer(
                in_channels,
                out_channels,
                kernel_size=1)
        self.out = get_act_dropout_layer(dropout)

    def forward(self, x):
        out = self.net(x)
        if self.residual:
            resudial = self.residual_layer(x)
            out += resudial
        return self.out(out)


class QuartzNet(nn.Module):
    def __init__(self,
                 model_config,
                 in_channels,
                 num_classes):
        super(QuartzNet, self).__init__()
        layers = []
        for lcfg in model_config:
            groups = lcfg.get('groups', 1)
            separable = lcfg.get('separable', False)
            residual = lcfg.get('residual', True)
            layers.append(
                QuartzUnit(
                    in_channels=in_channels,
                    out_channels=lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'] if 'dropout' in lcfg else 0.0,
                    residual=residual,
                    groups=groups,
                    separable=separable))
            in_channels = lcfg['filters']
        self.features = nn.Sequential(*layers)

        self.output = nn.Conv1d(
            in_channels=1024,
            out_channels=num_classes,
            kernel_size=1,
            bias=True)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            elif isinstance(module, nn.BatchNorm1d):
                if module.track_running_stats:
                    module.running_mean.zero_()
                    module.running_var.fill_(1)
                    module.num_batches_tracked.zero_()
                if module.affine:
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.output(x)
        return x


def quartznet5x5(num_classes=120,
                 in_channels=64,
                 **kwargs):
    config = [
        {'filters': 256,
         'repeat': 1,
         'kernel': 33,
         'stride': 2,
         'dilation': 1,
         'dropout': 0.2,
         'residual': False,
         'separable': True},
        {'filters': 256,
         'repeat': 5,
         'kernel': 33,
         'stride': 1,
         'dilation': 1,
         'dropout': 0.2,
         'residual': True,
         'separable': True},
        {'filters': 256,
         'repeat': 5,
         'kernel': 39,
         'stride': 1,
         'dilation': 1,
         'dropout': 0.2,
         'residual': True,
         'separable': True},
        {'filters': 512,
         'repeat': 5,
         'kernel': 51,
         'stride': 1,
         'dilation': 1,
         'dropout': 0.2,
         'residual': True,
         'separable': True},
        {'filters': 512,
         'repeat': 5,
         'kernel': 63,
         'stride': 1,
         'dilation': 1,
         'dropout': 0.2,
         'residual': True,
         'separable': True},
        {'filters': 512,
         'repeat': 5,
         'kernel': 75,
         'stride': 1,
         'dilation': 1,
         'dropout': 0.2,
         'residual': True,
         'separable': True},
        {'filters': 512,
         'repeat': 1,
         'kernel': 87,
         'stride': 1,
         'dilation': 2,
         'dropout': 0.2,
         'residual': False,
         'separable': True},
        {'filters': 1024,
         'repeat': 1,
         'kernel': 1,
         'stride': 1,
         'dilation': 1,
         'dropout': 0.2,
         'residual': False,
         'separable': False}]

    net = QuartzNet(
        model_config=config,
        num_classes=num_classes,
        in_channels=in_channels)
    return net


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import numpy as np
    import torch

    pretrained = False
    audio_features = 120
    num_classes = 11

    models = [
        quartznet5x5,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            num_classes=num_classes,
            pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != quartznet5x5 or weight_count == 6710915)

        batch = 4
        seq_len = np.random.randint(60, 150)
        x = torch.randn(batch, audio_features, seq_len)
        y = net(x)
        # y.sum().backward()
        assert (tuple(y.size())[:2] == (batch, num_classes))


if __name__ == "__main__":
    _test()
