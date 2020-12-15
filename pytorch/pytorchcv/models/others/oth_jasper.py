"""
Jasper : https://arxiv.org/pdf/1904.03288.pdf
"""
import torch
from torch import nn
import torch.nn.functional as F


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class JasperBlock(nn.Module):
    """
    Jasper Block
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 repeat=3,
                 kernel_size=11,
                 stride=1,
                 dilation=1,
                 dropout=0.2,
                 residual=True,
                 residual_panes=[]):
        super().__init__()
        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        activation = nn.ReLU()

        self.conv = nn.ModuleList()
        inplanes_loop = in_channels
        for _ in range(repeat - 1):
            self.conv.extend(
                self._get_conv_bn_layer(
                    in_channels=inplanes_loop,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding_val,
                    dilation=dilation))
            self.conv.extend(
                self._get_act_dropout_layer(
                    drop_prob=dropout,
                    activation=activation))
            inplanes_loop = out_channels
        self.conv.extend(
            self._get_conv_bn_layer(
                in_channels=inplanes_loop,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding_val,
                dilation=dilation))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            if len(residual_panes) == 0:
                res_panes = [in_channels]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(nn.ModuleList(
                    modules=self._get_conv_bn_layer(
                        in_channels=ip,
                        out_channels=out_channels,
                        kernel_size=1)
                    )
                )

        self.out = nn.Sequential(
            *self._get_act_dropout_layer(
                drop_prob=dropout,
                activation=activation)
        )

    def _get_conv_bn_layer(self,
                           in_channels,
                           out_channels,
                           kernel_size,
                           stride=1,
                           dilation=1,
                           padding=0,
                           bias=False):
        layers = [
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=1,
                bias=bias),
            nn.BatchNorm1d(
                num_features=out_channels,
                eps=1e-3,
                momentum=0.1)
        ]
        return layers

    @staticmethod
    def _get_act_dropout_layer(drop_prob,
                               activation):
        layers = [
            activation,
            nn.Dropout(p=drop_prob)
        ]
        return layers

    def forward(self, input_):
        xs = input_

        # compute forward convolutions
        out = xs[-1]
        for i, l in enumerate(self.conv):
            out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    res_out = res_layer(res_out)
                out += res_out

        # compute the output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        return out


class JasperEncoder(nn.Module):
    """
    Jasper encoder
    """
    def __init__(self,
                 in_channels,
                 cfg):
        super().__init__()
        residual_panes = []
        encoder_layers = []
        for lcfg in cfg:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(in_channels)
                dense_res = residual_panes

            encoder_layers.append(
                JasperBlock(
                    in_channels=in_channels,
                    out_channels=lcfg['filters'],
                    repeat=lcfg['repeat'],
                    kernel_size=lcfg['kernel'],
                    stride=lcfg['stride'],
                    dilation=lcfg['dilation'],
                    dropout=lcfg['dropout'],
                    residual=lcfg['residual'],
                    residual_panes=dense_res))
            in_channels = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder([x])


class JasperDecoderForCTC(nn.Module):
    """
    Jasper decoder
    """
    def __init__(self,
                 in_channels,
                 num_classes):
        super().__init__()
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=num_classes,
                kernel_size=1,
                bias=True), )

    def forward(self, x):
        x = self.decoder_layers(x[-1]).transpose(1, 2)
        return F.log_softmax(x, dim=2)


class Jasper(nn.Module):
    """
    Contains jasper encoder and decoder
    """
    def __init__(self,
                 in_channels,
                 encoder_channels,
                 num_classes,
                 config):
        super().__init__()
        self.encoder = JasperEncoder(
            in_channels=in_channels,
            cfg=config)
        self.decoder = JasperDecoderForCTC(
            in_channels=encoder_channels,
            num_classes=num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def _calc_width(net):
    import numpy as np
    net_params = filter(lambda p: p.requires_grad, net.parameters())
    weight_count = 0
    for param in net_params:
        weight_count += np.prod(param.size())
    return weight_count


def _test():
    import numpy as np

    config = [
        {
            'filters': 256,
            'repeat': 1,
            'kernel': [11],
            'stride': [2],
            'dilation': [1],
            'dropout': 0.2,
            'residual': False
        },
        {
            'filters': 256,
            'repeat': 5,
            'kernel': [11],
            'stride': [1],
            'dilation': [1],
            'dropout': 0.2,
            'residual': True,
            'residual_dense': True
        },
        {
            'filters': 256,
            'repeat': 5,
            'kernel': [11],
            'stride': [1],
            'dilation': [1],
            'dropout': 0.2,
            'residual': True,
            'residual_dense': True
        },
        {
            'filters': 512,
            'repeat': 5,
            'kernel': [11],
            'stride': [1],
            'dilation': [1],
            'dropout': 0.2,
            'residual': True,
            'residual_dense': True
        },
        {
            'filters': 512,
            'repeat': 1,
            'kernel': [1],
            'stride': [1],
            'dilation': [1],
            'dropout': 0.4,
            'residual': False
        }
    ]

    num_classes = 11
    audio_features = 120

    model = Jasper
    net = model(
        in_channels=audio_features,
        encoder_channels=512,
        num_classes=num_classes,
        config=config)
    net.eval()
    weight_count = _calc_width(net)
    print("m={}, {}".format(model.__name__, weight_count))
    assert (model != Jasper or weight_count == 21397003)

    batch = 1
    seq_len = np.random.randint(60, 150)
    x = torch.randn(batch, audio_features, seq_len)
    y = net(x)
    # y.sum().backward()
    assert (tuple(y.size()) == (batch, seq_len // 2, num_classes)) or\
           (tuple(y.size()) == (batch, seq_len // 2 + 1, num_classes))


if __name__ == "__main__":
    _test()
