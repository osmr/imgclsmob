import torch
from torch import nn


def get_same_padding(kernel_size, stride, dilation):
    if stride > 1 and dilation > 1:
        raise ValueError("Only stride OR dilation may be greater than 1")
    return (kernel_size // 2) * dilation


class MaskedConv1d(nn.Conv1d):
    """
    1D convolution with sequence masking
    """
    __constants__ = ["use_conv_mask"]

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, use_conv_mask=True):
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride,
                         padding=padding, dilation=dilation,
                         groups=groups, bias=bias)
        self.use_conv_mask = use_conv_mask

    def get_seq_len(self, lens):
        return ((lens + 2 * self.padding - self.dilation *
                 (self.kernel_size - 1) - 1) / self.stride + 1)

    def forward(self, inp):
        if self.use_conv_mask:
            x, lens = inp
            max_len = x.size(1)
            idxs = torch.arange(max_len).to(lens.dtype).to(lens.device).expand(len(lens), max_len)

            mask = idxs >= lens.unsqueeze(1)
            x = x.masked_fill(mask.unsqueeze(1).to(device=x.device).bool(), 0)
            del mask
            del idxs
            lens = self.get_seq_len(lens)
            return super().forward(x), lens

        else:
            return super().forward(inp)


class JasperBlock(nn.Module):
    __constants__ = ["use_conv_mask", "conv"]
    """
    Jasper Block: https://arxiv.org/pdf/1904.03288.pdf
    """
    def __init__(self,
                 inplanes,
                 planes,
                 repeat=3,
                 kernel_size=11,
                 stride=1,
                 dilation=1,
                 padding='same',
                 dropout=0.2,
                 activation=None,
                 residual=True,
                 residual_panes=[],
                 use_conv_mask=False):
        super().__init__()

        if padding != "same":
            raise ValueError("currently only 'same' padding is supported")

        padding_val = get_same_padding(kernel_size[0], stride[0], dilation[0])
        self.use_conv_mask = use_conv_mask
        self.conv = nn.ModuleList()
        inplanes_loop = inplanes
        for _ in range(repeat - 1):
            self.conv.extend(
                self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                        stride=stride, dilation=dilation,
                                        padding=padding_val))
            self.conv.extend(
                self._get_act_dropout_layer(drop_prob=dropout, activation=activation))
            inplanes_loop = planes
        self.conv.extend(
            self._get_conv_bn_layer(inplanes_loop, planes, kernel_size=kernel_size,
                                    stride=stride, dilation=dilation,
                                    padding=padding_val))

        self.res = nn.ModuleList() if residual else None
        res_panes = residual_panes.copy()
        self.dense_residual = residual
        if residual:
            if len(residual_panes) == 0:
                res_panes = [inplanes]
                self.dense_residual = False
            for ip in res_panes:
                self.res.append(nn.ModuleList(
                    modules=self._get_conv_bn_layer(ip, planes, kernel_size=1)
                    )
                )

        self.out = nn.Sequential(
            *self._get_act_dropout_layer(drop_prob=dropout, activation=activation)
        )

    def _get_conv_bn_layer(self, in_channels, out_channels, kernel_size=11,
                           stride=1, dilation=1, padding=0, bias=False):
        layers = [
            MaskedConv1d(in_channels, out_channels, kernel_size, stride=stride,
                         dilation=dilation, padding=padding, bias=bias,
                         use_conv_mask=self.use_conv_mask),
            nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]
        return layers

    @staticmethod
    def _get_act_dropout_layer(drop_prob=0.2, activation=None):
        if activation is None:
            activation = nn.Hardtanh(min_val=0.0, max_val=20.0)
        layers = [
            activation,
            nn.Dropout(p=drop_prob)
        ]
        return layers

    def forward(self, input_):

        if self.use_conv_mask:
            xs, lens_orig = input_
        else:
            xs = input_
            lens_orig = 0

        # compute forward convolutions
        out = xs[-1]
        lens = lens_orig
        for i, l in enumerate(self.conv):
            if self.use_conv_mask and isinstance(l, MaskedConv1d):
                out, lens = l((out, lens))
            else:
                out = l(out)

        # compute the residuals
        if self.res is not None:
            for i, layer in enumerate(self.res):
                res_out = xs[i]
                for j, res_layer in enumerate(layer):
                    if j == 0 and self.use_conv_mask:
                        res_out, _ = res_layer((res_out, lens_orig))
                    else:
                        res_out = res_layer(res_out)
                out += res_out

        # compute the output
        out = self.out(out)
        if self.res is not None and self.dense_residual:
            out = xs + [out]
        else:
            out = [out]

        if self.use_conv_mask:
            return out, lens
        else:
            return out


class JasperEncoder(nn.Module):
    __constants__ = ["use_conv_mask"]
    """
    Jasper encoder
    """

    def __init__(self, **kwargs):
        cfg = {}
        for key, value in kwargs.items():
            cfg[key] = value

        super().__init__()
        self._cfg = cfg

        activation = nn.ReLU()
        self.use_conv_mask = False
        feat_in = cfg['input']['features'] * cfg['input'].get('frame_splicing', 1)

        residual_panes = []
        encoder_layers = []
        self.dense_residual = False
        for lcfg in cfg['jasper']:
            dense_res = []
            if lcfg.get('residual_dense', False):
                residual_panes.append(feat_in)
                dense_res = residual_panes
                self.dense_residual = True

            encoder_layers.append(
                JasperBlock(feat_in, lcfg['filters'], repeat=lcfg['repeat'],
                            kernel_size=lcfg['kernel'], stride=lcfg['stride'],
                            dilation=lcfg['dilation'], dropout=lcfg['dropout'],
                            residual=lcfg['residual'], activation=activation,
                            residual_panes=dense_res, use_conv_mask=self.use_conv_mask))
            feat_in = lcfg['filters']

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        if self.use_conv_mask:
            audio_signal, length = x
            return self.encoder(([audio_signal], length))
        else:
            return self.encoder([x])


class JasperDecoderForCTC(nn.Module):
    """
    Jasper decoder
    """
    def __init__(self, **kwargs):
        super().__init__()
        self._feat_in = kwargs.get("feat_in")
        self._num_classes = kwargs.get("num_classes")

        self.decoder_layers = nn.Sequential(
            nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True), )

    def forward(self, encoder_output):
        out = self.decoder_layers(encoder_output[-1]).transpose(1, 2)
        return nn.functional.log_softmax(out, dim=2)


class Jasper(nn.Module):
    """
    Contains jasper encoder and decoder
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.jasper_encoder = JasperEncoder(**kwargs.get("jasper_model_definition"))
        self.jasper_decoder = JasperDecoderForCTC(
            feat_in=kwargs.get("feat_in"),
            num_classes=kwargs.get("num_classes"))

    def forward(self, x):
        x = self.jasper_encoder(x)
        x = self.jasper_decoder(x)
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

    config = {
        'net': 'Jasper',
        'input': {
            'normalize': 'per_feature',
            'sample_rate': 16000,
            'window_size': 0.02,
            'window_stride': 0.01,
            'window': 'hann',
            'features': 120,
            'n_fft': 512,
            'frame_splicing': 1,
            'dither': 1e-05,
            'feat_type': 'logfbank',
            'normalize_transcripts': False,
            'trim_silence': True,
            'pad_to': 16,
            'max_duration': 25.0,
            'speed_perturbation': False
        },
        'jasper': [
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
    }

    num_classes = 11

    model = Jasper
    net = model(jasper_model_definition=config, num_classes=num_classes, feat_in=512)
    net.eval()
    weight_count = _calc_width(net)
    print("m={}, {}".format(model.__name__, weight_count))
    assert (model != Jasper or weight_count == 21397003)

    batch = 1
    x_len = np.random.randint(60, 150)
    x = torch.randn(batch, 120, x_len)
    y = net(x)
    # y.sum().backward()
    assert (tuple(y.size()) == (batch, x_len // 2, num_classes)) or\
           (tuple(y.size()) == (batch, x_len // 2 + 1, num_classes))


if __name__ == "__main__":
    _test()
