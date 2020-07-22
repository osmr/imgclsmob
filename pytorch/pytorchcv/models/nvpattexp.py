"""
    Neural Voice Puppetry Audio-to-Expression net for speech-driven facial animation, implemented in PyTorch.
    Original paper: 'Neural Voice Puppetry: Audio-driven Facial Reenactment,' https://arxiv.org/abs/1912.05566.
"""

__all__ = ['NvpAttExp', 'nvpattexp116bazel76']

import os
import torch
import torch.nn as nn
from .common import DenseBlock, ConvBlock, ConvBlock1d, SelectableDense


class NvpAttExpEncoder(nn.Module):
    """
    Neural Voice Puppetry Audio-to-Expression encoder.

    Parameters:
    ----------
    audio_features : int
        Number of audio features (characters/sounds).
    audio_window_size : int
        Size of audio window (for time related audio features).
    seq_len : int, default
        Size of feature window.
    encoder_features : int
        Number of encoder features.
    """
    def __init__(self,
                 audio_features,
                 audio_window_size,
                 seq_len,
                 encoder_features):
        super(NvpAttExpEncoder, self).__init__()
        self.audio_features = audio_features
        self.audio_window_size = audio_window_size
        self.seq_len = seq_len
        conv_channels = (32, 32, 64, 64)
        conv_slopes = (0.02, 0.02, 0.2, 0.2)
        fc_channels = (128, 64, encoder_features)
        fc_slopes = (0.02, 0.02, None)
        att_conv_channels = (16, 8, 4, 2, 1)
        att_conv_slopes = 0.02

        in_channels = audio_features
        self.conv_branch = nn.Sequential()
        for i, (out_channels, slope) in enumerate(zip(conv_channels, conv_slopes)):
            self.conv_branch.add_module("conv{}".format(i + 1), ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 1),
                stride=(2, 1),
                padding=(1, 0),
                bias=True,
                use_bn=False,
                activation=(lambda: nn.LeakyReLU(negative_slope=slope, inplace=True))))
            in_channels = out_channels

        self.fc_branch = nn.Sequential()
        for i, (out_channels, slope) in enumerate(zip(fc_channels, fc_slopes)):
            activation = (lambda: nn.LeakyReLU(negative_slope=slope, inplace=True)) if slope is not None else\
                (lambda: nn.Tanh())
            self.fc_branch.add_module("fc{}".format(i + 1), DenseBlock(
                in_features=in_channels,
                out_features=out_channels,
                bias=True,
                use_bn=False,
                activation=activation))
            in_channels = out_channels

        self.att_conv_branch = nn.Sequential()
        for i, out_channels, in enumerate(att_conv_channels):
            self.att_conv_branch.add_module("att_conv{}".format(i + 1), ConvBlock1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                use_bn=False,
                activation=(lambda: nn.LeakyReLU(negative_slope=att_conv_slopes, inplace=True))))
            in_channels = out_channels

        self.att_fc = DenseBlock(
            in_features=seq_len,
            out_features=seq_len,
            bias=True,
            use_bn=False,
            activation=(lambda: nn.Softmax(dim=1)))

    def forward(self, x):
        batch = x.shape[0]
        batch_seq_len = batch * self.seq_len

        x = x.view(batch_seq_len, 1, self.audio_window_size, self.audio_features)
        x = x.transpose(1, 3).contiguous()
        x = self.conv_branch(x)
        x = x.view(batch_seq_len, 1, -1)
        x = self.fc_branch(x)
        x = x.view(batch, self.seq_len, -1)
        x = x.transpose(1, 2).contiguous()

        y = x[:, :, (self.seq_len // 2)]

        w = self.att_conv_branch(x)
        w = w.view(batch, self.seq_len)
        w = self.att_fc(w)
        w = w.view(batch, self.seq_len, 1)
        x = torch.bmm(x, w)
        x = x.squeeze(dim=-1)

        return x, y


class NvpAttExp(nn.Module):
    """
    Neural Voice Puppetry Audio-to-Expression model from 'Neural Voice Puppetry: Audio-driven Facial Reenactment,'
    https://arxiv.org/abs/1912.05566.

    Parameters:
    ----------
    audio_features : int, default 29
        Number of audio features (characters/sounds).
    audio_window_size : int, default 16
        Size of audio window (for time related audio features).
    seq_len : int, default 8
        Size of feature window.
    base_persons : int, default 116
        Number of base persons (identities).
    blendshapes : int, default 76
        Number of 3D model blendshapes.
    encoder_features : int, default 32
        Number of encoder features.
    """
    def __init__(self,
                 audio_features=29,
                 audio_window_size=16,
                 seq_len=8,
                 base_persons=116,
                 blendshapes=76,
                 encoder_features=32):
        super(NvpAttExp, self).__init__()
        self.base_persons = base_persons

        self.encoder = NvpAttExpEncoder(
            audio_features=audio_features,
            audio_window_size=audio_window_size,
            seq_len=seq_len,
            encoder_features=encoder_features)
        self.decoder = SelectableDense(
            in_features=encoder_features,
            out_features=blendshapes,
            bias=False,
            num_options=base_persons)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, pid):
        x, y = self.encoder(x)
        x = self.decoder(x, pid)
        y = self.decoder(y, pid)
        return x, y


def get_nvpattexp(base_persons,
                  blendshapes,
                  model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create Neural Voice Puppetry Audio-to-Expression model with specific parameters.

    Parameters:
    ----------
    base_persons : int
        Number of base persons (subjects).
    blendshapes : int
        Number of 3D model blendshapes.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = NvpAttExp(
        base_persons=base_persons,
        blendshapes=blendshapes,
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


def nvpattexp116bazel76(**kwargs):
    """
    Neural Voice Puppetry Audio-to-Expression model for 116 base persons and Bazel topology with 76 blendshapes from
    'Neural Voice Puppetry: Audio-driven Facial Reenactment,' https://arxiv.org/abs/1912.05566.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_nvpattexp(base_persons=116, blendshapes=76, model_name="nvpattexp116bazel76", **kwargs)


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
        nvpattexp116bazel76,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != nvpattexp116bazel76 or weight_count == 327397)

        batch = 14
        seq_len = 8
        audio_window_size = 16
        audio_features = 29
        blendshapes = 76

        x = torch.randn(batch, seq_len, audio_window_size, audio_features)
        pid = torch.full(size=(batch,), fill_value=3, dtype=torch.int64)
        y1, y2 = net(x, pid)
        # y1.sum().backward()
        assert (y1.shape == y2.shape == (batch, blendshapes))


if __name__ == "__main__":
    _test()
