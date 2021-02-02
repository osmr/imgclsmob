"""
    VisemeNet for speech-driven facial animation, implemented in PyTorch.
    Original paper: 'VisemeNet: Audio-Driven Animator-Centric Speech Animation,' https://arxiv.org/abs/1805.09488.
"""

__all__ = ['VisemeNet', 'visemenet20']

import os
import torch
import torch.nn as nn
from .common import DenseBlock


class VisemeDenseBranch(nn.Module):
    """
    VisemeNet dense branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of middle/output channels.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list):
        super(VisemeDenseBranch, self).__init__()
        self.branch = nn.Sequential()
        for i, out_channels in enumerate(out_channels_list[:-1]):
            self.branch.add_module("block{}".format(i + 1), DenseBlock(
                in_features=in_channels,
                out_features=out_channels,
                bias=True,
                use_bn=True))
            in_channels = out_channels
        self.final_fc = nn.Linear(
            in_features=in_channels,
            out_features=out_channels_list[-1])

    def forward(self, x):
        x = self.branch(x)
        y = self.final_fc(x)
        return y, x


class VisemeRnnBranch(nn.Module):
    """
    VisemeNet RNN branch.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels_list : list of int
        Number of middle/output channels.
    rnn_num_layers : int
        Number of RNN layers.
    dropout_rate : float
        Dropout rate.
    """
    def __init__(self,
                 in_channels,
                 out_channels_list,
                 rnn_num_layers,
                 dropout_rate):
        super(VisemeRnnBranch, self).__init__()
        self.rnn = nn.LSTM(
            input_size=in_channels,
            hidden_size=out_channels_list[0],
            num_layers=rnn_num_layers,
            dropout=dropout_rate)
        self.fc_branch = VisemeDenseBranch(
            in_channels=out_channels_list[0],
            out_channels_list=out_channels_list[1:])

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x[:, -1, :]
        y, _ = self.fc_branch(x)
        return y


class VisemeNet(nn.Module):
    """
    VisemeNet model from 'VisemeNet: Audio-Driven Animator-Centric Speech Animation,' https://arxiv.org/abs/1805.09488.

    Parameters:
    ----------
    audio_features : int, default 195
        Number of audio features (characters/sounds).
    audio_window_size : int, default 8
        Size of audio window (for time related audio features).
    stage2_window_size : int, default 64
        Size of window for stage #2.
    num_face_ids : int, default 76
        Number of face IDs.
    num_landmarks : int, default 76
        Number of landmarks.
    num_phonemes : int, default 21
        Number of phonemes.
    num_visemes : int, default 20
        Number of visemes.
    dropout_rate : float, default 0.5
        Dropout rate for RNNs.
    """
    def __init__(self,
                 audio_features=195,
                 audio_window_size=8,
                 stage2_window_size=64,
                 num_face_ids=76,
                 num_landmarks=76,
                 num_phonemes=21,
                 num_visemes=20,
                 dropout_rate=0.5):
        super(VisemeNet, self).__init__()
        stage1_rnn_hidden_size = 256
        stage1_fc_mid_channels = 256
        stage2_rnn_in_features = (audio_features + num_landmarks + stage1_fc_mid_channels) * \
                                 stage2_window_size // audio_window_size
        self.audio_window_size = audio_window_size
        self.stage2_window_size = stage2_window_size

        self.stage1_rnn = nn.LSTM(
            input_size=audio_features,
            hidden_size=stage1_rnn_hidden_size,
            num_layers=3,
            dropout=dropout_rate)
        self.lm_branch = VisemeDenseBranch(
            in_channels=(stage1_rnn_hidden_size + num_face_ids),
            out_channels_list=[stage1_fc_mid_channels, num_landmarks])
        self.ph_branch = VisemeDenseBranch(
            in_channels=(stage1_rnn_hidden_size + num_face_ids),
            out_channels_list=[stage1_fc_mid_channels, num_phonemes])

        self.cls_branch = VisemeRnnBranch(
            in_channels=stage2_rnn_in_features,
            out_channels_list=[256, 200, num_visemes],
            rnn_num_layers=1,
            dropout_rate=dropout_rate)
        self.reg_branch = VisemeRnnBranch(
            in_channels=stage2_rnn_in_features,
            out_channels_list=[256, 200, 100, num_visemes],
            rnn_num_layers=3,
            dropout_rate=dropout_rate)
        self.jali_branch = VisemeRnnBranch(
            in_channels=stage2_rnn_in_features,
            out_channels_list=[128, 200, 2],
            rnn_num_layers=3,
            dropout_rate=dropout_rate)

    def forward(self, x, pid):
        y, _ = self.stage1_rnn(x)
        y = y[:, -1, :]
        y = torch.cat((y, pid), dim=1)

        lm, _ = self.lm_branch(y)
        lm += pid

        ph, ph1 = self.ph_branch(y)

        z = torch.cat((lm, ph1), dim=1)

        z2 = torch.cat((z, x[:, self.audio_window_size // 2, :]), dim=1)
        n_net2_input = z2.shape[1]
        z2 = torch.cat((torch.zeros((self.stage2_window_size // 2, n_net2_input)), z2), dim=0)
        z = torch.stack(
            [z2[i:i + self.stage2_window_size].reshape(
                (self.audio_window_size, n_net2_input * self.stage2_window_size // self.audio_window_size))
              for i in range(z2.shape[0] - self.stage2_window_size)],
            dim=0)
        cls = self.cls_branch(z)
        reg = self.reg_branch(z)
        jali = self.jali_branch(z)

        return cls, reg, jali


def get_visemenet(model_name=None,
                  pretrained=False,
                  root=os.path.join("~", ".torch", "models"),
                  **kwargs):
    """
    Create VisemeNet model with specific parameters.

    Parameters:
    ----------
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    net = VisemeNet(
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


def visemenet20(**kwargs):
    """
    VisemeNet model for 20 visemes (without co-articulation rules) from 'VisemeNet: Audio-Driven Animator-Centric
    Speech Animation,' https://arxiv.org/abs/1805.09488.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_visemenet(model_name="visemenet20", **kwargs)


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
        visemenet20,
    ]

    for model in models:

        net = model(pretrained=pretrained)

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != visemenet20 or weight_count == 14574303)

        batch = 34
        audio_window_size = 8
        audio_features = 195
        num_face_ids = 76
        num_visemes = 20

        x = torch.randn(batch, audio_window_size, audio_features)
        pid = torch.full(size=(batch, num_face_ids), fill_value=3)
        y1, y2, y3 = net(x, pid)
        assert (y1.shape[0] == y2.shape[0] == y3.shape[0])
        assert (y1.shape[1] == y2.shape[1] == num_visemes)
        assert (y3.shape[1] == 2)


if __name__ == "__main__":
    _test()
