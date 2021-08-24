"""
    Jasper/DR for ASR, implemented in PyTorch.
    Original paper: 'Jasper: An End-to-End Convolutional Neural Acoustic Model,' https://arxiv.org/abs/1904.03288.
"""

__all__ = ['Jasper', 'jasper5x3', 'jasper10x4', 'jasper10x5', 'get_jasper', 'MaskConv1d', 'NemoAudioReader',
           'NemoMelSpecExtractor', 'CtcDecoder']

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import DualPathSequential, DualPathParallelConcurent


def outmask_fill(x, x_len, value=0.0):
    """
    Masked fill a tensor.

    Parameters:
    ----------
    x : tensor
        Input tensor.
    x_len : tensor
        Tensor with lengths.
    value : float, default 0.0
        Filled value.

    Returns:
    -------
    tensor
        Resulted tensor.
    """
    max_len = x.size(2)
    mask = torch.arange(max_len).to(x_len.device).expand(len(x_len), max_len) >= x_len.unsqueeze(1)
    mask = mask.unsqueeze(dim=1).to(device=x.device)
    x = x.masked_fill(mask=mask, value=value)
    return x


def masked_normalize(x, x_len):
    """
    Normalize a tensor with mask.

    Parameters:
    ----------
    x : tensor
        Input tensor.
    x_len : tensor
        Tensor with lengths.

    Returns:
    -------
    tensor
        Resulted tensor.
    """
    x = outmask_fill(x, x_len)
    x_mean = x.sum(dim=2) / x_len.unsqueeze(dim=1)
    x_m0 = x - x_mean.unsqueeze(dim=2)
    x_m0 = outmask_fill(x_m0, x_len)
    x_std = x_m0.sum(dim=2) / x_len.unsqueeze(dim=1)
    x = x_m0 / x_std.unsqueeze(dim=2)
    return x


def masked_normalize2(x, x_len):
    """
    Normalize a tensor with mask (scheme #2).

    Parameters:
    ----------
    x : tensor
        Input tensor.
    x_len : tensor
        Tensor with lengths.

    Returns:
    -------
    tensor
        Resulted tensor.
    """
    x = outmask_fill(x, x_len)
    x_mean = x.sum(dim=2) / x_len.unsqueeze(dim=1)
    x2_mean = x.square().sum(dim=2) / x_len.unsqueeze(dim=1)
    x_std = (x2_mean - x_mean.square()).sqrt()
    x = (x - x_mean.unsqueeze(dim=2)) / x_std.unsqueeze(dim=2)
    return x


def masked_normalize3(x, x_len):
    """
    Normalize a tensor with mask (scheme #3).

    Parameters:
    ----------
    x : tensor
        Input tensor.
    x_len : tensor
        Tensor with lengths.

    Returns:
    -------
    tensor
        Resulted tensor.
    """
    x_eps = 1e-5
    x_mean = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)
    x_std = torch.zeros(x.shape[:2], dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        x_mean[i, :] = x[i, :, : x_len[i]].mean(dim=1)
        x_std[i, :] = x[i, :, : x_len[i]].std(dim=1)
    x_std += x_eps
    return (x - x_mean.unsqueeze(dim=2)) / x_std.unsqueeze(dim=2)


class NemoAudioReader(object):
    """
    Audio Reader from NVIDIA NEMO toolkit.

    Parameters:
    ----------
    desired_audio_sample_rate : int, default 16000
        Desired audio sample rate.
    trunc_value : int or None, default None
        Value to truncate.
    """
    def __init__(self, desired_audio_sample_rate=16000):
        super(NemoAudioReader, self).__init__()
        self.desired_audio_sample_rate = desired_audio_sample_rate

    def read_from_file(self, audio_file_path):
        """
        Read audio from file.

        Parameters:
        ----------
        audio_file_path : str
            Path to audio file.

        Returns:
        -------
        np.array
            Audio data.
        """
        from soundfile import SoundFile
        with SoundFile(audio_file_path, "r") as data:
            sample_rate = data.samplerate
            audio_data = data.read(dtype="float32")

        audio_data = audio_data.transpose()

        if sample_rate != self.desired_audio_sample_rate:
            from librosa.core import resample as lr_resample
            audio_data = lr_resample(y=audio_data, orig_sr=sample_rate, target_sr=self.desired_audio_sample_rate)
        if audio_data.ndim >= 2:
            audio_data = np.mean(audio_data, axis=1)

        return audio_data

    def read_from_files(self, audio_file_paths):
        """
        Read audios from files.

        Parameters:
        ----------
        audio_file_paths : list of str
            Paths to audio files.

        Returns:
        -------
        list of np.array
            Audio data.
        """
        assert (type(audio_file_paths) in (list, tuple))

        audio_data_list = []
        for audio_file_path in audio_file_paths:
            audio_data = self.read_from_file(audio_file_path)
            audio_data_list.append(audio_data)
        return audio_data_list


class NemoMelSpecExtractor(nn.Module):
    """
    Mel-Spectrogram Extractor from NVIDIA NEMO toolkit.

    Parameters:
    ----------
    sample_rate : int, default 16000
        Sample rate of the input audio data.
    window_size_sec : float, default 0.02
        Size of window for FFT in seconds.
    window_stride_sec : float, default 0.01
        Stride of window for FFT in seconds.
    n_fft : int, default 512
        Length of FT window.
    n_filters : int, default 64
        Number of Mel spectrogram freq bins.
    preemph : float, default 0.97
        Amount of pre emphasis to add to audio.
    dither : float, default 1.0e-05
        Amount of white-noise dithering.
    """
    def __init__(self,
                 sample_rate=16000,
                 window_size_sec=0.02,
                 window_stride_sec=0.01,
                 n_fft=512,
                 n_filters=64,
                 preemph=0.97,
                 dither=1.0e-5):
        super(NemoMelSpecExtractor, self).__init__()
        self.log_zero_guard_value = 2 ** -24
        win_length = int(window_size_sec * sample_rate)
        self.hop_length = int(window_stride_sec * sample_rate)
        self.n_filters = n_filters

        window_tensor = torch.hann_window(win_length, periodic=False)
        self.register_buffer("window", window_tensor)
        self.stft = lambda x: torch.stft(
            x,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=win_length,
            window=self.window.to(dtype=torch.float),
            center=True)

        self.dither = dither
        self.preemph = preemph

        self.pad_align = 16

        from librosa.filters import mel as librosa_mel
        filter_bank = librosa_mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_filters,
            fmin=0.0,
            fmax=(sample_rate / 2.0))
        fb_tensor = torch.from_numpy(filter_bank).unsqueeze(0)
        self.register_buffer("fb", fb_tensor)

    def forward(self, x, x_len):
        """
        Preprocess audio.

        Parameters:
        ----------
        xs : list of np.array
            Audio data.

        Returns:
        -------
        x : np.array
            Audio data.
        x_len : np.array
            Audio data lengths.
        """
        x_len = torch.ceil(x_len.float() / self.hop_length).long()

        if self.dither > 0:
            x += self.dither * torch.randn_like(x)

        x = torch.cat((x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        with torch.cuda.amp.autocast(enabled=False):
            x = self.stft(x)

        x = x.pow(2).sum(-1)
        x = torch.matmul(self.fb.to(x.dtype), x)
        x = torch.log(x + self.log_zero_guard_value)

        x = masked_normalize2(x, x_len)
        x = outmask_fill(x, x_len)

        x_len_max = x.size(-1)
        pad_rem = x_len_max % self.pad_align
        if pad_rem != 0:
            x = F.pad(x, pad=(0, self.pad_align - pad_rem))

        return x, x_len

    def calc_flops(self, x):
        assert (x.shape[0] == 1)
        num_flops = x.numel()
        num_macs = 0
        return num_flops, num_macs


class CtcDecoder(object):
    """
    CTC decoder (to decode a sequence of labels to words).

    Parameters:
    ----------
    vocabulary : list of str
        Vocabulary of the dataset.
    """
    def __init__(self,
                 vocabulary):
        super().__init__()
        self.blank_id = len(vocabulary)
        self.labels_map = dict([(i, vocabulary[i]) for i in range(len(vocabulary))])

    def __call__(self,
                 predictions):
        """
        Decode a sequence of labels to words.

        Parameters:
        ----------
        predictions : np.array of int or list of list of int
            Tensor with predicted labels.

        Returns:
        -------
        list of str
            Words.
        """
        hypotheses = []
        for prediction in predictions:
            decoded_prediction = []
            previous = self.blank_id
            for p in prediction:
                if (p != previous or previous == self.blank_id) and p != self.blank_id:
                    decoded_prediction.append(p)
                previous = p
            hypothesis = "".join([self.labels_map[c] for c in decoded_prediction])
            hypotheses.append(hypothesis)
        return hypotheses


def conv1d1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            bias=False):
    """
    1-dim kernel version of the 1D convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)


class MaskConv1d(nn.Conv1d):
    """
    Masked 1D convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int or tuple/list of 1 int
        Convolution window size.
    stride : int or tuple/list of 1 int
        Strides of the convolution.
    padding : int or tuple/list of 1 int, default 0
        Padding value for convolution layer.
    dilation : int or tuple/list of 1 int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_mask : bool, default True
        Whether to use mask.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 use_mask=True):
        super(MaskConv1d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        self.use_mask = use_mask

    def forward(self, x, x_len):
        if self.use_mask:
            x = outmask_fill(x, x_len)
            x_len = (x_len + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) -
                     1) // self.stride[0] + 1
        x = F.conv1d(
            input=x,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups)
        return x, x_len


def mask_conv1d1(in_channels,
                 out_channels,
                 stride=1,
                 groups=1,
                 bias=False):
    """
    Masked 1-dim kernel version of the 1D convolution layer.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default 1
        Strides of the convolution.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    """
    return MaskConv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        groups=groups,
        bias=bias)


class MaskConvBlock1d(nn.Module):
    """
    Masked 1D convolution block with batch normalization, activation, and dropout.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    stride : int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 dropout_rate=0.0):
        super(MaskConvBlock1d, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_dropout = (dropout_rate != 0.0)

        self.conv = MaskConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = activation()
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, x_len):
        x, x_len = self.conv(x, x_len)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x, x_len


def mask_conv1d1_block(in_channels,
                       out_channels,
                       stride=1,
                       padding=0,
                       **kwargs):
    """
    1-dim kernel version of the masked 1D convolution block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, default 1
        Strides of the convolution.
    padding : int, default 0
        Padding value for convolution layer.
    """
    return MaskConvBlock1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=1,
        stride=stride,
        padding=padding,
        **kwargs)


class ChannelShuffle1d(nn.Module):
    """
    1D version of the channel shuffle layer.

    Parameters:
    ----------
    channels : int
        Number of channels.
    groups : int
        Number of groups.
    """
    def __init__(self,
                 channels,
                 groups):
        super(ChannelShuffle1d, self).__init__()
        assert (channels % groups == 0)
        self.groups = groups

    def forward(self, x):
        batch, channels, seq_len = x.size()
        channels_per_group = channels // self.groups
        x = x.view(batch, self.groups, channels_per_group, seq_len)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batch, channels, seq_len)
        return x

    def __repr__(self):
        s = "{name}(groups={groups})"
        return s.format(
            name=self.__class__.__name__,
            groups=self.groups)


class DwsConvBlock1d(nn.Module):
    """
    Depthwise version of the 1D standard convolution block with batch normalization, activation, dropout, and channel
    shuffle.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    stride : int
        Strides of the convolution.
    padding : int
        Padding value for convolution layer.
    dilation : int, default 1
        Dilation value for convolution layer.
    groups : int, default 1
        Number of groups.
    bias : bool, default False
        Whether the layer uses a bias vector.
    use_bn : bool, default True
        Whether to use BatchNorm layer.
    bn_eps : float, default 1e-5
        Small float added to variance in Batch norm.
    activation : function or str or None, default nn.ReLU(inplace=True)
        Activation function or name of activation function.
    dropout_rate : float, default 0.0
        Parameter of Dropout layer. Faction of the input units to drop.
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
                 use_bn=True,
                 bn_eps=1e-5,
                 activation=(lambda: nn.ReLU(inplace=True)),
                 dropout_rate=0.0):
        super(DwsConvBlock1d, self).__init__()
        self.activate = (activation is not None)
        self.use_bn = use_bn
        self.use_dropout = (dropout_rate != 0.0)
        self.use_channel_shuffle = (groups > 1)

        self.dw_conv = MaskConv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias)
        self.pw_conv = mask_conv1d1(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            bias=bias)
        if self.use_channel_shuffle:
            self.shuffle = ChannelShuffle1d(
                channels=out_channels,
                groups=groups)
        if self.use_bn:
            self.bn = nn.BatchNorm1d(
                num_features=out_channels,
                eps=bn_eps)
        if self.activate:
            self.activ = activation()
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, x_len):
        x, x_len = self.dw_conv(x, x_len)
        x, x_len = self.pw_conv(x, x_len)
        if self.use_channel_shuffle:
            x = self.shuffle(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activate:
            x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x, x_len


class JasperUnit(nn.Module):
    """
    Jasper unit with residual connection.

    Parameters:
    ----------
    in_channels : int or list of int
        Number of input channels.
    out_channels : int
        Number of output channels.
    kernel_size : int
        Convolution window size.
    bn_eps : float
        Small float added to variance in Batch norm.
    dropout_rate : float
        Parameter of Dropout layer. Faction of the input units to drop.
    repeat : int
        Count of body convolution blocks.
    use_dw : bool
        Whether to use depthwise block.
    use_dr : bool
        Whether to use dense residual scheme.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bn_eps,
                 dropout_rate,
                 repeat,
                 use_dw,
                 use_dr):
        super(JasperUnit, self).__init__()
        self.use_dropout = (dropout_rate != 0.0)
        self.use_dr = use_dr
        block_class = DwsConvBlock1d if use_dw else MaskConvBlock1d

        if self.use_dr:
            self.identity_block = DualPathParallelConcurent()
            for i, dense_in_channels_i in enumerate(in_channels):
                self.identity_block.add_module("block{}".format(i + 1), mask_conv1d1_block(
                    in_channels=dense_in_channels_i,
                    out_channels=out_channels,
                    bn_eps=bn_eps,
                    dropout_rate=0.0,
                    activation=None))
            in_channels = in_channels[-1]
        else:
            self.identity_block = mask_conv1d1_block(
                in_channels=in_channels,
                out_channels=out_channels,
                bn_eps=bn_eps,
                dropout_rate=0.0,
                activation=None)

        self.body = DualPathSequential()
        for i in range(repeat):
            activation = (lambda: nn.ReLU(inplace=True)) if i < repeat - 1 else None
            dropout_rate_i = dropout_rate if i < repeat - 1 else 0.0
            self.body.add_module("block{}".format(i + 1), block_class(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=(kernel_size // 2),
                bn_eps=bn_eps,
                dropout_rate=dropout_rate_i,
                activation=activation))
            in_channels = out_channels

        self.activ = nn.ReLU(inplace=True)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x, x_len):
        if self.use_dr:
            x_len, y, y_len = x_len if type(x_len) is tuple else (x_len, None, None)
            y = [x] if y is None else y + [x]
            y_len = [x_len] if y_len is None else y_len + [x_len]
            identity, _ = self.identity_block(y, y_len)
            identity = torch.stack(tuple(identity), dim=1)
            identity = identity.sum(dim=1)
        else:
            identity, _ = self.identity_block(x, x_len)

        x, x_len = self.body(x, x_len)
        x = x + identity
        x = self.activ(x)
        if self.use_dropout:
            x = self.dropout(x)

        if self.use_dr:
            return x, (x_len, y, y_len)
        else:
            return x, x_len


class JasperFinalBlock(nn.Module):
    """
    Jasper specific final block.

    Parameters:
    ----------
    in_channels : int
        Number of input channels.
    channels : list of int
        Number of output channels for each block.
    kernel_sizes : list of int
        Kernel sizes for each block.
    bn_eps : float
        Small float added to variance in Batch norm.
    dropout_rates : list of int
        Dropout rates for each block.
    use_dw : bool
        Whether to use depthwise block.
    use_dr : bool
        Whether to use dense residual scheme.
    """
    def __init__(self,
                 in_channels,
                 channels,
                 kernel_sizes,
                 bn_eps,
                 dropout_rates,
                 use_dw,
                 use_dr):
        super(JasperFinalBlock, self).__init__()
        self.use_dr = use_dr
        conv1_class = DwsConvBlock1d if use_dw else MaskConvBlock1d

        self.conv1 = conv1_class(
            in_channels=in_channels,
            out_channels=channels[-2],
            kernel_size=kernel_sizes[-2],
            stride=1,
            padding=(2 * kernel_sizes[-2] // 2 - 1),
            dilation=2,
            bn_eps=bn_eps,
            dropout_rate=dropout_rates[-2])
        self.conv2 = MaskConvBlock1d(
            in_channels=channels[-2],
            out_channels=channels[-1],
            kernel_size=kernel_sizes[-1],
            stride=1,
            padding=(kernel_sizes[-1] // 2),
            bn_eps=bn_eps,
            dropout_rate=dropout_rates[-1])

    def forward(self, x, x_len):
        if self.use_dr:
            x_len = x_len[0]
        x, x_len = self.conv1(x, x_len)
        x, x_len = self.conv2(x, x_len)
        return x, x_len


class Jasper(nn.Module):
    """
    Jasper/DR/QuartzNet model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    channels : list of int
        Number of output channels for each unit and initial/final block.
    kernel_sizes : list of int
        Kernel sizes for each unit and initial/final block.
    bn_eps : float
        Small float added to variance in Batch norm.
    dropout_rates : list of int
        Dropout rates for each unit and initial/final block.
    repeat : int
        Count of body convolution blocks.
    use_dw : bool
        Whether to use depthwise block.
    use_dr : bool
        Whether to use dense residual scheme.
    from_audio : bool, default True
        Whether to treat input as audio instead of Mel-specs.
    dither : float, default 0.0
        Amount of white-noise dithering.
    return_text : bool, default False
        Whether to return text instead of logits.
    vocabulary : list of str or None, default None
        Vocabulary of the dataset.
    in_channels : int, default 64
        Number of input channels (audio features).
    num_classes : int, default 29
        Number of classification classes (number of graphemes).
    """
    def __init__(self,
                 channels,
                 kernel_sizes,
                 bn_eps,
                 dropout_rates,
                 repeat,
                 use_dw,
                 use_dr,
                 from_audio=True,
                 dither=0.0,
                 return_text=False,
                 vocabulary=None,
                 in_channels=64,
                 num_classes=29):
        super(Jasper, self).__init__()
        self.in_size = in_channels
        self.num_classes = num_classes
        self.vocabulary = vocabulary
        self.from_audio = from_audio
        self.return_text = return_text

        if self.from_audio:
            self.preprocessor = NemoMelSpecExtractor(dither=dither)

        self.features = DualPathSequential()
        init_block_class = DwsConvBlock1d if use_dw else MaskConvBlock1d
        self.features.add_module("init_block", init_block_class(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=kernel_sizes[0],
            stride=2,
            padding=(kernel_sizes[0] // 2),
            bn_eps=bn_eps,
            dropout_rate=dropout_rates[0]))
        in_channels = channels[0]
        in_channels_list = []
        for i, (out_channels, kernel_size, dropout_rate) in\
                enumerate(zip(channels[1:-2], kernel_sizes[1:-2], dropout_rates[1:-2])):
            in_channels_list += [in_channels]
            self.features.add_module("unit{}".format(i + 1), JasperUnit(
                in_channels=(in_channels_list if use_dr else in_channels),
                out_channels=out_channels,
                kernel_size=kernel_size,
                bn_eps=bn_eps,
                dropout_rate=dropout_rate,
                repeat=repeat,
                use_dw=use_dw,
                use_dr=use_dr))
            in_channels = out_channels
        self.features.add_module("final_block", JasperFinalBlock(
            in_channels=in_channels,
            channels=channels,
            kernel_sizes=kernel_sizes,
            bn_eps=bn_eps,
            dropout_rates=dropout_rates,
            use_dw=use_dw,
            use_dr=use_dr))
        in_channels = channels[-1]

        self.output = conv1d1(
            in_channels=in_channels,
            out_channels=num_classes,
            bias=True)

        if self.return_text:
            self.ctc_decoder = CtcDecoder(vocabulary=vocabulary)

        self._init_params()

    def _init_params(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x, x_len=None):
        if x_len is None:
            assert (type(x) in (list, tuple))
            x, x_len = x

        if self.from_audio:
            x, x_len = self.preprocessor(x, x_len)

        x, x_len = self.features(x, x_len)
        x = self.output(x)

        if self.return_text:
            greedy_predictions = x.transpose(1, 2).log_softmax(dim=-1).argmax(dim=-1, keepdim=False).cpu().numpy()
            return self.ctc_decoder(greedy_predictions)
        else:
            return x, x_len


def get_jasper(version,
               use_dw=False,
               use_dr=False,
               bn_eps=1e-3,
               vocabulary=None,
               model_name=None,
               pretrained=False,
               root=os.path.join("~", ".torch", "models"),
               **kwargs):
    """
    Create Jasper/DR/QuartzNet model with specific parameters.

    Parameters:
    ----------
    version : tuple of str
        Model type and configuration.
    use_dw : bool, default False
        Whether to use depthwise block.
    use_dr : bool, default False
        Whether to use dense residual scheme.
    bn_eps : float, default 1e-3
        Small float added to variance in Batch norm.
    vocabulary : list of str or None, default None
        Vocabulary of the dataset.
    model_name : str or None, default None
        Model name for loading pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    import numpy as np

    blocks, repeat = tuple(map(int, version[1].split("x")))
    main_stage_repeat = blocks // 5

    model_type = version[0]
    if model_type == "jasper":
        channels_per_stage = [256, 256, 384, 512, 640, 768, 896, 1024]
        kernel_sizes_per_stage = [11, 11, 13, 17, 21, 25, 29, 1]
        dropout_rates_per_stage = [0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    elif model_type == "quartznet":
        channels_per_stage = [256, 256, 256, 512, 512, 512, 512, 1024]
        kernel_sizes_per_stage = [33, 33, 39, 51, 63, 75, 87, 1]
        dropout_rates_per_stage = [0.0] * 8
    else:
        raise ValueError("Unsupported Jasper family model type: {}".format(model_type))

    stage_repeat = np.full((8,), 1)
    stage_repeat[1:-2] *= main_stage_repeat
    channels = sum([[a] * r for (a, r) in zip(channels_per_stage, stage_repeat)], [])
    kernel_sizes = sum([[a] * r for (a, r) in zip(kernel_sizes_per_stage, stage_repeat)], [])
    dropout_rates = sum([[a] * r for (a, r) in zip(dropout_rates_per_stage, stage_repeat)], [])

    net = Jasper(
        channels=channels,
        kernel_sizes=kernel_sizes,
        bn_eps=bn_eps,
        dropout_rates=dropout_rates,
        repeat=repeat,
        use_dw=use_dw,
        use_dr=use_dr,
        vocabulary=vocabulary,
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


def jasper5x3(**kwargs):
    """
    Jasper 5x3 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version=("jasper", "5x3"), model_name="jasper5x3", **kwargs)


def jasper10x4(**kwargs):
    """
    Jasper 10x4 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version=("jasper", "10x4"), model_name="jasper10x4", **kwargs)


def jasper10x5(**kwargs):
    """
    Jasper 10x5 model from 'Jasper: An End-to-End Convolutional Neural Acoustic Model,'
    https://arxiv.org/abs/1904.03288.

    Parameters:
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.torch/models'
        Location for keeping the model parameters.
    """
    return get_jasper(version=("jasper", "10x5"), model_name="jasper10x5", **kwargs)


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
    from_audio = True
    audio_features = 64
    num_classes = 29
    use_cuda = True

    models = [
        jasper5x3,
        jasper10x4,
        jasper10x5,
    ]

    for model in models:

        net = model(
            in_channels=audio_features,
            num_classes=num_classes,
            from_audio=from_audio,
            pretrained=pretrained)

        if use_cuda:
            net = net.cuda()

        # net.train()
        net.eval()
        weight_count = _calc_width(net)
        print("m={}, {}".format(model.__name__, weight_count))
        assert (model != jasper5x3 or weight_count == 107681053)
        assert (model != jasper10x4 or weight_count == 261393693)
        assert (model != jasper10x5 or weight_count == 322286877)

        batch = 3
        aud_scale = 640 if from_audio else 1
        seq_len = np.random.randint(150, 250, batch) * aud_scale
        seq_len_max = seq_len.max() + 2
        x_shape = (batch, seq_len_max) if from_audio else (batch, audio_features, seq_len_max)
        x = torch.randn(x_shape)
        x_len = torch.tensor(seq_len, dtype=torch.long, device=x.device)

        if use_cuda:
            x = x.cuda()
            x_len = x_len.cuda()

        y, y_len = net(x, x_len)
        # y.sum().backward()

        assert (tuple(y.size())[:2] == (batch, net.num_classes))
        if from_audio:
            assert (y.size()[2] in range(seq_len_max // aud_scale * 2, seq_len_max // aud_scale * 2 + 9))
        else:
            assert (y.size()[2] in [seq_len_max // 2, seq_len_max // 2 + 1])


if __name__ == "__main__":
    _test()
