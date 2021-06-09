"""
    Automatic Speech Recognition (ASR) abstract dataset.
"""

__all__ = ['AsrDataset', 'NemoMelSpecExtractor', 'asr_test_transform']

import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms


class AsrDataset(data.Dataset):
    """
    Automatic Speech Recognition (ASR) abstract dataset.

    Parameters:
    ----------
    root : str
        Path to the folder stored the dataset.
    mode : str
        'train', 'val', 'test', or 'demo'.
    transform : func
        A function that takes data and transforms it.
    """
    def __init__(self,
                 root,
                 mode,
                 transform):
        super(AsrDataset, self).__init__()
        assert (mode in ("train", "val", "test", "demo"))
        self.root = root
        self.mode = mode
        self.transform = transform
        self.data = []
        self.preprocessor = NemoMelSpecExtractor(dither=0.0)

    def __getitem__(self, index):
        wav_file_path, label_text = self.data[index]

        audio_data_list = self.read_audio([wav_file_path])
        x_np, x_np_len = self.preprocessor(audio_data_list)

        return (x_np[0], x_np_len[0]), label_text

    def __len__(self):
        return len(self.data)

    @staticmethod
    def read_audio(audio_file_paths,
                   desired_audio_sample_rate=16000):
        """
        Read audio.

        Parameters:
        ----------
        audio_file_paths : list of str
            Paths to audio files.
        desired_audio_sample_rate : int, default 16000
            Desired audio sample rate.

        Returns:
        -------
        list of np.array
            Audio data.
        """
        from soundfile import SoundFile

        audio_data_list = []
        for audio_file_path in audio_file_paths:
            with SoundFile(audio_file_path, "r") as data:
                sample_rate = data.samplerate
                audio_data = data.read(dtype="float32")
            audio_data = audio_data.transpose()
            if desired_audio_sample_rate != sample_rate:
                from librosa.core import resample as lr_resample
                audio_data = lr_resample(y=audio_data, orig_sr=sample_rate, target_sr=desired_audio_sample_rate)
            if audio_data.ndim >= 2:
                audio_data = np.mean(audio_data, axis=1)
            audio_data_list.append(audio_data)

        return audio_data_list


class NemoMelSpecExtractor(object):
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
                 dither=1.0e-05,
                 **kwargs):
        super(NemoMelSpecExtractor, self).__init__(**kwargs)
        self.log_zero_guard_value = 2 ** -24
        win_length = int(window_size_sec * sample_rate)
        self.hop_length = int(window_stride_sec * sample_rate)

        from scipy import signal as scipy_signal
        from librosa import stft as librosa_stft
        from librosa.filters import mel as librosa_mel

        window_fn = scipy_signal.hann(win_length, sym=True)
        self.stft = lambda x: librosa_stft(
            x,
            n_fft=n_fft,
            hop_length=self.hop_length,
            win_length=win_length,
            window=window_fn,
            center=True)

        self.dither = dither
        self.preemph = preemph

        self.pad_align = 16

        self.filter_bank = librosa_mel(
            sample_rate,
            n_fft,
            n_mels=n_filters,
            fmin=0,
            fmax=(sample_rate / 2))

    def __call__(self, xs):
        """
        Preprocess audio.

        Parameters:
        ----------
        xs : list of np.array
            Audio data.

        Returns:
        -------
        np.array
            Audio data.
        np.array
            Audio data lengths.
        """
        x_eps = 1e-5

        batch = len(xs)
        x_len = np.zeros((batch,), dtype=np.long)

        ys = []
        for i, xi in enumerate(xs):
            x_len[i] = np.ceil(float(len(xi)) / self.hop_length).astype(np.long)

            if self.dither > 0:
                xi += self.dither * np.random.randn(*xi.shape)

            xi = np.concatenate((xi[:1], xi[1:] - self.preemph * xi[:-1]), axis=0)

            yi = self.stft(xi)
            yi = np.abs(yi)
            yi = np.square(yi)
            yi = np.matmul(self.filter_bank, yi)
            yi = np.log(yi + self.log_zero_guard_value)

            assert (yi.shape[1] != 1)
            yi_mean = yi.mean(axis=1)
            yi_std = yi.std(axis=1)
            yi_std += x_eps
            yi = (yi - np.expand_dims(yi_mean, axis=-1)) / np.expand_dims(yi_std, axis=-1)

            ys.append(yi)

        channels = ys[0].shape[0]
        x_len_max = max([yj.shape[-1] for yj in ys])
        x = np.zeros((batch, channels, x_len_max), dtype=np.float32)
        for i, yi in enumerate(ys):
            x_len_i = x_len[i]
            x[i, :, :x_len_i] = yi[:, :x_len_i]

        pad_rem = x_len_max % self.pad_align
        if pad_rem != 0:
            x = np.pad(x, ((0, 0), (0, 0), (0, self.pad_align - pad_rem)))

        return x, x_len


def asr_test_transform(ds_metainfo):
    assert (ds_metainfo is not None)
    return transforms.Compose([
        transforms.ToTensor(),
    ])
