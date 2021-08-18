"""
    Automatic Speech Recognition (ASR) abstract dataset.
"""

__all__ = ['AsrDataset', 'asr_test_transform']

from mxnet.gluon.data import dataset
from mxnet.gluon.data.vision import transforms
from gluon.gluoncv2.models.jasper import NemoAudioReader


class AsrDataset(dataset.Dataset):
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
        self._transform = transform
        self.data = []
        self.audio_reader = NemoAudioReader()

    def __getitem__(self, index):
        wav_file_path, label_text = self.data[index]
        audio_data = self.audio_reader.read_from_file(wav_file_path)
        audio_len = audio_data.shape[0]
        return (audio_data, audio_len), label_text

    def __len__(self):
        return len(self.data)


def asr_test_transform(ds_metainfo):
    assert (ds_metainfo is not None)
    return transforms.Compose([])
