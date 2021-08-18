"""
    LibriSpeech ASR dataset.
"""

__all__ = ['LibriSpeech', 'LibriSpeechMetaInfo']

import os
import numpy as np
from .dataset_metainfo import DatasetMetaInfo
from .asr_dataset import AsrDataset, asr_test_transform


class LibriSpeech(AsrDataset):
    """
    LibriSpeech dataset for Automatic Speech Recognition (ASR).

    Parameters:
    ----------
    root : str, default '~/.torch/datasets/LibriSpeech'
        Path to the folder stored the dataset.
    mode : str, default 'test'
        'train', 'val', 'test', or 'demo'.
    subset : str, default 'dev-clean'
        Data subset.
    transform : function, default None
        A function that takes data and transforms it.
    """
    def __init__(self,
                 root=os.path.join("~", ".torch", "datasets", "LibriSpeech"),
                 mode="test",
                 subset="dev-clean",
                 transform=None):
        super(LibriSpeech, self).__init__(
            root=root,
            mode=mode,
            transform=transform)
        self.vocabulary = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                           'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', "'"]
        vocabulary_dict = {c: i for i, c in enumerate(self.vocabulary)}

        import soundfile

        root_dir_path = os.path.expanduser(root)
        assert os.path.exists(root_dir_path)

        data_dir_path = os.path.join(root_dir_path, subset)
        assert os.path.exists(data_dir_path)

        for speaker_id in os.listdir(data_dir_path):
            speaker_dir_path = os.path.join(data_dir_path, speaker_id)
            for chapter_id in os.listdir(speaker_dir_path):
                chapter_dir_path = os.path.join(speaker_dir_path, chapter_id)
                transcript_file_path = os.path.join(chapter_dir_path, "{}-{}.trans.txt".format(speaker_id, chapter_id))
                with open(transcript_file_path, "r") as f:
                    transcripts = dict(x.split(" ", maxsplit=1) for x in f.readlines())
                for flac_file_name in os.listdir(chapter_dir_path):
                    if flac_file_name.endswith(".flac"):
                        wav_file_name = flac_file_name.replace(".flac", ".wav")
                        wav_file_path = os.path.join(chapter_dir_path, wav_file_name)
                        if not os.path.exists(wav_file_path):
                            flac_file_path = os.path.join(chapter_dir_path, flac_file_name)
                            pcm, sample_rate = soundfile.read(flac_file_path)
                            soundfile.write(wav_file_path, pcm, sample_rate)
                        text = transcripts[wav_file_name.replace(".wav", "")]
                        text = text.strip("\n ").lower()
                        text = np.array([vocabulary_dict[c] for c in text], dtype=np.long)
                        self.data.append((wav_file_path, text))


class LibriSpeechMetaInfo(DatasetMetaInfo):
    def __init__(self):
        super(LibriSpeechMetaInfo, self).__init__()
        self.label = "LibriSpeech"
        self.short_label = "ls"
        self.root_dir_name = "LibriSpeech"
        self.dataset_class = LibriSpeech
        self.dataset_class_extra_kwargs = {"subset": "dev-clean"}
        self.ml_type = "asr"
        self.num_classes = 29
        self.val_metric_extra_kwargs = [{"vocabulary": None}]
        self.val_metric_capts = ["Val.WER"]
        self.val_metric_names = ["WER"]
        self.test_metric_extra_kwargs = [{"vocabulary": None}]
        self.test_metric_capts = ["Test.WER"]
        self.test_metric_names = ["WER"]
        self.val_transform = asr_test_transform
        self.test_transform = asr_test_transform
        self.test_net_extra_kwargs = {"from_audio": True}
        self.saver_acc_ind = 0

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        """
        Create python script parameters (for dataset specific metainfo).

        Parameters:
        ----------
        parser : ArgumentParser
            ArgumentParser instance.
        work_dir_path : str
            Path to working directory.
        """
        super(LibriSpeechMetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--subset",
            type=str,
            default="dev-clean",
            help="data subset")

    def update(self,
               args):
        """
        Update dataset metainfo after user customizing.

        Parameters:
        ----------
        args : ArgumentParser
            Main script arguments.
        """
        super(LibriSpeechMetaInfo, self).update(args)
        self.dataset_class_extra_kwargs["subset"] = args.subset

    def update_from_dataset(self,
                            dataset):
        """
        Update dataset metainfo after a dataset class instance creation.

        Parameters:
        ----------
        args : obj
            A dataset class instance.
        """
        vocabulary = dataset.vocabulary
        self.num_classes = len(vocabulary) + 1
        self.val_metric_extra_kwargs[0]["vocabulary"] = vocabulary
        self.test_metric_extra_kwargs[0]["vocabulary"] = vocabulary
