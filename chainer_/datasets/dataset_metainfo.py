"""
    Base dataset metainfo class.
"""

import os


class DatasetMetaInfo(object):
    def __init__(self):
        self.use_imgrec = False
        self.label = None
        self.root_dir_name = None
        self.root_dir_path = None
        self.dataset_class = None
        self.num_training_samples = None
        self.in_channels = None
        self.num_classes = None
        self.input_image_size = None
        self.train_metric_capts = None
        self.train_metric_names = None
        self.train_metric_extra_kwargs = None
        self.val_metric_capts = None
        self.val_metric_names = None
        self.val_metric_extra_kwargs = None
        self.test_metric_capts = None
        self.test_metric_names = None
        self.test_metric_extra_kwargs = None
        self.saver_acc_ind = None
        self.ml_type = None
        self.allow_hybridize = True
        self.train_net_extra_kwargs = None
        self.test_net_extra_kwargs = None
        self.load_ignore_extra = False

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        parser.add_argument(
            "--data-dir",
            type=str,
            default=os.path.join(work_dir_path, self.root_dir_name),
            help="path to directory with {} dataset".format(self.label))
        parser.add_argument(
            "--num-classes",
            type=int,
            default=self.num_classes,
            help="number of classes")
        parser.add_argument(
            "--in-channels",
            type=int,
            default=self.in_channels,
            help="number of input channels")

    def update(self,
               args):
        self.root_dir_path = args.data_dir
        self.num_classes = args.num_classes
        self.in_channels = args.in_channels

    def update_from_dataset(self,
                            dataset):
        """
        Update dataset metainfo after a dataset class instance creation.

        Parameters:
        ----------
        args : obj
            A dataset class instance.
        """
        pass
