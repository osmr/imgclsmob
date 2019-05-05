"""
    Abstract classification dataset.
"""

import os


class MetaInfo(object):
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
        self.val_metric_capts = None
        self.val_metric_names = None
        self.train_metric_capts = None
        self.train_metric_names = None
        self.saver_acc_ind = None

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir):
        parser.add_argument(
            "--data-dir",
            type=str,
            default=os.path.join(work_dir, self.root_dir_name),
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
