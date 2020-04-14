"""
    CUB-200-2011 classification dataset.
"""

import os
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet.gluon.data import dataset
from .imagenet1k_cls_dataset import ImageNet1KMetaInfo


class CUB200_2011(dataset.Dataset):
    """
    CUB-200-2011 fine-grained classification dataset.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/CUB_200_2011'
        Path to the folder stored the dataset.
    mode : str, default 'train'
        'train', 'val', or 'test'.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "CUB_200_2011"),
                 mode="train",
                 transform=None):
        super(CUB200_2011, self).__init__()

        root_dir_path = os.path.expanduser(root)
        assert os.path.exists(root_dir_path)

        images_file_name = "images.txt"
        images_file_path = os.path.join(root_dir_path, images_file_name)
        if not os.path.exists(images_file_path):
            raise Exception("Images file doesn't exist: {}".format(images_file_name))

        class_file_name = "image_class_labels.txt"
        class_file_path = os.path.join(root_dir_path, class_file_name)
        if not os.path.exists(class_file_path):
            raise Exception("Image class file doesn't exist: {}".format(class_file_name))

        split_file_name = "train_test_split.txt"
        split_file_path = os.path.join(root_dir_path, split_file_name)
        if not os.path.exists(split_file_path):
            raise Exception("Split file doesn't exist: {}".format(split_file_name))

        images_df = pd.read_csv(
            images_file_path,
            sep="\s+",
            header=None,
            index_col=False,
            names=["image_id", "image_path"],
            dtype={"image_id": np.int32, "image_path": np.unicode})
        class_df = pd.read_csv(
            class_file_path,
            sep="\s+",
            header=None,
            index_col=False,
            names=["image_id", "class_id"],
            dtype={"image_id": np.int32, "class_id": np.uint8})
        split_df = pd.read_csv(
            split_file_path,
            sep="\s+",
            header=None,
            index_col=False,
            names=["image_id", "split_flag"],
            dtype={"image_id": np.int32, "split_flag": np.uint8})
        df = images_df.join(class_df, rsuffix="_class_df").join(split_df, rsuffix="_split_df")
        split_flag = 1 if mode == "train" else 0
        subset_df = df[df.split_flag == split_flag]

        self.image_ids = subset_df["image_id"].values.astype(np.int32)
        self.class_ids = subset_df["class_id"].values.astype(np.int32) - 1
        self.image_file_names = subset_df["image_path"].values.astype(np.unicode)

        images_dir_name = "images"
        self.images_dir_path = os.path.join(root_dir_path, images_dir_name)
        assert os.path.exists(self.images_dir_path)

        self._transform = transform

    def __getitem__(self, index):
        image_file_name = self.image_file_names[index]
        image_file_path = os.path.join(self.images_dir_path, image_file_name)
        img = mx.image.imread(image_file_path, flag=1)
        label = int(self.class_ids[index])
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def __len__(self):
        return len(self.image_ids)


class CUB200MetaInfo(ImageNet1KMetaInfo):
    def __init__(self):
        super(CUB200MetaInfo, self).__init__()
        self.label = "CUB200_2011"
        self.short_label = "cub"
        self.root_dir_name = "CUB_200_2011"
        self.dataset_class = CUB200_2011
        self.num_training_samples = None
        self.num_classes = 200
        self.train_metric_capts = ["Train.Err"]
        self.train_metric_names = ["Top1Error"]
        self.train_metric_extra_kwargs = [{"name": "err"}]
        self.val_metric_capts = ["Val.Err"]
        self.val_metric_names = ["Top1Error"]
        self.val_metric_extra_kwargs = [{"name": "err"}]
        self.saver_acc_ind = 0
        self.test_net_extra_kwargs = {"aux": False}
        self.load_ignore_extra = True

    def add_dataset_parser_arguments(self,
                                     parser,
                                     work_dir_path):
        super(CUB200MetaInfo, self).add_dataset_parser_arguments(parser, work_dir_path)
        parser.add_argument(
            "--no-aux",
            dest="no_aux",
            action="store_true",
            help="no `aux` mode in model")

    def update(self,
               args):
        super(CUB200MetaInfo, self).update(args)
        if args.no_aux:
            self.test_net_extra_kwargs = None
            self.load_ignore_extra = False
