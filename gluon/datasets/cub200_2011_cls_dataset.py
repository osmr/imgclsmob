"""
    CUB-200-2011 classification dataset.
"""

import os
import numpy as np
import pandas as pd
import mxnet as mx
from mxnet.gluon.data import dataset


class CUB200_2011(dataset.Dataset):
    """
    Load the CUB-200-2011 fine-grained classification dataset.

    Refer to :doc:`../build/examples_datasets/imagenet` for the description of
    this dataset and how to prepare it.

    Parameters
    ----------
    root : str, default '~/.mxnet/datasets/CUB_200_2011'
        Path to the folder stored the dataset.
    train : bool, default True
        Whether to load the training or validation set.
    transform : function, default None
        A function that takes data and label and transforms them.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "CUB_200_2011"),
                 train=True,
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
        split_flag = 1 if train else 0
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


class CUB200MetaInfo(object):
    label = "CUB"
    root_dir_name = "CUB_200_2011"
    dataset_class = CUB200_2011
    num_training_samples = None
    in_channels = 3
    num_classes = 200
    input_image_size = (224, 224)
