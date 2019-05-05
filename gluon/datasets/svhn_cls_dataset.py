"""
    SVHN classification dataset.
"""

import os
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.utils import download, check_sha1
from .cifar10_cls_dataset import CIFAR10MetaInfo


class SVHN(gluon.data.dataset._DownloadedDataset):
    """
    SVHN image classification dataset from http://ufldl.stanford.edu/housenumbers/.
    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0`.

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/svhn
        Path to temp folder for storing data.
    train : bool, default True
        Whether to load the training or testing set.
    transform : function, default None
        A user defined callback that transforms each sample.
    """
    def __init__(self,
                 root=os.path.join("~", ".mxnet", "datasets", "svhn"),
                 train=True,
                 transform=None):
        self._train = train
        self._train_data = [("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "train_32x32.mat",
                             "e6588cae42a1a5ab5efe608cc5cd3fb9aaffd674")]
        self._test_data = [("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "test_32x32.mat",
                            "29b312382ca6b9fba48d41a7b5c19ad9a5462b20")]
        super(SVHN, self).__init__(root, transform)

    def _get_data(self):
        if any(not os.path.exists(path) or not check_sha1(path, sha1) for path, sha1 in
               ((os.path.join(self._root, name), sha1) for _, name, sha1 in self._train_data + self._test_data)):
            for url, _, sha1 in self._train_data + self._test_data:
                download(url=url, path=self._root, sha1_hash=sha1)

        if self._train:
            data_files = self._train_data[0]
        else:
            data_files = self._test_data[0]

        import scipy.io as sio

        loaded_mat = sio.loadmat(os.path.join(self._root, data_files[1]))

        data = loaded_mat["X"]
        data = np.transpose(data, (3, 0, 1, 2))
        self._data = mx.nd.array(data, dtype=data.dtype)

        self._label = loaded_mat["y"].astype(np.int32).squeeze()
        np.place(self._label, self._label == 10, 0)


class SVHNMetaInfo(CIFAR10MetaInfo):
    def __init__(self):
        super(SVHNMetaInfo, self).__init__()
        self.label = "SVHN"
        self.root_dir_name = "svhn"
        self.dataset_class = SVHN
        self.num_training_samples = 73257
