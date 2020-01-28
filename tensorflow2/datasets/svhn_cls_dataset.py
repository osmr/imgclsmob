"""
    SVHN classification dataset.
"""

import os
import hashlib
import numpy as np
from .cifar10_cls_dataset import CIFAR10MetaInfo


def _download(url, path=None, overwrite=False, sha1_hash=None, retries=5, verify_ssl=True):
    """Download an given URL

    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    retries : integer, default 5
        The number of times to attempt the download in case of failure or non 200 return codes
    verify_ssl : bool, default True
        Verify SSL certificates.

    Returns
    -------
    str
        The file path of the downloaded file.
    """
    import warnings
    try:
        import requests
    except ImportError:
        class requests_failed_to_import(object):
            pass
        requests = requests_failed_to_import

    if path is None:
        fname = url.split("/")[-1]
        # Empty filenames are invalid
        assert fname, "Can't construct file-name from this URL. Please set the `path` option manually."
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split("/")[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            "Unverified HTTPS request is being made (verify_ssl=False). "
            "Adding certificate verification is strongly advised.")

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print("Downloading {} from {}...".format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning("File {} is downloaded but the content hash does not match."
                                      " The repo may be outdated or download may be incomplete. "
                                      "If the 'repo_url' is overridden, consider switching to "
                                      "the default repo.".format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, "s" if retries > 1 else ""))

    return fname


def _check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.

    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.

    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    sha1 = hashlib.sha1()
    with open(filename, "rb") as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


def get_svhn_data(root,
                  mode):
    """
    SVHN image classification dataset from http://ufldl.stanford.edu/housenumbers/.
    Each sample is an image (in 3D NDArray) with shape (32, 32, 3).
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0`.

    Parameters
    ----------
    root : str
        Path to temp folder for storing data.
    mode : str
        'train', 'val', or 'test'.
    """
    _train_data = [("http://ufldl.stanford.edu/housenumbers/train_32x32.mat", "train_32x32.mat",
                    "e6588cae42a1a5ab5efe608cc5cd3fb9aaffd674")]
    _test_data = [("http://ufldl.stanford.edu/housenumbers/test_32x32.mat", "test_32x32.mat",
                   "29b312382ca6b9fba48d41a7b5c19ad9a5462b20")]

    if any(not os.path.exists(path) or not _check_sha1(path, sha1) for path, sha1 in
           ((os.path.join(root, name), sha1) for _, name, sha1 in _train_data + _test_data)):
        for url, _, sha1 in _train_data + _test_data:
            _download(url=url, path=root, sha1_hash=sha1)

    if mode == "train":
        data_files = _train_data[0]
    else:
        data_files = _test_data[0]

    import scipy.io as sio
    loaded_mat = sio.loadmat(os.path.join(root, data_files[1]))

    data = loaded_mat["X"]
    data = np.transpose(data, (3, 0, 1, 2))
    label = loaded_mat["y"].astype(np.int32).squeeze()
    np.place(label, label == 10, 0)

    return data, label


class SVHNMetaInfo(CIFAR10MetaInfo):
    def __init__(self):
        super(SVHNMetaInfo, self).__init__()
        self.label = "SVHN"
        self.root_dir_name = "svhn"
        self.dataset_class = None
        self.num_training_samples = 73257
        self.train_generator = svhn_train_generator
        self.val_generator = svhn_val_generator
        self.test_generator = svhn_val_generator


def svhn_train_generator(data_generator,
                         ds_metainfo,
                         batch_size):
    """
    Create image generator for training subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    assert(ds_metainfo is not None)
    x_train, y_train = get_svhn_data(
        root=ds_metainfo.root_dir_path,
        mode="train")
    generator = data_generator.flow(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        shuffle=False)
    return generator


def svhn_val_generator(data_generator,
                       ds_metainfo,
                       batch_size):
    """
    Create image generator for validation subset.

    Parameters:
    ----------
    data_generator : ImageDataGenerator
        Image transform sequence.
    ds_metainfo : DatasetMetaInfo
        ImageNet-1K dataset metainfo.
    batch_size : int
        Batch size.

    Returns
    -------
    Sequential
        Image transform sequence.
    """
    assert(ds_metainfo is not None)
    x_test, y_test = get_svhn_data(
        root=ds_metainfo.root_dir_path,
        mode="val")
    generator = data_generator.flow(
        x=x_test,
        y=y_test,
        batch_size=batch_size,
        shuffle=False)
    return generator
