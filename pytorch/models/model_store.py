"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
import hashlib
from torch.utils.model_zoo import load_url

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('resnet18', '1030', 'a516bab56c559f465bfeef3c63c15499c6f270fa', 'v0.0.3'),
    ('resnet34', '0818', '6f947d409313c862a1ef22c46e29b09c85eb9abf', 'v0.0.3'),
    ('resnet50', '0705', 'f7a2027ee704dac27a9b5184276dd7fd7969b9e8', 'v0.0.3'),
    ('resnet50b', '0665', '89691746c7fc77f79e2ce4e939001c59c99c25f2', 'v0.0.3'),
    ('resnet101', '0622', 'ab0cf005bbe9b17e53f9e3c330c6147a8c80b3a5', 'v0.0.3'),
    ('resnet101b', '0581', 'd983e68295a38498cf7b2feb4dd0dc33102201b0', 'v0.0.3'),
    ('resnet152', '0582', 'af1a3bd5285762330a8e8a5e7ec1ba23ed429e55', 'v0.0.3'),
    ('resnet152b', '0550', '216604cf5b7014f1349a879270926ef57273f952', 'v0.0.3'),
    ('preresnet18', '1057', '119bd3de6ab9abd1de9c4d67a8d6fee28eb800fd', 'v0.0.3'),
    ('preresnet34', '0841', 'b4dd761f32f603e4ea352f73ab84c0db3d5299af', 'v0.0.3'),
    ('preresnet50', '1889', '8a1091cb1a978001dc79a7114ee938b4bab31ea6', 'v0.0.3'),
    ('preresnet50b', '0687', '65be98fbe7b82c79bccd9c794ce9d9a3482aec9c', 'v0.0.3'),
    ('preresnet101', '1808', '0340579d000ae82e027eea0188fc4d6525b12470', 'v0.0.3'),
    ('preresnet101b', '0603', 'b1e37a09424dde15ecba72365d46b1f59abd479b', 'v0.0.3'),
    ('preresnet152', '1488', '0cecb4fc084715774242d3c4a95ed3592cc6ca0e', 'v0.0.3'),
    ('preresnet152b', '0591', '2c91ab2c8d90f3990e7c30fd6ee2184f6c2c3bee', 'v0.0.3')]}

imgclsmob_repo_url = 'https://github.com/osmr/tmp1'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.torch', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TORCH_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.params'.format(
        name=model_name,
        error=error,
        short_sha1=short_sha1)
    local_model_store_dir_path = os.path.expanduser(local_model_store_dir_path)
    file_path = os.path.join(local_model_store_dir_path, file_name)
    if os.path.exists(file_path):
        if _check_sha1(file_path, sha1_hash):
            return file_path
        else:
            logging.warning('Mismatch in the content of model file detected. Downloading again.')
    else:
        logging.info('Model file not found. Downloading to {}.'.format(file_path))

    if not os.path.exists(local_model_store_dir_path):
        os.makedirs(local_model_store_dir_path)

    zip_file_path = file_path + '.zip'
    load_url(
        url='{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip'.format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name))
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')


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
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

