"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('resnet10', '1549', 'b31f113596ba5fae687eb775e2dda81a293060d2', 'v0.0.22'),
    ('resnet12', '1556', '4b3d59a2f9047a991840b3291de0971535befeb8', 'v0.0.22'),
    ('resnet14', '1454', '7c69aaa0fe5692d007fb5676f24e20d973ffe6b6', 'v0.0.22'),
    ('resnet16', '1253', '6e751065a66acf1de7e67f418606c6c222eaafd1', 'v0.0.22'),
    ('resnet18_wd4', '2770', '72465bfeef07399a54c03d6531aaadaf7e54a94c', 'v0.0.22'),
    ('resnet18_wd2', '1646', '58261fc55698d1f5fccba964a39de30e9cf4f658', 'v0.0.22'),
    ('resnet18_w3d4', '1256', 'ce2011dfcddf9cac229d7e3a63b3764e15bcbc47', 'v0.0.22'),
    ('resnet18', '0997', '9862a84fbb34789888ffb631d64534294b312e20', 'v0.0.22'),
    ('resnet34', '0795', '0b392267b08907dc14023b24fd84df0268087002', 'v0.0.22'),
    ('resnet50', '0683', '9c795737b3ec3983de7139f817d40736fd7187fe', 'v0.0.22'),
    ('resnet50b', '0646', '225a550ed5f2d8bf0027ae7f105dbe39e041d5cc', 'v0.0.22'),
    ('resnet101', '0601', 'd8cddbea530e052e726d5a1007985beb10ec36eb', 'v0.0.22'),
    ('resnet101b', '0559', 'b5c3b4b65dd7e2c7278e35489281b7abf0fda42c', 'v0.0.22'),
    ('resnet152', '0567', '62d194fccb015a6e4517272dfa7e98a35ec6b6c6', 'v0.0.22'),
    ('resnet152b', '0539', '2b1757288ef04c89060c850d9ca725f0d589f4a5', 'v0.0.22'),
    ('squeezenet_v1_0', '1896', '6cbb35ce171a38c7dc47c402511ca2800e9d7e99', 'v0.0.20'),
    ('squeezenet_v1_1', '1925', '0ca73cf33c7d6e3d1295fcb372b08528ffafbe2a', 'v0.0.20'),
    ('mobilenet_wd4', '2428', '21ddc10dd575d589153a49c3807673a2aa579254', 'v0.0.21'),
    ('mobilenet_wd2', '1566', 'd398ee981ff7487a9b74f447dbbf6b497375d5a2', 'v0.0.21'),
    ('mobilenet_w3d4', '1252', '6675b58c7eab180a054b4999b08666fab729dbb0', 'v0.0.21'),
    ('mobilenet_w1', '1031', '3ecb405b83bbf772ef15ae304d0ccdebda7cb326', 'v0.0.21')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.chainer', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $CHAINER_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.npz'.format(
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
    _download(
        url='{repo_url}/releases/download/{repo_release_tag}/{file_name}.zip'.format(
            repo_url=imgclsmob_repo_url,
            repo_release_tag=repo_release_tag,
            file_name=file_name),
        path=zip_file_path,
        overwrite=True)
    with zipfile.ZipFile(zip_file_path) as zf:
        zf.extractall(local_model_store_dir_path)
    os.remove(zip_file_path)

    if _check_sha1(file_path, sha1_hash):
        return file_path
    else:
        raise ValueError('Downloaded file has different hash. Please try again.')


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
        fname = url.split('/')[-1]
        # Empty filenames are invalid
        assert fname, 'Can\'t construct file-name from this URL. ' \
            'Please set the `path` option manually.'
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path
    assert retries >= 0, "Number of retries should be at least 0"

    if not verify_ssl:
        warnings.warn(
            'Unverified HTTPS request is being made (verify_ssl=False). '
            'Adding certificate verification is strongly advised.')

    if overwrite or not os.path.exists(fname) or (sha1_hash and not _check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        while retries+1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print('Downloading %s from %s...'%(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url %s"%url)
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk: # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning('File {} is downloaded but the content hash does not match.'\
                                      ' The repo may be outdated or download may be incomplete. '\
                                      'If the "repo_url" is overridden, consider switching to '\
                                      'the default repo.'.format(fname))
                break
            except Exception as e:
                retries -= 1
                if retries <= 0:
                    raise e
                else:
                    print("download failed, retrying, {} attempt{} left"
                          .format(retries, 's' if retries > 1 else ''))

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
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash

