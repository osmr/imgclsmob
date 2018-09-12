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
    ('resnet12', '1448', '11acb729500299883bc9829028a168735275566b', 'v0.0.30'),
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
    ('preresnet18', '1034', '7d174fc2868ef8d124d42fbcf0b81c6be155264e', 'v0.0.23'),
    ('preresnet34', '0812', '829f5a239d51b9138d0b3d1aae5ae4a6082d9bc3', 'v0.0.23'),
    ('preresnet50', '0669', '40bd5e93861bf9ee8892cd766afbcc23a6d3b68c', 'v0.0.23'),
    ('preresnet50b', '0667', 'b7d221efa64231c2f3b83b197ddf570fb86a409b', 'v0.0.23'),
    ('preresnet101', '0575', 'f6f6789a895f681be08db6cb9ef184d9009a2f4b', 'v0.0.23'),
    ('preresnet101b', '0587', '4211c5abf0be8d849796a4af36729f74d90620d6', 'v0.0.23'),
    ('preresnet152', '0530', '021d99dc3004530a3a1f591e88807ce84e025033', 'v0.0.23'),
    ('preresnet152b', '0566', 'fdd337e701c06a928e0706ad98fa722508a4dabe', 'v0.0.23'),
    ('resnext101_32x4d', '0569', 'c6d1c30dcca4e83c48a2b77cfd36739a0192e244', 'v0.0.26'),
    ('resnext101_64x4d', '0543', 'dd8b7d963c2415ee1207f3705fbc33cb4ba46427', 'v0.0.26'),
    ('seresnet50', '0641', 'f3d68cfc8423b786c53390313cabfe0c4410f2d7', 'v0.0.24'),
    ('seresnet101', '0588', 'e45a9f8f09f1a7439e66032a0d79d7d5a20783b6', 'v0.0.24'),
    ('seresnet152', '0577', 'a089ba52930e9949313b9fba00a1b2e6e68f6ea4', 'v0.0.24'),
    ('seresnext50_32x4d', '0558', '5c435c1b730a0cea61b9657c8796f3c6b95ce9e8', 'v0.0.27'),
    ('seresnext101_32x4d', '0501', '98ea6fc4d36e742a01a0256707a5fa118be166dd', 'v0.0.27'),
    ('senet154', '0463', '381d2494a2ad725f62325188f94cd91c795c9902', 'v0.0.28'),
    ('densenet121', '0779', '06d5ebbf5b3f923ce8863268995ab5ed0f5b5019', 'v0.0.29'),
    ('densenet161', '0620', '6d05f3b9991bc570cb35fff22410d2065b667835', 'v0.0.29'),
    ('densenet169', '0686', '1978656b46c2b7de94c1e12350c74f492d683f7e', 'v0.0.29'),
    ('densenet201', '0629', '7770293931c03c2852115267dde3100d7140bbba', 'v0.0.29'),
    ('condensenet74_c4_g4', '0861', 'ef6077ec5348504346b3bcbaacbc308f825a9f87', 'v0.0.36'),
    ('condensenet74_c8_g8', '1043', '277fbfb898e0c8c7de8475184bcf5e651da10acc', 'v0.0.36'),
    ('dpn68', '0701', 'ad8cd4ec04a611726ee1ffcff69118a5587da691', 'v0.0.34'),
    ('dpn98', '0553', '9cd5733573f7a99062d16cd8850bb82d684704bb', 'v0.0.34'),
    ('dpn131', '0523', 'e37215991fa7e9f49245843d53de63ef1717f293', 'v0.0.34'),
    ('darknet_tiny', '1947', '0ba271d4f43e69520ab3bd07cf1f8dec5ed38c66', 'v0.0.32'),
    ('squeezenet_v1_0', '1896', '6cbb35ce171a38c7dc47c402511ca2800e9d7e99', 'v0.0.20'),
    ('squeezenet_v1_1', '1925', '0ca73cf33c7d6e3d1295fcb372b08528ffafbe2a', 'v0.0.20'),
    ('shufflenetv2_wd2', '2117', 'b19589d8fd3e6b0a05cd13b92e522f2dc2a71ca6', 'v0.0.37'),
    ('shufflenetv2_w1', '1580', 'a43c71ca8382aa21f3f12d431ee4264d608f8ad8', 'v0.0.38'),
    ('menet108_8x1_g3', '2242', '7c1b69e03fc0400e8bb4c53d38057941db4a1b93', 'v0.0.33'),
    ('menet128_8x1_g4', '2191', '4d64040c03eb2a0a728406abe54997d98f85e76c', 'v0.0.33'),
    ('menet228_12x1_g3', '1401', '07a0ace231aad769b91c5b591e14d766ca41991e', 'v0.0.33'),
    ('menet256_12x1_g4', '1391', 'ee68bd6fbb6c6c248a625435344bc615325d50a1', 'v0.0.33'),
    ('menet348_12x1_g3', '1140', '49feaea78bc6831b1c472d0aa52cbc38679918d5', 'v0.0.33'),
    ('menet352_12x1_g8', '1368', '2d523fac34b7863f0fab00fd5cf087b33c274708', 'v0.0.33'),
    ('menet456_24x1_g3', '1039', 'f68c36a2a19f1fe625a2b02cb855a42012a0a32b', 'v0.0.33'),
    ('mobilenet_wd4', '2428', '21ddc10dd575d589153a49c3807673a2aa579254', 'v0.0.21'),
    ('mobilenet_wd2', '1566', 'd398ee981ff7487a9b74f447dbbf6b497375d5a2', 'v0.0.21'),
    ('mobilenet_w3d4', '1252', '6675b58c7eab180a054b4999b08666fab729dbb0', 'v0.0.21'),
    ('mobilenet_w1', '1031', '3ecb405b83bbf772ef15ae304d0ccdebda7cb326', 'v0.0.21'),
    ('fdmobilenet_wd4', '3196', '463330f87909a68bd29664972be2cd2a716866f4', 'v0.0.25'),
    ('fdmobilenet_wd2', '2109', 'cc9bd6954b9c6818de209bdb01be9e3731e70376', 'v0.0.25'),
    ('fdmobilenet_w1', '1470', 'b40709cbc1bed29abec9f3d50ca65d5edf49f70e', 'v0.0.25'),
    ('mobilenetv2_wd4', '2549', 'b5ff8bfd6237290ecc9e2d72c03160f60ee04dd3', 'v0.0.31'),
    ('mobilenetv2_wd2', '1498', '4b767a983ab4b42f29f00ac63eb9a0a56b5af69e', 'v0.0.31'),
    ('mobilenetv2_w3d4', '1148', 'a6f852ea49ed066b2db2a43054c4e2fa7f28f8bb', 'v0.0.31'),
    ('mobilenetv2_w1', '1005', '3b6d1764934efd35d4cf402ea5194546dc5004e4', 'v0.0.31'),
    ('nasnet_a_mobile', '0849', 'c234bb7d0e491dcc5875eb1ce3ca806d2080545f', 'v0.0.35')]}

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
        assert fname, 'Can\'t construct file-name from this URL. Please set the `path` option manually.'
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
        while retries + 1 > 0:
            # Disable pyling too broad Exception
            # pylint: disable=W0703
            try:
                print('Downloading {} from {}...'.format(fname, url))
                r = requests.get(url, stream=True, verify=verify_ssl)
                if r.status_code != 200:
                    raise RuntimeError("Failed downloading url {}".format(url))
                with open(fname, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                if sha1_hash and not _check_sha1(fname, sha1_hash):
                    raise UserWarning('File {} is downloaded but the content hash does not match.'
                                      ' The repo may be outdated or download may be incomplete. '
                                      'If the "repo_url" is overridden, consider switching to '
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
