"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('alexnet', '2126', '56fb1c54f3fd3b95ac6b25b441372770e7f9e0c9', 'v0.0.121'),
    ('vgg11', '1175', 'daa3c646109c9ade6f5d68ebc1d120c382d7e847', 'v0.0.122'),
    ('vgg13', '1112', '90b447ec7667cef7493e0213e34451e09f8a4ffb', 'v0.0.122'),
    ('vgg16', '0869', '13d19be6eea8e6b5905e2e10e8c2815adeb4764f', 'v0.0.122'),
    ('vgg19', '0823', 'cab851b8de19912ac056a4ef424b4260760b3026', 'v0.0.122'),
    ('bn_vgg11b', '1057', '8b6a294a4c9d2455851fa831e1793d204d3f8fa8', 'v0.0.123'),
    ('bn_vgg13b', '1016', 'b26cafd39447f039a8124dda8a177b2dc72d98f3', 'v0.0.123'),
    ('bn_vgg16b', '0865', '2272fdd110106e830920bfdd5999fa58737f20e4', 'v0.0.123'),
    ('bn_vgg19b', '0814', '852e2ca228821f3ea1d32a12ce47a9a001236f5e', 'v0.0.123'),
    ('resnet10', '1554', '294a0786be0cb61ed9add17f85917949423648ba', 'v0.0.49'),
    ('resnet12', '1445', '285da75beb82032d5a71e0accf589f2912559020', 'v0.0.49'),
    ('resnet14', '1242', 'e2ffca6eeace503f514bfd0af07cc2d40dc0814e', 'v0.0.49'),
    ('resnet16', '1109', '8f70f97e26f2a03df33670a8c23e514f6d1af196', 'v0.0.49'),
    ('resnet18_wd4', '2445', 'dd6ba54d866c896de0355ab01a062ef0539cb9ae', 'v0.0.49'),
    ('resnet18_wd2', '1496', '9bc78e3b04db804290ce654d067d9bec3aec3a0f', 'v0.0.49'),
    ('resnet18_w3d4', '1254', 'f6374cc3b848ce71e1b0d99e07aa30c8bf9c4bdb', 'v0.0.49'),
    ('resnet18', '0994', '3ff2352af4e192a23fc37fc4a919a5f0a500c788', 'v0.0.49'),
    ('resnet34', '0792', '3ea662f5aeb33d60b762204f7b00d2f08979a356', 'v0.0.49'),
    ('resnet50', '0687', '9eb5e8d7e75568a16b9d1f50a0d148a8c2baa13a', 'v0.0.49'),
    ('resnet50b', '0644', 'fd813b71426e03d863e3cbfc4d057cb9dafaba88', 'v0.0.49'),
    ('resnet101', '0599', 'ab4289478d017d3b929011c26fbcf8b54dd8ce07', 'v0.0.49'),
    ('resnet101b', '0560', '241918fa75a8fb44b3e5ee90061859c0764b8202', 'v0.0.49'),
    ('resnet152', '0561', '001efbfffb6907bafdf5e54794f8ecb8ab831ce0', 'v0.0.49'),
    ('resnet152b', '0537', '8870623c9bdfbf0a9b69df350cbdb19dabd3958c', 'v0.0.49'),
    ('preresnet18', '0988', '36f6c05c959feb51e542c52ac5c6a4a5ef771d68', 'v0.0.50'),
    ('preresnet34', '0811', '1663d6958d1895e59da2e6e5f6acc7e768cacf58', 'v0.0.50'),
    ('preresnet50', '0668', '90326d19b9fc870ba91ca9b5a99544cc3a1f8a43', 'v0.0.50'),
    ('preresnet50b', '0663', 'c30588eeb1daa0b2761c83f21ede6dee8e235d51', 'v0.0.50'),
    ('preresnet101', '0575', '5dff088de44ce782ac72b4c5fbc03de83b379d1c', 'v0.0.50'),
    ('preresnet101b', '0588', 'fad1f60cb51fe6afedc982cb984dd736070cfc0b', 'v0.0.50'),
    ('preresnet152', '0531', 'a5ac128d79e3e6eb01a4a5eeb571e252482edbc7', 'v0.0.50'),
    ('preresnet152b', '0576', 'ea9dda1ed3497452723bc21d4b189ae43ea497ed', 'v0.0.50'),
    ('preresnet200b', '0564', '9172d4c02aef8c6ff1504dcf3c299518325afae0', 'v0.0.50'),
    ('resnext101_32x4d', '0578', '7623f640632e32961869c12310527066eac7519e', 'v0.0.51'),
    ('resnext101_64x4d', '0541', '7b58eaae7e86f487d3cb9bdf484bcfae52b8ca74', 'v0.0.51'),
    ('seresnet50', '0643', 'fabfa4062a7724ea31752434a687e1837eb30932', 'v0.0.52'),
    ('seresnet101', '0588', '933d34159345f5cf9a663504f03cd423b527aeac', 'v0.0.52'),
    ('seresnet152', '0577', 'd25ced7d6369f3d14ed2cfe54fb70bc4be9c68e0', 'v0.0.52'),
    ('seresnext50_32x4d', '0557', '997ef4dd811c2f126f685d91af61a0dad96a7d26', 'v0.0.53'),
    ('seresnext101_32x4d', '0499', '59e4e5846d8e78601c255102f302c89b2d9402e7', 'v0.0.53'),
    ('senet154', '0465', '962aeede627d5196eaf0cf8c25b6f7281f62e9ea', 'v0.0.54'),
    ('densenet121', '0780', '52b0611c336904038764ed777ccad356ede89b21', 'v0.0.55'),
    ('densenet161', '0618', '070fcb455db45c45aeb67fa4fb0fda4a89b7ef45', 'v0.0.55'),
    ('densenet169', '0689', 'ae41b4a6e3008020b71ec75705cefe35b244dc80', 'v0.0.55'),
    ('densenet201', '0635', 'cf3afbb259163bb76eee519f9d43ddbdf0a583b9', 'v0.0.55'),
    ('darknet_tiny', '1746', '147e949b779914331f740badc82339a2fb5bcb11', 'v0.0.69'),
    ('darknet_ref', '1668', '2ef080bb6f470e5ffb0c625ff3047de97cfeb6e2', 'v0.0.64'),
    ('squeezenet_v1_0', '1756', 'a489092344c0214c402655210e031a1441bd70d1', 'v0.0.128'),
    ('squeezenet_v1_1', '1739', 'b9a8f9eae7a48d053895fe4a362d1d8eb592e994', 'v0.0.88'),
    ('squeezeresnet_v1_1', '1784', '43ee9cbbb91046f5316ee14e227f8323b1801b51', 'v0.0.70'),
    ('shufflenetv2_wd2', '1840', '9b4b0964301ba3f2e393c3d3b9a43de3bb480b05', 'v0.0.90'),
    ('shufflenetv2_w1', '1338', 'cb35f09e4c1876b87ae01e51d98c6699ae4adb1e', 'v0.0.93'),
    ('shufflenetv2_w3d2', '1247', 'f7f813b4b9de6d2b7b36ccec69fe9120f0669edc', 'v0.0.65'),
    ('shufflenetv2_w2', '1223', '632914682dbfe87c883d35e89e14561b4da9e72e', 'v0.0.84'),
    ('menet108_8x1_g3', '2031', 'a4d43433e2d9f770c406b3f780a8960609c0e9b8', 'v0.0.89'),
    ('menet128_8x1_g4', '1914', '5bb8f2287930abb3e921842f053d6592f7034ea7', 'v0.0.103'),
    ('menet228_12x1_g3', '1401', '954b3ba0cd681c3b63b03f77d4ff1dea03207664', 'v0.0.58'),
    ('menet256_12x1_g4', '1391', 'a63a606a2f344aa322f21d46f1192fd9d927294e', 'v0.0.58'),
    ('menet348_12x1_g3', '1142', '0715c86612e932522c3c4911104d941a77fd53b7', 'v0.0.58'),
    ('menet352_12x1_g8', '1375', '9007c933f9e4d1d78bd88e84aa892c127288ca2d', 'v0.0.58'),
    ('menet456_24x1_g3', '1044', 'c090af591e6b0a996d92fe4bc3332bb271046785', 'v0.0.58'),
    ('mobilenet_wd4', '2217', 'fb7abda85e29c592f0196ff4a76b9ee2951c6e3c', 'v0.0.62'),
    ('mobilenet_wd2', '1481', 'b78fec3399bcb41b633216348ba4e0d9a142f90e', 'v0.0.66'),
    ('mobilenet_w3d4', '1228', '09a1eb5582629d4bbb4a10ea008b1ca6cdc9acb2', 'v0.0.59'),
    ('mobilenet_w1', '1003', 'ec69d89b0de5faa9207f3032b987229b929007e4', 'v0.0.59'),
    ('fdmobilenet_wd4', '3137', '153934e4ba7a340644dedebc0c35096adfc7eeb6', 'v0.0.68'),
    ('fdmobilenet_wd2', '1969', '5678a212ba44317306e2960ddeed6a5c0489122f', 'v0.0.83'),
    ('fdmobilenet_w1', '1374', '21b24355df5ebfff4eee5deae82d6e8003ad8d45', 'v0.0.129'),
    ('mobilenetv2_wd4', '2524', 'a8ea2889320ff7a449e5260912c9f749e672b128', 'v0.0.61'),
    ('mobilenetv2_wd2', '1465', '774d5bca86ba482312fdbd33dce5e3578e98cd80', 'v0.0.61'),
    ('mobilenetv2_w3d4', '1126', 'f2f664dae078f6f5aa3d73760b8eb182df78a9bf', 'v0.0.61'),
    ('mobilenetv2_w1', '0990', 'cbb8be963ca651c18f7d8d0e5df0510087d0d755', 'v0.0.61'),
    ('igcv3_w1', '0955', 'e2bde79d84c2edf7659efe4a65112de20bc76dba', 'v0.0.126'),
    ('mnasnet', '1145', '11b6acf13ce516b7b28875c5a6d6932a1aa0b96a', 'v0.0.117')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.keras', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $KERAS_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.h5'.format(
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
