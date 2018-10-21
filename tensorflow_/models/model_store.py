"""
    Model store which provides pretrained models.
"""

__all__ = ['get_model_file', 'load_model', 'download_model']

import os
import zipfile
import logging
import hashlib

_model_sha1 = {name: (error, checksum, repo_release_tag) for name, error, checksum, repo_release_tag in [
    ('resnet10', '1552', 'e2c1184863c05df5512c0747c7bcbffcb0e7bf2d', 'v0.0.72'),
    ('resnet12', '1450', '8865f58bd3daecc3a30e1f002790719f2f3f0c58', 'v0.0.72'),
    ('resnet14', '1245', '8596c8f1c48b91998456419e68e58de20d75f0d1', 'v0.0.72'),
    ('resnet16', '1105', '8ee84db280879e366e26c626cf33a47b0298a9f6', 'v0.0.72'),
    ('resnet18_wd4', '2450', 'b536eea54c6e7b93aee7c6c39bfe75cf6b985fa1', 'v0.0.72'),
    ('resnet18_wd2', '1493', 'dfb5d150464d1d0feb49755dbaa02f59cca2f4b9', 'v0.0.72'),
    ('resnet18_w3d4', '1250', '2040e339471119d43b5368ad1b050f6ee7cecaf6', 'v0.0.72'),
    ('resnet18', '0999', 'b2422403ae5b209f71b6d800491b76876ddde6d6', 'v0.0.72'),
    ('resnet34', '0793', 'aaf4f066bc0aedb131f73c52d75ca5740c96ccaa', 'v0.0.72'),
    ('resnet50', '0687', '3fd48a1afbe57e1c002bb7e9a9bc8eb341a88165', 'v0.0.72'),
    ('resnet50b', '0648', '6a8d5eda288391855996ecbea16c692aadc7a1bc', 'v0.0.72'),
    ('resnet101', '0601', '3fc260bc67ab133b39f087862f5bc70cf6aa9442', 'v0.0.72'),
    ('resnet101b', '0557', '8731697ca0ad66278cd950e93cd939cd321e7f00', 'v0.0.72'),
    ('resnet152', '0559', 'd3c4d7b20d0f5365d82149478373465414187d53', 'v0.0.72'),
    ('resnet152b', '0535', 'bcccd3d79b1528fcfffff8f8db3b6b0c2559e83a', 'v0.0.72'),
    ('preresnet18', '0988', '3295cbda0f6f03041028908fa119d0b9f1de6ce5', 'v0.0.73'),
    ('preresnet34', '0808', 'ceab73cc87b9174b37f6096278a196b4a29c47c8', 'v0.0.73'),
    ('preresnet50', '0668', '822837cf6d9d7366d9964a698ebb0dc7ef507b13', 'v0.0.73'),
    ('preresnet50b', '0661', '49f158a2b381d5d7481c29fa59fd7872d05ddeaf', 'v0.0.73'),
    ('preresnet101', '0572', 'cd61594e9e2fb758ca69a38baf31223351638c4f', 'v0.0.73'),
    ('preresnet101b', '0591', '93ae5e69d58c7f9b50e576e9254dc2eb7711231f', 'v0.0.73'),
    ('preresnet152', '0529', 'b761f286ab284b916f388cc5d6af00e5ea049081', 'v0.0.73'),
    ('preresnet152b', '0576', 'c036165cdaf177cbd44a58f88241b5161f729a33', 'v0.0.73'),
    ('preresnet200b', '0560', '881e0e2869428d89831bde0c7da219ed69236f16', 'v0.0.73'),
    ('resnext101_32x4d', '0580', 'bf746cb6f52d3329daeb66427517b03159806992', 'v0.0.74'),
    ('resnext101_64x4d', '0543', 'f51ffdb055495036e22f1867205aee69f27518b1', 'v0.0.74'),
    ('seresnet50', '0643', 'e022e5b9e58e19c692d00394c85daa57ea943b82', 'v0.0.75'),
    ('seresnet101', '0589', '305d23018de942b25df59d8ac9d2dd14374d7d28', 'v0.0.75'),
    ('seresnet152', '0578', 'd06ab6d909129693da68c552b91f3f344795114f', 'v0.0.75'),
    ('seresnext50_32x4d', '0553', '207232148cb61e6f2af55aa2e3945be52a4329d1', 'v0.0.76'),
    ('seresnext101_32x4d', '0497', '268d7d224024cd3d33ba7b259b6975bd1d064619', 'v0.0.76'),
    ('densenet121', '0782', '1bfa61d49c84f0539825a2792b6c318b55ef8938', 'v0.0.77'),
    ('densenet161', '0617', '9deca33a34a5c4a0a84f0a37920dbfd1cad85cb7', 'v0.0.77'),
    ('densenet169', '0687', '239105396a404a1673e0fc04ca0dac2aa60dabcc', 'v0.0.77'),
    ('densenet201', '0635', '5eda789595ba0b8b450705220704687fa8ea8788', 'v0.0.77'),
    ('darknet_tiny', '1751', '750ff8d9b17beb5ab88200aa787dfcb5b6ca8b36', 'v0.0.71'),
    ('darknet_ref', '1672', '3c8ed62a43b9e8934b4beb7c47ce4c7b2cdb7a64', 'v0.0.71'),
    ('squeezenet_v1_0', '1902', '694730ac41bb9698b5337aca99addb47d8e5b5aa', 'v0.0.78'),
    ('squeezenet_v1_1', '1923', '9f63fc6053e2dd2ccb49f99cac3fc1fc58429836', 'v0.0.78'),
    ('squeezeresnet_v1_1', '1792', '44c1792845488013cb3b9286c9cb7f868d590ab9', 'v0.0.79'),
    ('mobilenet_wd4', '2221', '15ee9820a315d20c732c085a4cd1edd0e3c0658a', 'v0.0.80'),
    ('mobilenet_wd2', '1484', '84d1de07c3d939c08bf1bdd4583c6b739aca0b04', 'v0.0.80'),
    ('mobilenet_w3d4', '1227', '7f3f25dc1e01ca59df627fbab0fadef7a9b9b14c', 'v0.0.80'),
    ('mobilenet_w1', '1004', 'fa4fdc2eca16d14f2e013b97d78934f7da98202b', 'v0.0.80'),
    ('fdmobilenet_wd4', '3144', '3febaec9763ae1c65b5b96c3ad2da678d31ce54d', 'v0.0.81'),
    ('fdmobilenet_wd2', '2108', '641b5d028712638cb40a26171c1926b1c31f40cb', 'v0.0.81'),
    ('fdmobilenet_w1', '1476', '4db4956bdb94e14ca0b2d0d2c73dc55dbe51c8d2', 'v0.0.81'),
    ('mobilenetv2_wd4', '2526', 'b16970032bda2300297a06bda6e0b531f820c1ee', 'v0.0.82'),
    ('mobilenetv2_wd2', '1460', '12376d2486b9409e0b41c2d46a9689c47ac32cb9', 'v0.0.82'),
    ('mobilenetv2_w3d4', '1124', '3531c997eb6a2052a715f2a37673151a1a0a075d', 'v0.0.82'),
    ('mobilenetv2_w1', '0990', 'e80f9fe41d03eed8734b7434e788e4230de42465', 'v0.0.82')]}

imgclsmob_repo_url = 'https://github.com/osmr/imgclsmob'


def get_model_name_suffix_data(model_name):
    if model_name not in _model_sha1:
        raise ValueError('Pretrained model for {name} is not available.'.format(name=model_name))
    error, sha1_hash, repo_release_tag = _model_sha1[model_name]
    return error, sha1_hash, repo_release_tag


def get_model_file(model_name,
                   local_model_store_dir_path=os.path.join('~', '.tensorflow', 'models')):
    """
    Return location for the pretrained on local file system. This function will download from online model zoo when
    model cannot be found or has mismatch. The root directory will be created if it doesn't exist.

    Parameters
    ----------
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TENSORFLOW_HOME/models
        Location for keeping the model parameters.

    Returns
    -------
    file_path
        Path to the requested pretrained model file.
    """
    error, sha1_hash, repo_release_tag = get_model_name_suffix_data(model_name)
    short_sha1 = sha1_hash[:8]
    file_name = '{name}-{error}-{short_sha1}.tf.npz'.format(
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


def load_model(sess,
               file_path,
               ignore_extra=True):
    """
    Load model state dictionary from a file.

    Parameters
    ----------
    sess: Session
        A Session to use to load the weights.
    file_path : str
        Path to the file.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    import numpy as np
    import tensorflow as tf

    assert sess is not None
    assert os.path.exists(file_path) and os.path.isfile(file_path)
    if file_path.endswith('.npy'):
        src_params = np.load(file_path, encoding='latin1').item()
    elif file_path.endswith('.npz'):
        src_params = dict(np.load(file_path))
    else:
        raise NotImplementedError
    dst_params = {v.name: v for v in tf.global_variables()}
    sess.run(tf.global_variables_initializer())
    for src_key in src_params.keys():
        if src_key in dst_params.keys():
            assert (src_params[src_key].shape == tuple(dst_params[src_key].get_shape().as_list()))
            sess.run(dst_params[src_key].assign(src_params[src_key]))
        elif not ignore_extra:
            raise Exception("The file `{}` is incompatible with the model".format(file_path))
        else:
            print("Key `{}` is ignored".format(src_key))


def download_model(sess,
                   model_name,
                   local_model_store_dir_path=os.path.join('~', '.tensorflow', 'models'),
                   ignore_extra=True):
    """
    Load model state dictionary from a file with downloading it if necessary.

    Parameters
    ----------
    sess: Session
        A Session to use to load the weights.
    model_name : str
        Name of the model.
    local_model_store_dir_path : str, default $TENSORFLOW_HOME/models
        Location for keeping the model parameters.
    ignore_extra : bool, default True
        Whether to silently ignore parameters from the file that are not present in this Module.
    """
    load_model(
        sess=sess,
        file_path=get_model_file(
            model_name=model_name,
            local_model_store_dir_path=local_model_store_dir_path),
        ignore_extra=ignore_extra)
