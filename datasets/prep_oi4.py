import argparse
import os
import cv2
import logging
import hashlib
from PIL import Image
import numpy as np
import pandas as pd
from common.logger_utils import initialize_logging
if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))




def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for image classification from Open Images V4',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--src-data-dir',
        type=str,
        default='../imgclsmob_data/oi4',
        help='directory with source files.')
    parser.add_argument(
        '--dst-data-dir',
        type=str,
        default='../imgclsmob_data/oi4',
        help='directory for destination dataset and log-file.')

    parser.add_argument(
        '--logging-file-name',
        type=str,
        default='prepare.log',
        help='filename of log')
    parser.add_argument(
        '--log-packages',
        type=str,
        default='pandas, PIL, cv2',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


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
                logging.info('Downloading {} from {}...'.format(fname, url))
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
                    logging.info("download failed, retrying, {} attempt{} left".format(
                        retries,
                        's' if retries > 1 else ''))

    return fname


def create_train_data_subset(src_dir_path,
                             dst_dir_path,
                             src_data_subset_name,
                             dst_data_subset_name,
                             annotations_sha1,
                             urls_sha1,
                             num_classes,
                             max_image_count):
    logging.info("Processing <{}> subset...".format(src_data_subset_name))

    dst_data_subset_dir_path = os.path.join(dst_dir_path, dst_data_subset_name)
    if not os.path.exists(dst_data_subset_dir_path):
        os.makedirs(dst_data_subset_dir_path)

    annotations_base_url = "https://storage.googleapis.com/openimages/2018_04/{}/".format(src_data_subset_name)
    annotations_file_name = "{}-annotations-human-imagelabels.csv".format(src_data_subset_name)
    annotations_url = annotations_base_url + annotations_file_name
    annotations_file_path = os.path.join(src_dir_path, annotations_file_name)
    _download(url=annotations_url, path=annotations_file_path, sha1_hash=annotations_sha1)

    urls_base_url = "https://storage.googleapis.com/openimages/2018_04/{}/".format(src_data_subset_name)
    urls_file_name = "{}-images-with-labels-with-rotation.csv".format(src_data_subset_name)
    urls_url = urls_base_url + urls_file_name
    urls_file_path = os.path.join(src_dir_path, urls_file_name)
    _download(url=urls_url, path=urls_file_path, sha1_hash=urls_sha1)

    ann_df1 = pd.read_csv(annotations_file_path)
    ann_df1 = ann_df1.query("Confidence == 1")[["ImageID", "LabelName"]]

    ann_df2 = ann_df1.groupby(["LabelName"]).size().reset_index(name="n")
    ann_df3 = ann_df2.nlargest(num_classes, "n").sort_values(by="n")

    ann_df4 = ann_df1[ann_df1.LabelName.isin(ann_df3.LabelName)][["ImageID"]].drop_duplicates()
    ann_df5 = ann_df1[ann_df1.ImageID.isin(ann_df4.ImageID)]

    url_df1 = pd.read_csv(urls_file_path)
    url_df1 = url_df1[["ImageID", "OriginalURL", "Rotation"]]

    cls_df = pd.DataFrame(columns=["ImageID", "LabelName"])

    for id_ann_df3, row_ann_df3 in ann_df3.iterrows():
        label_name = row_ann_df3["LabelName"]
        ann_df7 = ann_df5.query("LabelName == '{}'".format(label_name))[["ImageID"]].drop_duplicates().\
            sort_values(by="ImageID")
        assert (len(ann_df7.index) <= row_ann_df3["n"])
        logging.info("Label <{}> is processed".format(label_name))

        label_dir_name = label_name[3:]
        label_dir_path = os.path.join(dst_data_subset_dir_path, label_dir_name)
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)
            logging.info("New label directory is created.")

        image_count = 0
        for id_ann_df7, row_ann_df7 in ann_df7.iterrows():
            if image_count > max_image_count:
                break
            image_id = row_ann_df7["ImageID"]
            image_file_path = os.path.join(label_dir_path, image_id + ".jpg")
            if not os.path.exists(image_file_path):
                url_df2 = url_df1.query("ImageID == '{}'".format(image_id))
                assert (len(url_df2.index) == 1)
                url = url_df2.OriginalURL.values.astype(np.unicode)[0]
                rot = url_df2.Rotation.values.astype(np.float32)[0]
                try:
                    _download(url=url, path=image_file_path)
                    img_size_bytes = os.path.getsize(image_file_path)
                    if img_size_bytes < 5000:
                        raise Exception("Image too small, size = {}".format(img_size_bytes))
                    img = Image.open(image_file_path)
                    img.verify()
                    img.close()
                    if (not np.isnan(rot)) and (rot != 0.0):
                        img = cv2.imread(image_file_path)
                        if rot == 90.0:
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        elif rot == 180.0:
                            img = cv2.rotate(img, cv2.ROTATE_180)
                        elif rot == 270.0:
                            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            raise Exception("Wrong rotate angle: {}".format(rot))
                        cv2.imwrite(image_file_path, img)
                    cls_df = cls_df.append({"ImageID": image_id, "LabelName": label_name}, ignore_index=True)
                    image_count = image_count + 1
                except Exception as err:
                    logging.warning(err)
                    if os.path.exists(image_file_path):
                        os.remove(image_file_path)
            else:
                cls_df = cls_df.append({"ImageID": image_id, "LabelName": label_name}, ignore_index=True)
                image_count = image_count + 1
            ann_df5 = ann_df5[ann_df5.ImageID != image_id]
        logging.info("Desired image count = {}, real image count = {}".format(row_ann_df3["n"], image_count))

    cls_list_file_name = "{}-cls.csv".format(src_data_subset_name)
    cls_list_file_path = os.path.join(dst_dir_path, cls_list_file_name)
    cls_df.to_csv(cls_list_file_path, index=False)


def main():
    args = parse_args()

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.src_data_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    src_dir_path = args.src_data_dir
    if not os.path.exists(src_dir_path):
        logging.error('Source directory does not exist.')
        return
    dst_dir_path = args.dst_data_dir
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)

    create_train_data_subset(
        src_dir_path=src_dir_path,
        dst_dir_path=dst_dir_path,
        src_data_subset_name="train",
        dst_data_subset_name="train",
        annotations_sha1="4203637e3fb28f3c57c7d4e0c53121cd5e9e098e",
        urls_sha1="2f64a7d611426cbc4ac3ffa029908d1871c9317d",
        num_classes=1000,
        max_image_count=5000)

    # create_data_subset(
    #     src_dir_path=src_dir_path,
    #     dst_dir_path=dst_dir_path,
    #     src_data_subset_name="validation",
    #     dst_data_subset_name="val",
    #     annotations_sha1="1f17e6a50187e7df3d323a56c6557045ed60b284",
    #     urls_sha1="97af77939cf447fa9a4384972d4ec3b231132972")


if __name__ == '__main__':
    main()
