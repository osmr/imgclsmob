import argparse
import os
import zipfile
import logging
import pandas as pd

from common.logger_utils import initialize_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for image classification from Open Images V4 Bounding Boxes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/oi4bb',
        help='working data directory with source files.')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='../imgclsmob_data/oi4bb',
        help='directory of destination dataset and log-file')
    parser.add_argument(
        '--remove-archives',
        action='store_true',
        help='remove archives.')

    parser.add_argument(
        '--logging-file-name',
        type=str,
        default='prepare.log',
        help='filename of log')
    parser.add_argument(
        '--log-packages',
        type=str,
        default='pandas',
        help='list of python packages for logging')
    parser.add_argument(
        '--log-pip-packages',
        type=str,
        default='',
        help='list of pip packages for logging')
    args = parser.parse_args()
    return args


def extract_val(src_dir_path,
                dst_dir_path,
                remove_src,
                val_archive_file_name="validation.zip"):
    assert (os.path.exists(src_dir_path))
    assert (os.path.exists(dst_dir_path))

    val_archive_file_path = os.path.join(src_dir_path, val_archive_file_name)
    with zipfile.ZipFile(val_archive_file_path) as zf:
        zf.extractall(dst_dir_path)
    if remove_src:
        os.remove(val_archive_file_path)


def create_val_cls_list(src_dir_path,
                        dst_dir_path,
                        val_annotation_file_name="validation-annotations-bbox.csv",
                        val_cls_list_file_name="validation-cls.csv"):
    assert (os.path.exists(src_dir_path))
    assert (os.path.exists(dst_dir_path))

    val_annotation_file_path = os.path.join(src_dir_path, val_annotation_file_name)
    val_cls_list_file_path = os.path.join(src_dir_path, val_cls_list_file_name)

    df = pd.read_csv(val_annotation_file_path)
    df2 = df.assign(Square=(df.XMax - df.XMin) * (df.YMax - df.YMin))
    df2 = df2[["ImageID", "LabelName", "Square"]]
    df2 = df2.loc[df2.groupby(["ImageID"])["Square"].idxmax()]
    df2 = df2[["ImageID", "LabelName"]]
    df2.to_csv(val_cls_list_file_path, index=False)


def main():
    args = parse_args()

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    src_dir_path = args.data_dir
    if not os.path.exists(src_dir_path):
        logging.error('Source directory does not exist.')
        return
    dst_dir_path = args.save_dir
    if not os.path.exists(dst_dir_path):
        os.makedirs(dst_dir_path)
    remove_src = args.remove_archives

    extract_val(
        src_dir_path=src_dir_path,
        dst_dir_path=dst_dir_path,
        remove_src=remove_src)
    create_val_cls_list(
        src_dir_path=src_dir_path,
        dst_dir_path=dst_dir_path)


if __name__ == '__main__':
    main()
