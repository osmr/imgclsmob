if __name__ == '__main__' and __package__ is None:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import argparse
import os
import zipfile
import logging
import shutil
import numpy as np
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
        '--rewrite',
        action='store_true',
        help='rewrite all existed files.')

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


def get_label_list(src_dir_path):
    assert (os.path.exists(src_dir_path))
    classes_file_name = "class-descriptions-boxable.csv"
    classes_file_path = os.path.join(src_dir_path, classes_file_name)
    df = pd.read_csv(
        classes_file_path,
        header=None,
        dtype={'LabelName': np.unicode, 'LabelDesc': np.unicode})
    label_names = df[0].values.astype(np.unicode)
    assert (len(label_names) == len(np.unique(label_names)))
    np.sort(label_names)
    return label_names


def extract_data_from_archive(src_dir_path,
                              dst_dir_path,
                              rewrite,
                              remove_src,
                              archive_file_stem,
                              dst_data_dir_name):
    assert (os.path.exists(src_dir_path))
    assert (os.path.exists(dst_dir_path))
    archive_file_name = archive_file_stem + ".zip"
    logging.info('Extracting data from archive <{}>'.format(archive_file_name))

    dst_data_dir_path = os.path.join(dst_dir_path, dst_data_dir_name)
    if os.path.exists(dst_data_dir_path) and not rewrite:
        logging.info('Data are already exist...Skip.')
        return

    archive_file_path = os.path.join(src_dir_path, archive_file_name)
    with zipfile.ZipFile(archive_file_path) as zf:
        zf.extractall(dst_dir_path)

    if remove_src:
        os.remove(archive_file_path)

    if archive_file_stem == dst_data_dir_name:
        return

    if not os.path.exists(dst_data_dir_path):
        os.makedirs(dst_data_dir_path)
    src_data_dir_path = os.path.join(dst_dir_path, archive_file_stem)

    file_name_list = os.listdir(src_data_dir_path)
    for file_name in file_name_list:
        src_file_path = os.path.join(src_data_dir_path, file_name)
        dst_file_path = os.path.join(dst_data_dir_path, file_name)
        shutil.move(
            src=src_file_path,
            dst=dst_file_path)

    os.rmdir(src_data_dir_path)


def create_cls_list(src_dir_path,
                    dst_dir_path,
                    rewrite,
                    annotation_file_name,
                    cls_list_file_name):
    assert (os.path.exists(src_dir_path))
    assert (os.path.exists(dst_dir_path))
    logging.info('Creating classification list <{}>'.format(cls_list_file_name))

    cls_list_file_path = os.path.join(dst_dir_path, cls_list_file_name)
    if os.path.exists(cls_list_file_path) and not rewrite:
        logging.info('Already exist...Skip.')
        return
    annotation_file_path = os.path.join(src_dir_path, annotation_file_name)

    df = pd.read_csv(annotation_file_path)
    df2 = df.assign(Square=(df.XMax - df.XMin) * (df.YMax - df.YMin))
    df2 = df2[["ImageID", "LabelName", "Square"]]
    df2 = df2.sort_values(["ImageID", "LabelName", "Square"])
    df2 = df2.loc[df2.groupby(["ImageID"])["Square"].idxmax()]
    df2 = df2[["ImageID", "LabelName"]]
    df2.to_csv(cls_list_file_path, index=False)


def create_dataset(src_dir_path,
                   dst_dir_path,
                   rewrite,
                   remove_src,
                   src_data_dir_name,
                   dst_dataset_dir_name,
                   cls_list_file_name,
                   unique_label_names):
    assert (os.path.exists(src_dir_path))
    assert (os.path.exists(dst_dir_path))
    logging.info('Creating dataset <{}>'.format(dst_dataset_dir_name))

    dst_dataset_dir_path = os.path.join(dst_dir_path, dst_dataset_dir_name)
    if os.path.exists(dst_dataset_dir_path):
        logging.info('Already exist...Skip.')
        # if not rewrite:
        #     return
    else:
        os.makedirs(dst_dataset_dir_path)
    src_data_dir_path = os.path.join(dst_dir_path, src_data_dir_name)

    cls_list_file_path = os.path.join(dst_dir_path, cls_list_file_name)
    df = pd.read_csv(
        cls_list_file_path,
        dtype={'ImageID': np.unicode, 'LabelName': np.unicode})
    image_names = df['ImageID'].values.astype(np.unicode)
    label_names = df['LabelName'].values.astype(np.unicode)
    for label_name in unique_label_names:
        label_dir_name = label_name[3:]
        label_dir_path = os.path.join(dst_dataset_dir_path, label_dir_name)
        if not os.path.exists(label_dir_path):
            os.makedirs(label_dir_path)
    for i, (image_name, label_name) in enumerate(zip(image_names, label_names)):
        src_image_file_path = os.path.join(src_data_dir_path, "{}.jpg".format(image_name))
        assert (os.path.exists(src_image_file_path))
        label_dir_name = label_name[3:]
        label_dir_path = os.path.join(dst_dataset_dir_path, label_dir_name)
        assert (os.path.exists(label_dir_path))
        dst_image_file_path = os.path.join(label_dir_path, "{}.jpg".format(image_name))
        shutil.move(
            src=src_image_file_path,
            dst=dst_image_file_path)


def process_data(src_dir_path,
                 dst_dir_path,
                 rewrite,
                 remove_src,
                 data_name,
                 archive_file_stem_list,
                 unique_label_names):
    assert (os.path.exists(src_dir_path))
    assert (os.path.exists(dst_dir_path))
    logging.info('Process data for <{}>'.format(data_name))

    tmp_dir_name = data_name + "_tmp"
    for archive_file_stem in archive_file_stem_list:
        extract_data_from_archive(
            src_dir_path=src_dir_path,
            dst_dir_path=dst_dir_path,
            rewrite=rewrite,
            remove_src=remove_src,
            archive_file_stem=archive_file_stem,
            dst_data_dir_name=tmp_dir_name)

    annotation_file_name = data_name + "-annotations-bbox.csv"
    cls_list_file_name = data_name + "-cls.csv"
    create_cls_list(
        src_dir_path=src_dir_path,
        dst_dir_path=dst_dir_path,
        rewrite=rewrite,
        annotation_file_name=annotation_file_name,
        cls_list_file_name=cls_list_file_name)

    create_dataset(
        src_dir_path=src_dir_path,
        dst_dir_path=dst_dir_path,
        remove_src=remove_src,
        rewrite=rewrite,
        src_data_dir_name=tmp_dir_name,
        dst_dataset_dir_name=data_name,
        cls_list_file_name=cls_list_file_name,
        unique_label_names=unique_label_names)

    tmp_dir_path = os.path.join(dst_dir_path, tmp_dir_name)
    shutil.rmtree(tmp_dir_path)


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
    rewrite = args.rewrite

    unique_label_names = get_label_list(src_dir_path=src_dir_path)

    data_name_list = ["validation", "test", "train"]
    archive_file_stem_lists = [
        ["validation"],
        ["test"],
        ['train_00', 'train_01', 'train_02', 'train_03', 'train_04', 'train_05', 'train_06', 'train_07', 'train_08']
    ]

    for i in range(len(data_name_list)):
        process_data(
            src_dir_path=src_dir_path,
            dst_dir_path=dst_dir_path,
            rewrite=rewrite,
            remove_src=remove_src,
            data_name=data_name_list[i],
            archive_file_stem_list=archive_file_stem_lists[i],
            unique_label_names=unique_label_names)


if __name__ == '__main__':
    main()
