"""Prepare the image dataset as rec-files"""
import os
import argparse
import tarfile
import pickle
import gzip
import subprocess
from tqdm import tqdm
from mxnet.gluon.utils import check_sha1
from gluoncv.utils import download, makedirs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert dataset into record files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset-dir-path', required=True,
                        help="The directory that contains dataset")
    parser.add_argument('--num-thread', type=int, default=1,
                        help="Number of threads to use when building image record file.")
    args = parser.parse_args()
    return args


def build_rec_process(img_dir, train=False, num_thread=1):
    if not os.path.exists(img_dir):
        raise ValueError('Image dir ['+target_dir+'] doesnt exists')

    rec_dir = os.path.abspath(os.path.join(img_dir, '../rec'))
    makedirs(rec_dir)
    prefix = 'train' if train else 'val'
    print('Building ImageRecord file for ' + prefix + ' ...')
    to_path = rec_dir

    # download lst file and im2rec script
    script_path = os.path.join(rec_dir, 'im2rec.py')
    script_url = 'https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py'
    download(script_url, script_path)

    lst_path = os.path.join(rec_dir, prefix + '.lst')
    lst_url = 'http://data.mxnet.io/models/imagenet/resnet/' + prefix + '.lst'
    download(lst_url, lst_path)

    # execution
    import sys
    cmd = [
        sys.executable,
        script_path,
        rec_dir,
        img_dir,
        '--recursive',
        '--pass-through',
        '--pack-label',
        '--num-thread',
        str(num_thread)
    ]
    subprocess.call(cmd)
    os.remove(script_path)
    os.remove(lst_path)
    print('ImageRecord file for ' + prefix + ' has been built!')


def main():
    args = parse_args()

    dataset_dir_path = os.path.expanduser(args.dataset_dir_path)
    train_dir_path = os.path.join(dataset_dir_path, 'train')
    val_dir_path = os.path.join(dataset_dir_path, 'val')

    rec_dir_path = os.path.join(dataset_dir_path, 'rec')
    if not os.path.exists(rec_dir_path):
        os.makedirs(rec_dir_path)

    num_thread = args.num_thread
    #build_rec_process(train_dir_path, True, num_thread)
    build_rec_process(val_dir_path, False, num_thread)


if __name__ == '__main__':
    main()

