"""
    KHPA dataset routines.
"""

__all__ = ['add_dataset_parser_arguments', 'get_batch_fn', 'get_train_data_source', 'get_val_data_source', 'validate']

import math
from mxnet import gluon
from gluon.weighted_random_sampler import WeightedRandomSampler
from other.gluon.khpa.khpa_cls_dataset import KHPA


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        '--data-path',
        type=str,
        default='../imgclsmob_data/khpa',
        help='path to KHPA dataset')
    parser.add_argument(
        '--split-file',
        type=str,
        default='../imgclsmob_data/khpa/split.csv',
        help='path to file with splitting training subset on training and validation ones')
    parser.add_argument(
        '--gen-split',
        action='store_true',
        help='whether generate split file')
    parser.add_argument(
        '--num-split-folders',
        type=int,
        default=10,
        help='number of folders for validation subsets')
    parser.add_argument(
        '--stats-file',
        type=str,
        default='../imgclsmob_data/khpa/stats.json',
        help='path to file with the dataset statistics')
    parser.add_argument(
        '--gen-stats',
        action='store_true',
        help='whether generate a file with the dataset statistics')

    parser.add_argument(
        '--input-size',
        type=int,
        default=224,
        help='size of the input for model')
    parser.add_argument(
        '--resize-inv-factor',
        type=float,
        default=0.875,
        help='inverted ratio for input image crop')

    parser.add_argument(
        '--num-classes',
        type=int,
        default=56,
        help='number of classes')
    parser.add_argument(
        '--in-channels',
        type=int,
        default=4,
        help='number of input channels')


def get_batch_fn():
    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        # weight = gluon.utils.split_and_load(batch[2].astype(np.float32, copy=False), ctx_list=ctx, batch_axis=0)
        return data, label
    return batch_fn


def get_train_data_loader(data_dir_path,
                          split_file_path,
                          generate_split,
                          num_split_folders,
                          stats_file_path,
                          generate_stats,
                          batch_size,
                          num_workers,
                          model_input_image_size):
    dataset = KHPA(
        root=data_dir_path,
        split_file_path=split_file_path,
        generate_split=generate_split,
        num_split_folders=num_split_folders,
        stats_file_path=stats_file_path,
        generate_stats=generate_stats,
        model_input_image_size=model_input_image_size,
        train=True)
    sampler = WeightedRandomSampler(
        length=len(dataset),
        weights=dataset.sample_weights)
    return gluon.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        # shuffle=True,
        sampler=sampler,
        last_batch="discard",
        num_workers=num_workers)


def get_val_data_loader(data_dir_path,
                        split_file_path,
                        generate_split,
                        num_split_folders,
                        stats_file_path,
                        generate_stats,
                        batch_size,
                        num_workers,
                        model_input_image_size,
                        preproc_resize_image_size):
    return gluon.data.DataLoader(
        dataset=KHPA(
            root=data_dir_path,
            split_file_path=split_file_path,
            generate_split=generate_split,
            num_split_folders=num_split_folders,
            stats_file_path=stats_file_path,
            generate_stats=generate_stats,
            preproc_resize_image_size=preproc_resize_image_size,
            model_input_image_size=model_input_image_size,
            train=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers)


def get_train_data_source(dataset_args,
                          batch_size,
                          num_workers,
                          input_image_size=(224, 224)):
    return get_train_data_loader(
        data_dir_path=dataset_args.data_path,
        split_file_path=dataset_args.split_file,
        generate_split=dataset_args.gen_split,
        num_split_folders=dataset_args.num_split_folders,
        stats_file_path=dataset_args.stats_file,
        generate_stats=dataset_args.gen_stats,
        batch_size=batch_size,
        num_workers=num_workers,
        model_input_image_size=input_image_size)


def get_val_data_source(dataset_args,
                        batch_size,
                        num_workers,
                        input_image_size=(224, 224),
                        resize_inv_factor=0.875):
    assert (resize_inv_factor > 0.0)
    if isinstance(input_image_size, int):
        input_image_size = (input_image_size, input_image_size)
    resize_value = int(math.ceil(float(input_image_size[0]) / resize_inv_factor))

    return get_val_data_loader(
        data_dir_path=dataset_args.data_path,
        split_file_path=dataset_args.split_file,
        generate_split=dataset_args.gen_split,
        num_split_folders=dataset_args.num_split_folders,
        stats_file_path=dataset_args.stats_file,
        generate_stats=dataset_args.gen_stats,
        batch_size=batch_size,
        num_workers=num_workers,
        model_input_image_size=input_image_size,
        preproc_resize_image_size=resize_value)


def validate(metric_calc,
             net,
             val_data,
             batch_fn,
             data_source_needs_reset,
             dtype,
             ctx):
    if data_source_needs_reset:
        val_data.reset()
    metric_calc.reset()
    for batch in val_data:
        data_list, labels_list = batch_fn(batch, ctx)
        onehot_outputs_list = [net(X.astype(dtype, copy=False)).reshape(0, -1, 2) for X in data_list]
        labels_list_ = [Y.reshape(-1,) for Y in labels_list]
        onehot_outputs_list_ = [Y.reshape(-1, 2) for Y in onehot_outputs_list]
        metric_calc.update(
            src_pts=labels_list_,
            dst_pts=onehot_outputs_list_)
    metric_name_value = metric_calc.get()
    return metric_name_value
