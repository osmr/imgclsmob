import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

__all__ = ['add_dataset_parser_arguments', 'get_train_data_loader', 'get_val_data_loader']


def add_dataset_parser_arguments(parser):
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../imgclsmob_data/cifar10',
        help='path to directory with CIFAR-10 dataset')

    parser.add_argument(
        '--num-classes',
        type=int,
        default=10,
        help='number of classes')
    parser.add_argument(
        '--in-channels',
        type=int,
        default=3,
        help='number of input channels')


def get_train_data_loader(data_dir,
                          batch_size,
                          num_workers):
    mean_rgb = (0.4914, 0.4822, 0.4465)
    std_rgb = (0.2023, 0.1994, 0.2010)
    jitter_param = 0.4

    transform_train = transforms.Compose([
        transforms.RandomCrop(size=32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=jitter_param,
            contrast=jitter_param,
            saturation=jitter_param),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb),
    ])
    train_loader = torch.utils.data.DataLoader(
        dataset=datasets.CIFAR10(
            root=data_dir,
            train=True,
            transform=transform_train,
            download=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader


def get_val_data_loader(data_dir,
                        batch_size,
                        num_workers):
    mean_rgb = (0.4914, 0.4822, 0.4465)
    std_rgb = (0.2023, 0.1994, 0.2010)

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean_rgb,
            std=std_rgb),
    ])
    val_loader = torch.utils.data.DataLoader(
        dataset=datasets.CIFAR10(
            root=data_dir,
            train=False,
            transform=transform_val,
            download=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)

    return val_loader
