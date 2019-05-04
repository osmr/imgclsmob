"""
    CIFAR-100 classification dataset.
"""

from mxnet import gluon


class CIFAR100Fine(gluon.data.vision.CIFAR100):

    def __init__(self,
                 root,
                 train):
        super(CIFAR100Fine, self).__init__(
            root=root,
            fine_label=True,
            train=train)


class CIFAR100MetaInfo(object):
    label = "CIFAR100"
    root_dir_name = "cifar100"
    dataset_class = CIFAR100Fine
    num_training_samples = 50000
    in_channels = 3
    num_classes = 100
    input_image_size = (32, 32)
    use_imgrec = False
    val_metric_capts = ["Val.Err"]
    val_metric_names = ["err"]
    train_metric_capts = ["Train.Err"]
    train_metric_names = ["err"]
    saver_acc_ind = 0
