"""
    Script for evaluating trained model on Keras (validate/test).
"""

import argparse
import time
import logging
import keras
from common.logger_utils import initialize_logging
from keras_.utils import prepare_ke_context, prepare_model, get_data_rec, get_data_generator, backend_agnostic_compile


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification (Keras)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--rec-train",
        type=str,
        default="../imgclsmob_data/imagenet_rec/train.rec",
        help="the training data")
    parser.add_argument(
        "--rec-train-idx",
        type=str,
        default="../imgclsmob_data/imagenet_rec/train.idx",
        help="the index of training data")
    parser.add_argument(
        "--rec-val",
        type=str,
        default="../imgclsmob_data/imagenet_rec/val.rec",
        help="the validation data")
    parser.add_argument(
        "--rec-val-idx",
        type=str,
        default="../imgclsmob_data/imagenet_rec/val.idx",
        help="the index of validation data")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model from github repo")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="data type for training")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume from previously saved parameters if not None")

    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="size of the input for model")
    parser.add_argument(
        "--resize-inv-factor",
        type=float,
        default=0.875,
        help="inverted ratio for input image crop")

    parser.add_argument(
        "--num-gpus",
        type=int,
        default=0,
        help="number of gpus to use")
    parser.add_argument(
        "-j",
        "--num-data-workers",
        dest="num_workers",
        default=4,
        type=int,
        help="number of preprocessing workers")

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="training batch size per device (CPU/GPU)")

    parser.add_argument(
        "--save-dir",
        type=str,
        default="",
        help="directory of saved models and log-files")
    parser.add_argument(
        "--logging-file-name",
        type=str,
        default="train.log",
        help="filename of training log")

    parser.add_argument(
        "--log-packages",
        type=str,
        default="keras, mxnet, tensorflow-gpu",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="keras, keras-mxnet, mxnet, mxnet-cu100",
        help="list of pip packages for logging")
    args = parser.parse_args()
    return args


def test(net,
         val_gen,
         val_size,
         batch_size,
         num_gpus,
         calc_weight_count=False,
         extended_log=False):
    """
    Main test routine.

    Parameters:
    ----------
    net : Model
        Model.
    val_gen : generator
        Data loader.
    val_size : int
        Size of validation subset.
    batch_size : int
        Batch size.
    num_gpus : int
        Number of used GPUs.
    calc_weight_count : bool, default False
        Whether to calculate count of weights.
    extended_log : bool, default False
        Whether to log more precise accuracy values.
    """
    keras.backend.set_learning_phase(0)

    backend_agnostic_compile(
        model=net,
        loss="categorical_crossentropy",
        optimizer=keras.optimizers.SGD(
            lr=0.01,
            momentum=0.0,
            decay=0.0,
            nesterov=False),
        metrics=[keras.metrics.categorical_accuracy, keras.metrics.top_k_categorical_accuracy],
        num_gpus=num_gpus)

    # net.summary()
    tic = time.time()
    score = net.evaluate_generator(
        generator=val_gen,
        steps=(val_size // batch_size),
        verbose=True)
    err_top1_val = 1.0 - score[1]
    err_top5_val = 1.0 - score[2]

    if calc_weight_count:
        weight_count = keras.utils.layer_utils.count_params(net.trainable_weights)
        logging.info("Model: {} trainable parameters".format(weight_count))
    if extended_log:
        logging.info("Test: err-top1={top1:.4f} ({top1})\terr-top5={top5:.4f} ({top5})".format(
            top1=err_top1_val, top5=err_top5_val))
    else:
        logging.info("Test: err-top1={top1:.4f}\terr-top5={top5:.4f}".format(
            top1=err_top1_val, top5=err_top5_val))
    logging.info("Time cost: {:.4f} sec".format(
        time.time() - tic))


def main():
    """
    Main body of script.
    """
    args = parse_args()

    _, log_file_exist = initialize_logging(
        logging_dir_path=args.save_dir,
        logging_file_name=args.logging_file_name,
        script_args=args,
        log_packages=args.log_packages,
        log_pip_packages=args.log_pip_packages)

    batch_size = prepare_ke_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip())
    num_classes = net.classes if hasattr(net, "classes") else 1000
    input_image_size = net.in_size if hasattr(net, "in_size") else (args.input_size, args.input_size)

    train_data, val_data = get_data_rec(
        rec_train=args.rec_train,
        rec_train_idx=args.rec_train_idx,
        rec_val=args.rec_val,
        rec_val_idx=args.rec_val_idx,
        batch_size=batch_size,
        num_workers=args.num_workers,
        input_image_size=input_image_size,
        resize_inv_factor=args.resize_inv_factor,
        only_val=True)
    val_gen = get_data_generator(
        data_iterator=val_data,
        num_classes=num_classes)

    val_size = 50000
    assert (args.use_pretrained or args.resume.strip())
    test(
        net=net,
        val_gen=val_gen,
        val_size=val_size,
        batch_size=batch_size,
        num_gpus=args.num_gpus,
        calc_weight_count=True,
        extended_log=True)


if __name__ == "__main__":
    main()
