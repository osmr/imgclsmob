"""
    Script for evaluating trained model on TensorFlow (validate/test).
"""

import argparse
import tqdm
import time
import logging
from tensorpack.predict import PredictConfig, FeedfreePredictor
from tensorpack.utils.stats import RatioCounter
from tensorpack.input_source import QueueInput, StagingInput
from common.logger_utils import initialize_logging
from tensorflow_.utils_tp import prepare_tf_context, prepare_model, get_data, calc_flops


def parse_args():
    """
    Parse python script parameters.

    Returns
    -------
    ArgumentParser
        Resulted args.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a model for image classification (TensorFlow/TensorPack)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../imgclsmob_data/imagenet",
        help="training and validation pictures to use")

    parser.add_argument(
        "--data-format",
        type=str,
        default="channels_last",
        help="ordering of the dimensions in tensors. options are channels_last and channels_first")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="type of model to use. see model_provider for options")
    parser.add_argument(
        "--use-pretrained",
        action="store_true",
        help="enable using pretrained model")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="resume from previously saved parameters if not None")
    parser.add_argument(
        "--calc-flops",
        dest="calc_flops",
        action="store_true",
        help="calculate FLOPs")

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
        default="tensorflow-gpu",
        help="list of python packages for logging")
    parser.add_argument(
        "--log-pip-packages",
        type=str,
        default="tensorflow-gpu, tensorpack",
        help="list of pip packages for logging")
    args = parser.parse_args()
    return args


def test(net,
         session_init,
         val_dataflow,
         do_calc_flops=False,
         extended_log=False):
    """
    Main test routine.

    Parameters:
    ----------
    net : obj
        Model.
    session_init : SessionInit
        Session initializer.
    do_calc_flops : bool, default False
        Whether to calculate count of weights.
    extended_log : bool, default False
        Whether to log more precise accuracy values.
    """
    pred_config = PredictConfig(
        model=net,
        session_init=session_init,
        input_names=["input", "label"],
        output_names=["wrong-top1", "wrong-top5"]
    )
    err_top1 = RatioCounter()
    err_top5 = RatioCounter()

    tic = time.time()
    pred = FeedfreePredictor(pred_config, StagingInput(QueueInput(val_dataflow), device="/gpu:0"))

    for _ in tqdm.trange(val_dataflow.size()):
        err_top1_val, err_top5_val = pred()
        batch_size = err_top1_val.shape[0]
        err_top1.feed(err_top1_val.sum(), batch_size)
        err_top5.feed(err_top5_val.sum(), batch_size)

    err_top1_val = err_top1.ratio
    err_top5_val = err_top5.ratio

    if extended_log:
        logging.info("Test: err-top1={top1:.4f} ({top1})\terr-top5={top5:.4f} ({top5})".format(
            top1=err_top1_val, top5=err_top5_val))
    else:
        logging.info("Test: err-top1={top1:.4f}\terr-top5={top5:.4f}".format(
            top1=err_top1_val, top5=err_top5_val))
    logging.info("Time cost: {:.4f} sec".format(
        time.time() - tic))

    if do_calc_flops:
        calc_flops(model=net)


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

    batch_size = prepare_tf_context(
        num_gpus=args.num_gpus,
        batch_size=args.batch_size)

    net, inputs_desc = prepare_model(
        model_name=args.model,
        use_pretrained=args.use_pretrained,
        pretrained_model_file_path=args.resume.strip(),
        data_format=args.data_format)

    val_dataflow = get_data(
        is_train=False,
        batch_size=batch_size,
        data_dir_path=args.data_dir,
        input_image_size=net.image_size,
        resize_inv_factor=args.resize_inv_factor)

    assert (args.use_pretrained or args.resume.strip())
    test(
        net=net,
        session_init=inputs_desc,
        val_dataflow=val_dataflow,
        do_calc_flops=args.calc_flops,
        extended_log=True)


if __name__ == "__main__":
    main()
