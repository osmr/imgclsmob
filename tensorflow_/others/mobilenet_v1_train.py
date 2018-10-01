from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .imagenet import get_split
from .inception_preprocessing import preprocess_image
from .mobilenet_v1 import *

import tensorflow_ as tf
slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('master', '', 'Session master')
flags.DEFINE_integer('task', 0, 'Task')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps')
flags.DEFINE_integer('batch_size', 64, 'Batch size')
flags.DEFINE_integer('num_classes', 1001, 'Number of classes to distinguish')
flags.DEFINE_integer('number_of_steps', None, 'Number of training steps to perform before stopping')
flags.DEFINE_integer('image_size', 224, 'Input image resolution')
flags.DEFINE_float('depth_multiplier', 1.0, 'Depth multiplier for mobilenet')
flags.DEFINE_bool('quantize', False, 'Quantize training')
flags.DEFINE_string('fine_tune_checkpoint', '', 'Checkpoint from which to start finetuning.')
flags.DEFINE_string('checkpoint_dir', 'imgclsmob_data/tf_mobilenet_tmp/',
                    'Directory for writing training checkpoints and logs')
flags.DEFINE_string('dataset_dir', '../imgclsmob_data/imagenet/tf/', 'Location of dataset')
flags.DEFINE_integer('log_every_n_steps', 100, 'Number of steps per log')
flags.DEFINE_integer('save_summaries_secs', 100, 'How often to save summaries, secs')
flags.DEFINE_integer('save_interval_secs', 100, 'How often to save checkpoints, secs')
FLAGS = flags.FLAGS

_LEARNING_RATE_DECAY_FACTOR = 0.94


def get_learning_rate():
  if FLAGS.fine_tune_checkpoint:
    return 1e-4
  else:
    return 0.045


def get_quant_delay():
  if FLAGS.fine_tune_checkpoint:
    return 0
  else:
    return 250000


def get_dataset(split_name,
                dataset_dir,
                file_pattern=None,
                reader=None):
    """Given a dataset name and a split_name returns a Dataset.

    Args:
      split_name: A train/test split name.
      dataset_dir: The directory where the dataset files are stored.
      file_pattern: The file pattern to use for matching the dataset source files.
      reader: The subclass of tf.ReaderBase. If left as `None`, then the default
        reader defined by each dataset is used.

    Returns:
      A `Dataset` class.

    Raises:
      ValueError: If the dataset `name` is unknown.
    """
    return get_split(
        split_name,
        dataset_dir,
        file_pattern,
        reader)


def imagenet_input(is_training):
  """Data reader for imagenet.

  Reads in imagenet data and performs pre-processing on the images.

  Args:
     is_training: bool specifying if train or validation dataset is needed.
  Returns:
     A batch of images and labels.
  """
  if is_training:
    dataset = get_dataset('train', FLAGS.dataset_dir)
  else:
    dataset = get_dataset('validation', FLAGS.dataset_dir)

  provider = slim.dataset_data_provider.DatasetDataProvider(
      dataset,
      shuffle=is_training,
      common_queue_capacity=2 * FLAGS.batch_size,
      common_queue_min=FLAGS.batch_size)
  [image, label] = provider.get(['image', 'label'])

  def preprocessing_fn1(image, output_height, output_width, **kwargs):
      return preprocess_image(
          image, output_height, output_width, is_training=is_training, **kwargs)

  image_preprocessing_fn = preprocessing_fn1

  image = image_preprocessing_fn(image, FLAGS.image_size, FLAGS.image_size)

  images, labels = tf.train.batch(
      [image, label],
      batch_size=FLAGS.batch_size,
      num_threads=4,
      capacity=5 * FLAGS.batch_size)
  labels = slim.one_hot_encoding(labels, FLAGS.num_classes)
  return images, labels


def build_model():
  """Builds graph for model to train with rewrites for quantization.

  Returns:
    g: Graph with fake quantization ops and batch norm folding suitable for
    training quantized weights.
    train_tensor: Train op for execution during training.
  """
  g = tf.Graph()
  with g.as_default(), tf.device(
      tf.train.replica_device_setter(FLAGS.ps_tasks)):
    inputs, labels = imagenet_input(is_training=True)
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):
      logits, _ = mobilenet_v1.mobilenet_v1(
          inputs,
          is_training=True,
          depth_multiplier=FLAGS.depth_multiplier,
          num_classes=FLAGS.num_classes)

    tf.losses.softmax_cross_entropy(labels, logits)

    # Call rewriter to produce graph with fake quant ops and folded batch norms
    # quant_delay delays start of quantization till quant_delay steps, allowing
    # for better model accuracy.
    if FLAGS.quantize:
      tf.contrib.quantize.create_training_graph(quant_delay=get_quant_delay())

    total_loss = tf.losses.get_total_loss(name='total_loss')
    # Configure the learning rate using an exponential decay.
    num_epochs_per_decay = 2.5
    imagenet_size = 1271167
    decay_steps = int(imagenet_size / FLAGS.batch_size * num_epochs_per_decay)

    learning_rate = tf.train.exponential_decay(
        get_learning_rate(),
        tf.train.get_or_create_global_step(),
        decay_steps,
        _LEARNING_RATE_DECAY_FACTOR,
        staircase=True)
    opt = tf.train.GradientDescentOptimizer(learning_rate)

    train_tensor = slim.learning.create_train_op(
        total_loss,
        optimizer=opt)

  slim.summaries.add_scalar_summary(total_loss, 'total_loss', 'losses')
  slim.summaries.add_scalar_summary(learning_rate, 'learning_rate', 'training')
  return g, train_tensor


def get_checkpoint_init_fn():
  """Returns the checkpoint init_fn if the checkpoint is provided."""
  if FLAGS.fine_tune_checkpoint:
    variables_to_restore = slim.get_variables_to_restore()
    global_step_reset = tf.assign(tf.train.get_or_create_global_step(), 0)
    # When restoring from a floating point model, the min/max values for
    # quantized weights and activations are not present.
    # We instruct slim to ignore variables that are missing during restoration
    # by setting ignore_missing_vars=True
    slim_init_fn = slim.assign_from_checkpoint_fn(
        FLAGS.fine_tune_checkpoint,
        variables_to_restore,
        ignore_missing_vars=True)

    def init_fn(sess):
      slim_init_fn(sess)
      # If we are restoring from a floating point model, we need to initialize
      # the global step to zero for the exponential decay to result in
      # reasonable learning rates.
      sess.run(global_step_reset)
    return init_fn
  else:
    return None


def train_model():
  """Trains mobilenet_v1."""
  g, train_tensor = build_model()
  with g.as_default():
    slim.learning.train(
        train_tensor,
        FLAGS.checkpoint_dir,
        is_chief=(FLAGS.task == 0),
        master=FLAGS.master,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        number_of_steps=FLAGS.number_of_steps,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs,
        init_fn=get_checkpoint_init_fn(),
        global_step=tf.train.get_global_step())


def main(unused_arg):
  train_model()


if __name__ == '__main__':
  tf.app.run(main)
