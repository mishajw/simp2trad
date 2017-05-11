from scipy import misc  # TODO: Find way of reading .png without using scipy
from tf_utils import data_holder
from tf_utils import generic_runner
from tf_utils.data_holder import DataHolder
import logging
import numpy as np
import os
import tensorflow as tf
import unet_model

log = logging.getLogger("train")


def add_arguments(parser):
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    unet_model.add_arguments(parser)
    generic_runner.add_arguments(parser)
    data_holder.add_arguments(parser)


def train(args):
    input_image = tf.placeholder(
        dtype="float32", shape=[None, args.image_size, args.image_size, 1], name="input_image")
    output_image = tf.placeholder(
        dtype="float32", shape=[None, args.image_size, args.image_size, 1], name="output_image")
    model = unet_model.UnetModel(args, input_image)

    log.debug("Model input size: %s" % input_image.shape)
    log.debug("Model output size: %s" % model.output.shape)

    data = get_data_handler(args)

    def test_callback(cost_result, _):
        log.info("Cost: %s" % cost_result)

    with tf.name_scope("cost"):
        input_output_difference = tf.abs(input_image - output_image)
        truth_guess_difference = tf.abs(model.output - output_image)
        weighted_difference = (input_output_difference / 255) * truth_guess_difference

        cost = tf.reduce_mean(
            tf.sqrt(
                tf.reduce_mean(
                    tf.square(weighted_difference),
                    [1, 2, 3])))

    tf.summary.scalar("cost", cost)

    with tf.variable_scope("image_summaries"):
        tf.summary.image("input", input_image)
        tf.summary.image("truth", output_image)
        tf.summary.image("guess", model.output)
        tf.summary.image("input_output_difference", input_output_difference)
        tf.summary.image("truth_guess_difference", truth_guess_difference)
        tf.summary.image("weighted_difference", weighted_difference)

    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    generic_runner.run(
        "simp2trad",
        args,
        data.get_batch,
        data.get_test_data(),
        input_image,
        output_image,
        train_evaluations=[optimizer],
        test_evaluations=[cost],
        test_callback=test_callback)


def get_data_handler(args):
    """
    Get the DataHandler for data for the model
    :param args: arguments for reading data
    :return: the DataHandler
    """

    def file_to_data(file):
        return np.expand_dims(misc.imread(file), 2)

    def get_data(i):
        input_path = os.path.join(args.data_path, "input", "%d.png" % i)
        output_path = os.path.join(args.data_path, "output", "%d.png" % i)

        return file_to_data(input_path), file_to_data(output_path)

    data_length = len(os.listdir(os.path.join(args.data_path, "input")))

    return DataHolder(args, get_data, data_length)
