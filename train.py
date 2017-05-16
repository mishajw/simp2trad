from scipy import misc  # TODO: Find way of reading .png without using scipy
from tf_utils import data_holder
from tf_utils import generic_runner
from tf_utils.data_holder import DataHolder
import logging
import numpy as np
import os
import tensorflow as tf
import unet_cgan_trainer
import unet_l2_trainer
import unet_model

log = logging.getLogger("simp2trad.train")


def add_arguments(parser):
    parser.add_argument("--model", type=str, default="cgan")
    parser.add_argument("--cgan", action="store_const", dest="train_type", const="cgan")
    parser.add_argument("--l2", action="store_const", dest="train_type", const="l2")
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    unet_model.add_arguments(parser)
    generic_runner.add_arguments(parser)
    data_holder.add_arguments(parser)
    unet_cgan_trainer.add_arguments(parser)
    unet_l2_trainer.add_arguments(parser)


def train(args):
    input_image = tf.placeholder(
        dtype="float32", shape=[None, args.image_size, args.image_size, 1], name="input_image")
    output_image = tf.placeholder(
        dtype="float32", shape=[None, args.image_size, args.image_size, 1], name="output_image")

    data = get_data_handler(args)

    optimizer = tf.train.AdamOptimizer(args.learning_rate)

    if args.train_type == "cgan":
        unet_cgan_trainer.train(args, input_image, output_image, data, optimizer)
    elif args.train_type == "l2":
        unet_l2_trainer.train(args, input_image, output_image, data, optimizer)


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
