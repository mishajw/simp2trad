"""
Trains a unet_model with an L2 loss function

The training has the option to take into account the similarity to the input image. The purpose of this is to 
encourage the model to not simply repeat the input image
"""

from tf_utils import generic_runner
from unet_model import UnetModel
import logging
import tensorflow as tf
import tf_utils

log = logging.getLogger("unet_l2_trainer")


def add_arguments(parser):
    parser.add_argument("--input_similarity_cost", type=float, default=None)


def train(args, input_image, output_image, data, optimizer):
    """
    Train `unet_model` using L2
    :param args: arguments to say how to run
    :param input_image: the input image in the graph
    :param output_image: the output image in the graph
    :param data: the data to train with
    :param optimizer: how to optimize the output
    """

    unet_model = UnetModel(args, input_image)

    tf.summary.image(
        "all_images",
        tf_utils.create_generation_comparison_images(input_image, output_image, unet_model.output),
        max_outputs=8)

    def test_callback(cost_result, _):
        log.info("Cost: %s" % cost_result)

    with tf.name_scope("cost"):
        def similarity(a, b):
            return tf.reduce_mean(
                tf.sqrt(
                    tf.reduce_mean(
                        tf.square(
                            tf.abs(a - b)),
                        [1, 2, 3])))

        output_similarity = similarity(unet_model.output, output_image)

        # If we are taking into account input similarity...
        if args.input_similarity_cost is not None:
            input_similarity = similarity(unet_model.output, input_image)

            # Subtract the similarity from the cost, and also square the output similarity to get a better balance
            # between optimising for similarity to output and dissimilarity to input
            cost = tf.square(output_similarity) - (input_similarity * args.input_similarity_cost)

            tf.summary.scalar("input_similarity", input_similarity)
            tf.summary.scalar("output_similarity", output_similarity)
        else:
            cost = output_similarity

    optimization = optimizer.minimize(cost)

    tf.summary.scalar("cost", cost)
    generic_runner.run(
        args,
        "simp2trad",
        data.get_batch,
        data.get_test_data(),
        input_image,
        output_image,
        train_evaluations=[optimization],
        test_evaluations=[cost],
        test_callback=test_callback)
