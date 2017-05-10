from tf_utils import generic_runner
import logging
import numpy as np
import tensorflow as tf
import unet_model

log = logging.getLogger("train")


def add_arguments(parser):
    parser.add_argument("--learning_rate", type=int, default=0.0001)
    unet_model.add_arguments(parser)
    generic_runner.add_arguments(parser)


def train(args):
    input_image = tf.placeholder(dtype="float32", shape=[None, 256, 256, 1], name="input_image")
    output_image = tf.placeholder(dtype="float32", shape=[None, 256, 256, 1], name="output_image")
    model = unet_model.UnetModel(args, input_image)

    log.debug("Model input size: %s" % input_image.shape)
    log.debug("Model output size: %s" % model.output.shape)

    def get_batch(size):
        return np.full((size, 256, 256, 1), 0.2), np.full((size, 256, 256, 1), 0.8)

    def test_callback(cost_result, _):
        log.info("Cost: %s" % cost_result)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(
            tf.sqrt(
                tf.reduce_mean(
                    tf.square(
                        tf.abs(model.output - output_image)),
                    [1, 2, 3])))

    tf.summary.scalar("cost", cost)

    with tf.variable_scope("image_summaries"):
        tf.summary.image("truth", output_image)
        tf.summary.image("guess", model.output)

    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    generic_runner.run(
        "simp2trad",
        args,
        get_batch,
        get_batch(4),
        input_image,
        output_image,
        train_evaluations=[optimizer],
        test_evaluations=[cost],
        test_callback=test_callback)
