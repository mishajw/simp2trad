from unet_model import UnetModel
import argparse
import logging
import numpy as np
import sys
import tensorflow as tf
import tf_utils

log = logging.getLogger("main")

parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=int, default=0.0001)


def main():
    logging.basicConfig(level=logging.DEBUG)

    args, _ = parser.parse_known_args()

    input_image = tf.placeholder(dtype="float32", shape=[None, 256, 256, 1], name="input_image")
    output_image = tf.placeholder(dtype="float32", shape=[None, 256, 256, 1], name="output_image")
    model = UnetModel(sys.argv, input_image)

    log.debug("Model input size: %s" % input_image.shape)
    log.debug("Model output size: %s" % model.output.shape)

    def get_batch(size):
        return np.random.randn(size, 256, 256, 1), np.random.randn(size, 256, 256, 1)

    def test_callback(cost_result):
        log.info("Cost: %s" % cost_result)

    with tf.name_scope("cost"):
        cost = tf.reduce_mean(
            tf.sqrt(
                tf.reduce_mean(
                    tf.square(
                        tf.abs(model.output - output_image)),
                    [1, 2, 3])))

    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    tf_utils.generic_runner.run(
        "simp2trad",
        sys.argv,
        get_batch,
        get_batch(4),
        input_image,
        output_image,
        train_evaluations=[optimizer],
        test_evaluations=[cost],
        test_callback=test_callback)


if __name__ == "__main__":
    main()
