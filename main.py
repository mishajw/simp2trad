from unet_model import UnetModel
import logging
import sys
import tensorflow as tf
import tf_utils
import numpy as np

log = logging.getLogger("main")


def main():
    logging.basicConfig(level=logging.DEBUG)

    input_image = tf.placeholder(dtype="float32", shape=[None, 256, 256, 1], name="input_image")
    output_image = tf.placeholder(dtype="float32", shape=[None, 256, 256, 1], name="output_image")
    model = UnetModel(input_image)

    log.debug("Model input size: %s" % input_image.shape)
    log.debug("Model output size: %s" % model.output.shape)

    def get_batch(size):
        return np.random.randn(size, 256, 256, 1), np.random.randn(size, 256, 256, 1)

    with tf.name_scope("cost"):
        cost = tf.sqrt(
            tf.reduce_sum(
                tf.square(
                    tf.abs(model.output - output_image)),
                [1, 2, 3]))

    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    tf_utils.generic_runner.run(
        "simp2trad",
        sys.argv,
        get_batch,
        get_batch(4),
        input_image,
        output_image,
        train_evaluations=[optimizer],
        test_evaluations=[cost])


if __name__ == "__main__":
    main()
