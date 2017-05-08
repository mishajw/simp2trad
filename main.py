from unet_model import UnetModel
import tensorflow as tf
import logging

log = logging.getLogger("main")


def main():
    logging.basicConfig(level=logging.DEBUG)

    input_image = tf.placeholder(dtype="float32", shape=[None, 256, 256, 1], name="input_image")
    model = UnetModel(input_image)

    log.debug("Model input size: %s" % input_image.shape)
    log.debug("Model output size: %s" % model.output.shape)


if __name__ == "__main__":
    main()
