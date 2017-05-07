from unet_model import UnetModel
import tensorflow as tf


def main():
    input_image = tf.placeholder(dtype="float32", shape=[None, 256, 256, 1], name="input_image")
    model = UnetModel(input_image)

    print("Model input size: %s" % input_image.shape)
    print("Model output size: %s" % model.output.shape)


if __name__ == "__main__":
    main()
