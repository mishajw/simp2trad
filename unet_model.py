import tensorflow as tf
import logging

log = logging.getLogger("unet_model")


class UnetModel:
    def __init__(self, input_image):
        self.num_sections = 4
        self.num_layers_per_section = 2
        self.start_num_filters = 64

        sections = self.__create_all_sections(input_image)
        self.output = self.__create_flattening_layer(sections)

    def __create_all_sections(self, input_image):
        current_num_filters = self.start_num_filters
        current_input = input_image

        down_sections_output = []

        for section in range(self.num_sections):
            log.debug("Input shape at down section %d: %s" % (section, current_input.shape))

            with tf.name_scope("section_%d" % section):
                current_input = self.__create_layer_section(current_input, current_num_filters)

            down_sections_output.append(current_input)

            # If it's not the last section
            if section != self.num_sections - 1:
                current_input = tf.nn.max_pool(
                    current_input,
                    ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1],
                    padding="SAME",
                    name="max_pool_%d_to_%d" % (section, section + 1))

                current_num_filters *= 2

            log.debug("Output shape at down section %d: %s" % (section, current_input.shape))

        for section in range(self.num_sections - 1):
            log.debug("Input shape at up section %d: %s" % (section, current_input.shape))

            upsample_size = tf.stack([
                -1,
                current_input.shape[1] * 2,
                current_input.shape[2] * 2,
                current_input.shape[3]])

            upsample_filter = tf.Variable(
                tf.random_normal([2, 2, 1, current_num_filters], dtype="float32"),
                name="upsample_filter")

            current_input = tf.nn.conv2d_transpose(
                current_input,
                filter=upsample_filter,
                output_shape=upsample_size,
                strides=[1, 2, 2, 1],
                padding="SAME",
                name="max_pool_%d_to_%d" % (section, section + 1))

            current_num_filters = int(current_num_filters / 2)

            corresponding_down_section = down_sections_output[-(section + 2)]

            log.debug("Adding shapes at up section %d: %s + %s" %
                  (section, corresponding_down_section.shape, current_input.shape))

            current_input = tf.concat([corresponding_down_section, current_input], 3)

            log.debug("Result of add at up section %d: %s" % (section, current_input.shape))

            current_input = self.__create_layer_section(current_input, current_num_filters)

            log.debug("Output shape at up section %d: %s" % (section, current_input.shape))

        return current_input

    def __create_layer_section(self, section_input, layer_size):
        current_input = section_input

        for _ in range(self.num_layers_per_section):
            current_input = tf.layers.conv2d(
                current_input,
                filters=layer_size,
                kernel_size=[3, 3],
                padding="SAME",
                activation=tf.nn.relu)

        return current_input

    @staticmethod
    def __create_flattening_layer(layer_input):
        return tf.layers.conv2d(
            layer_input,
            filters=1,
            kernel_size=[1, 1],
            activation=tf.nn.relu)
