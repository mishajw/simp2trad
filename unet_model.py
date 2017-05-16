import logging
import tensorflow as tf
import tf_utils

log = logging.getLogger("unet_model")


def add_arguments(parser):
    parser.add_argument("--num_sections", type=int, default=2)
    parser.add_argument("--num_layers_per_section", type=int, default=1)
    parser.add_argument("--start_num_filters", type=int, default=16)


class UnetModel:
    def __init__(self, args, input_image):
        self.num_sections = args.num_sections
        self.num_layers_per_section = args.num_layers_per_section
        self.start_num_filters = args.start_num_filters

        sections = self.__create_all_sections(input_image)
        self.output = self.__create_flattening_layer(sections)

        with tf.variable_scope("output"):
            tf_utils.tensor_summary(self.output)

    def __create_all_sections(self, input_image):
        current_num_filters = self.start_num_filters
        current_input = input_image

        down_sections_output = []

        for section in range(self.num_sections):
            with tf.variable_scope("down_section_%d" % section):
                log.debug("Input shape at down section %d: %s" % (section, current_input.shape))

                with tf.variable_scope("layers"):
                    current_input = self.__create_layer_section(current_input, current_num_filters)
                    down_sections_output.append(current_input)

            # If it's not the last section
            with tf.name_scope("max_pool_%d_to_%d" % (section, section + 1)):
                if section != self.num_sections - 1:
                    current_input = tf.nn.max_pool(
                        current_input,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding="SAME")

                    current_num_filters *= 2

            log.debug("Output shape at down section %d: %s" % (section, current_input.shape))

        for section in range(1, self.num_sections):
            log.debug("Input shape at up section %d: %s" % (section, current_input.shape))

            with tf.variable_scope("upsample_%d_to_%d" % (section - 1, section)):
                upsample_size_tensor = tf.stack([
                    tf.shape(current_input)[0],
                    tf.shape(current_input)[1] * 2,
                    tf.shape(current_input)[2] * 2,
                    tf.shape(current_input)[3]])

                upsample_size_array = [
                    current_input.shape[0],
                    current_input.shape[1] * 2,
                    current_input.shape[2] * 2,
                    current_input.shape[3]]

                upsample_filter = tf.Variable(
                    tf.random_normal([2, 2, current_num_filters, current_num_filters], dtype="float32"),
                    name="upsample_filter_%d_to_%d" % (section, section + 1))

                current_input = tf.nn.conv2d_transpose(
                    current_input,
                    filter=upsample_filter,
                    output_shape=upsample_size_tensor,
                    strides=[1, 2, 2, 1],
                    padding="SAME",
                    name="upsample_%d_to_%d" % (section, section + 1))

                current_input.set_shape(upsample_size_array)

            current_num_filters = int(current_num_filters / 2)

            with tf.variable_scope("up_section_%d" % section):
                corresponding_down_section = down_sections_output[-(section + 1)]

                log.debug("Adding shapes at up section %d: %s + %s" %
                          (section, corresponding_down_section.shape, current_input.shape))

                current_input = tf.concat([corresponding_down_section, current_input], 3, name="up_down_link")

                log.debug("Result of add at up section %d: %s" % (section, current_input.shape))

                with tf.variable_scope("layers"):
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
            activation=tf.nn.relu,
            name="flattening_layer")
