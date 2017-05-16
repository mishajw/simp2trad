import tensorflow as tf

import tf_utils


def add_arguments(parser):
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--max_pool_frequency", type=int, default=2)
    parser.add_argument("--start_filters", type=int, default=32)
    parser.add_argument("--filter_multiplier", type=int, default=2)
    parser.add_argument("--fully_connected_size", type=int, default=128)


class DiscriminatorModel:
    def __init__(self, args, truth_image, generated_image):
        self.truth_output = self.__create_prediction_model(args, truth_image, False)
        self.generated_output = self.__create_prediction_model(args, generated_image, True)

        tf.summary.scalar("truth_output_summary", tf.reduce_mean(self.truth_output))
        tf.summary.scalar("generated_output_summary", tf.reduce_mean(self.generated_output))

    @staticmethod
    def __create_prediction_model(args, truth_image, reuse):
        with tf.variable_scope("prediction", reuse=reuse):
            current_input = truth_image
            current_filters = args.start_filters

            # Add convolutional layers and max pool layers
            for layer in range(args.layers):
                current_input = tf.layers.conv2d(
                    current_input,
                    filters=current_filters,
                    kernel_size=[3, 3],
                    activation=tf.nn.relu,
                    name="conv_%d" % layer)

                # Add a max_pool layer at every `args.max_pool_frequency` layer, if we're not at the beginning or end
                if layer != 0 and layer != args.layers - 1 and (layer + 1) % args.max_pool_frequency == 0:
                    current_input = tf.nn.max_pool(
                        current_input,
                        ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1],
                        padding="SAME",
                        name="max_pool_%d_to_%d" % (layer, layer + 1))

                    current_filters = int(current_filters * args.filter_multiplier)

            # Add fully connected layers
            with tf.variable_scope("fully_connected", reuse=reuse):
                flattened_size = (current_input.shape[1] * current_input.shape[2] * current_input.shape[3]).value

                flattened = tf.reshape(
                    current_input,
                    [tf.shape(current_input)[0], flattened_size])

                weights_to_fully_connected = tf.get_variable(
                    shape=[flattened_size, args.fully_connected_size],
                    initializer=tf.random_normal_initializer(),
                    name="weights_to_fully_connected")

                biases_to_fully_connected = tf.get_variable(
                    shape=[args.fully_connected_size],
                    initializer=tf.zeros_initializer(),
                    name="biases_to_fully_connected")

                fully_connected_activation = \
                    tf.nn.sigmoid(tf.matmul(flattened, weights_to_fully_connected) + biases_to_fully_connected)

                weights_to_output = tf.get_variable(
                    shape=[args.fully_connected_size, 1],
                    initializer=tf.random_normal_initializer(),
                    name="weights_to_output")

                biases_to_output = tf.get_variable(
                    shape=[1],
                    initializer=tf.zeros_initializer(),
                    name="biases_to_output")

                with tf.variable_scope("weights_to_fully_connected_summary", reuse=reuse):
                    tf_utils.tensor_summary(weights_to_fully_connected)

                with tf.variable_scope("weights_to_output_summary", reuse=reuse):
                    tf_utils.tensor_summary(weights_to_output)

                return tf.nn.sigmoid(tf.matmul(fully_connected_activation, weights_to_output) + biases_to_output)
