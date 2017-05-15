from tf_utils import generic_runner
from unet_model import UnetModel
import discriminator_model
import numpy as np
import tensorflow as tf
import tf_utils


def add_arguments(parser):
    parser.add_argument("--training_ratio", type=float, default=2)
    discriminator_model.add_arguments(parser)


def train(args, input_image, output_image, data, optimizer):
    generator_random_noise = tf.placeholder(dtype="float32", shape=input_image.shape)

    with tf.variable_scope("unet_model"):
        unet_model = UnetModel(args, input_image, generator_random_noise)

    tf.summary.image(
        "all_images",
        tf_utils.create_generation_comparison_images(input_image, output_image, unet_model.output),
        max_outputs=8)

    with tf.variable_scope("discriminator"):
        discriminator = discriminator_model.DiscriminatorModel(
            args, truth_image=output_image, generated_image=unet_model.output)

    trainable_variables = tf.trainable_variables()

    generator_variables = \
        [variable for variable in trainable_variables if not variable.name.startswith("discriminator")]

    discriminator_variables = \
        [variable for variable in trainable_variables if variable.name.startswith("discriminator")]

    with tf.variable_scope("generator_cost"):
        generator_cost = tf.reduce_mean(-tf.log(discriminator.generated_output))
        tf.summary.scalar("summary", generator_cost)
        generator_optimization = optimizer.minimize(generator_cost, var_list=generator_variables)

    with tf.variable_scope("discriminator_cost"):
        with tf.variable_scope("truth"):
            truth_discriminator_cost = \
                tf.reduce_mean(-tf.log(discriminator.truth_output))
            tf.summary.scalar("summary", truth_discriminator_cost)

        with tf.variable_scope("generated"):
            generated_discriminator_cost = \
                tf.reduce_mean(tf.log(discriminator.generated_output))
            tf.summary.scalar("summary", generated_discriminator_cost)

        discriminator_cost = truth_discriminator_cost + generated_discriminator_cost
        tf.summary.scalar("summary", discriminator_cost)

        discriminator_optimization = optimizer.minimize(discriminator_cost, var_list=discriminator_variables)

    generator_training_amount = 0
    discriminator_training_amount = 0

    def get_random_noise(batch_size):
        return np.random.randn(batch_size, input_image.shape[1], input_image.shape[2], input_image.shape[3])

    def train_step(session, step, training_input, training_output, summary_writer, all_summaries):
        nonlocal generator_training_amount, discriminator_training_amount

        random_noise = get_random_noise(len(training_input))

        if generator_training_amount <= discriminator_training_amount * args.training_ratio:
            # Train generator
            _, summary_result = session.run(
                [generator_optimization, all_summaries], {
                    input_image: training_input,
                    output_image: training_output,
                    generator_random_noise: random_noise
                })

            summary_writer.add_summary(summary_result, step)

            generator_training_amount += 1

        if generator_training_amount >= discriminator_training_amount * args.training_ratio:
            # Train discriminator
            session.run(
                [discriminator_optimization], {
                    input_image: training_output,
                    output_image: training_output,
                    generator_random_noise: random_noise
                })

            discriminator_training_amount += 1

    def test_step(session, step, testing_input, testing_output, summary_writer, all_summaries):
        generator_cost_result, truth_discriminator_cost_result, generated_discriminator_cost_result, summary_result = \
            session.run(
                [generator_cost, truth_discriminator_cost, generated_discriminator_cost, all_summaries], {
                    input_image: testing_input,
                    output_image: testing_output,
                    generator_random_noise: get_random_noise(len(testing_input))
                })

        print("Testing at step %d: "
              "Generator cost %f; "
              "Discriminator cost (truth data) %f; "
              "Discriminator cost (generated data) %f" %
              (step, generator_cost_result, truth_discriminator_cost_result, generated_discriminator_cost_result))

        summary_writer.add_summary(summary_result, step)

    generic_runner.run_with_test_train_steps(
        args, "simp2trad", data.get_batch, data.get_test_data(), train_step_fn=train_step, test_step_fn=test_step)
