import argparse
import data_generator
import logging
import train

log = logging.getLogger("main")


def add_arguments(parser):
    parser.add_argument("--generate_data", action="store_true")
    parser.add_argument("--train", action="store_true")
    train.add_arguments(parser)
    data_generator.add_arguments(parser)


def main():
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    add_arguments(parser)

    args = parser.parse_args()

    if args.generate_data:
        log.info("Generating data")
        data_generator.generate(args)

    if args.train:
        log.info("Training")
        train.train(args)


if __name__ == "__main__":
    main()
