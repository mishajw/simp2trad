from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import logging
import numpy as np
import os

log = logging.getLogger("data_generator")


def add_arguments(parser):
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--unicode_data_path", type=str, default="resources/unicode_translation.txt")
    parser.add_argument(
        "--font_path_simplified", type=str, default="/usr/share/fonts/adobe-source-han-sans/SourceHanSansCN-Light.otf")
    parser.add_argument(
        "--font_path_traditional", type=str, default="/usr/share/fonts/adobe-source-han-sans/SourceHanSansTW-Light.otf")
    parser.add_argument("--difference_threshold", type=float, default=3)


def generate(args):
    """
    Generate all data and write to file
    :param args: how to run, e.g. output file path and image size
    """
    unicode_data = get_unicode_data(args.unicode_data_path)

    # Set up paths for writing data
    data_path = args.data_path
    data_path_simplified = os.path.join(data_path, "input")
    data_path_traditional = os.path.join(data_path, "output")

    if not os.path.exists(data_path_simplified):
        os.makedirs(data_path_simplified)

    if not os.path.exists(data_path_traditional):
        os.makedirs(data_path_traditional)

    log.info("Going to create %d examples" % len(unicode_data))

    data = []

    for simplified, traditional in unicode_data:
        log.debug("Creating with unicode %s and %s" % (simplified, traditional))

        simplified_image = create_image_from_unicode(simplified, args.image_size, args.font_path_simplified)
        traditional_image = create_image_from_unicode(traditional, args.image_size, args.font_path_traditional)

        if are_equal(simplified_image, traditional_image, args.difference_threshold):
            continue

        data.append((simplified_image, traditional_image))

    for i, (simplified_image, traditional_image) in enumerate(data):
        simplified_image.save(os.path.join(data_path_simplified, "%d.png" % i))
        traditional_image.save(os.path.join(data_path_traditional, "%d.png" % i))


def are_equal(image_a, image_b, difference_threshold):
    """
    Check if two images are equal beyond a threshold
    :param image_a: first image for comparison
    :param image_b: second image for comparison
    :param difference_threshold: the threshold
    :return: true if they are equal
    """

    a, b = np.array(image_a), np.array(image_b)

    difference = np.sqrt(np.mean(np.square(np.abs(a - b))))

    return difference < difference_threshold


def create_image_from_unicode(unicode_number, size, font_path):
    """
    Given a unicode number, create an image with that unicode character in it
    :param unicode_number: the number for the unicode character
    :param size: the size of the image (size x size pixels)
    :param font_path: the location of the font to use to draw the unicode
    :return: PIL.Image with the unicode character drawn on it
    """

    unicode = chr(unicode_number)

    image = Image.new("L", (size, size), 255)
    draw = ImageDraw.Draw(image)
    draw.fontmode = 1  # Remove aliasing

    font = ImageFont.truetype(font_path, size)
    draw.text((0, -size * 0.28), u"%s" % unicode, font=font)

    return image


def get_unicode_data(path):
    """
    Get all the unicode data pairs in a file
    :param path: the file containing unicode data
    :return: list of pairs of data
    """
    with open(path) as file:
        lines = list(file)

    assert len(lines) == 2

    def separate(cs):
        cs = cs.strip()
        return [int(cs[i + 2:i + 6], 16) for i in range(0, len(cs), 6)]

    return list(zip(separate(lines[0]), separate(lines[1])))
