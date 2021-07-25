from sys import argv

from .color_spray import color_spray


def main() -> None:
    """Main package entry point.

    gray --> rgb
    """
    try:
        input_image_path = argv[1]
        output_image_path = argv[2]

        color_spray( input_image_path, output_image_path )
    except IndexError:
        RuntimeError('Usage: INPUT_GRAY_IMAGE_PATH OUTPUT_RGB_IMAGE_PATH')
    return None

