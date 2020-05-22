"""Main entrypoint for model training, evaluation and testing."""
import argparse
import sys


def main(config):
    """Run the object detection model according to the specified config."""
    print(config)


def parse_args(args):
    """Parse the arguments from the command line, and return a config object \
    that stores the values of each config parameter."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Print additional debugging information.'
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
