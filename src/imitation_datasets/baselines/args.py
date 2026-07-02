"""Module for benchmark arguments"""
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    """Get arguments from command line

    Args:
        -m or --methods (str):
            list of methods to benchmark

    Returns:
        Namespace: arguments from command line
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-m",
        "--methods",
        type=str,
        default="all",
        help="method's names separated by commas",
    )

    return parser.parse_args()
