"""Get arguments from command line"""
from argparse import ArgumentParser, Namespace


def get_args() -> Namespace:
    """Get arguments from command line

    Args:
        -g or --game (str):
            register environment name
        -e or --episodes (int):
            number of episodes to record
        -t or --threads (int):
            number of threads to use
        --threshold (float):
            reward threshold for each execution, if using register environment this is optional
        --mode (str):
            mode to run the script, all: record and collate, play: play the game, and 
            collate: collate the episodes

    Returns:
        Namespace: arguments from command line
    """
    parser = ArgumentParser()

    parser.add_argument(
        "-g", "--game", type=str, help="env name", required=True,
    )
    parser.add_argument(
        "-e", "--episodes", default=10, type=int, help="number of episodes"
    )
    parser.add_argument(
        "-t", "--threads", default=1, type=int, help="how many workers should the process execute",
    )
    parser.add_argument(
        "--threshold", type=float, default=None, help="reward threshold for each execution",
    )
    parser.add_argument(
        "--mode", default="play", choices=['all', 'play', 'collate'],
        type=str, help="reward threshold for each execution",
    )

    return parser.parse_args()
