from argparse import ArgumentParser
from typing import Any

from numpy import require

def get_args() -> Any:
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
        "--mode", default="play", type=str, help="reward threshold for each execution",
    )

    return parser.parse_args()
