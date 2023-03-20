import argparse
import logging

from gpt_2_shakespeare.utils.params import Registry
from gpt_2_shakespeare.training.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", default="INFO", type=str)
    parser.add_argument(
        "-p",
        "--parameters",
        choices=Registry.get_available_params_sets(),
        help="which parameters set to use",
        type=str,
        required=True,
    )
    return parser.parse_args()


def main():
    parsed_args = vars(_parse_args())
    logging_level = logging._nameToLevel[parsed_args["verbosity"]]
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s %(message)s",
        datefmt="%I:%M:%S %p",
    )
    Trainer(Registry.get_params(parsed_args["parameters"])).train()
