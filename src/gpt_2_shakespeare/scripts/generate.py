import argparse
import logging

from gpt_2_shakespeare.utils.params import Registry
from gpt_2_shakespeare.inference.inference import GPTInference


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbosity", default="INFO", type=str)
    parser.add_argument(
        "-p",
        "--parameters",
        choices=Registry.get_available_params_sets(),
        type=str,
        required=True,
        help="which parameters set to use.",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="name of the checkpoint in the model checkpoints directory.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="",
        help="textual input into a language model",
    )
    parser.add_argument(
        "-n",
        "--out_len",
        type=int,
        default=None,
        help="maximum number of output words from the model.",
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
    params = Registry.get_params(parsed_args["parameters"])
    ckpt_path = params.checkpoints_dir / parsed_args["checkpoint"]
    if not ckpt_path.exists():
        raise ValueError("Checkpoints path does not exit")
    gpt_inference = GPTInference(
        params=params,
        checkpoint=ckpt_path,
        out_len=parsed_args["out_len"],
    )
    logging.info("Inference initialized")
    model_out = gpt_inference(input=parsed_args["input"])
    logging.warning(f"model output: {model_out}")
