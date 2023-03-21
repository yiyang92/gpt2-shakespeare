import logging

from fastapi import FastAPI
from pydantic import BaseModel
from gpt_2_shakespeare.utils.params import Registry

from gpt_2_shakespeare.inference.inference import GPTInference


PARAMS_SET = "gpt_finetune_shake_speare"
CHECKPOINT_NAME = "gpt2s-4.pt"  # Should be placed under checkpoints folder
MAX_OUT_LEN = 500
LOG_LEVEL = "INFO"


class Input(BaseModel):
    text: str


def _init_logging() -> None:
    logging_level = logging._nameToLevel[LOG_LEVEL]
    logging.basicConfig(
        level=logging_level,
        format="%(asctime)s %(message)s",
        datefmt="%I:%M:%S %p",
    )


def _initiate_inference() -> GPTInference:
    params = Registry.get_params(PARAMS_SET)
    logging.info(f"Checkpoints should be at {params.checkpoints_dir}")
    ckpt_path = params.checkpoints_dir / CHECKPOINT_NAME
    inference = GPTInference(
        params=params,
        checkpoint=ckpt_path,
        out_len=MAX_OUT_LEN,
    )
    logging.info("Inference initialized")
    return inference


def get_app() -> tuple[FastAPI, GPTInference]:
    inference = _initiate_inference()
    app = FastAPI(title=__name__)
    return app, inference


_init_logging()
app, inference = get_app()


@app.post("/generate")
async def generate(input: Input):
    return inference(input=input.text)
