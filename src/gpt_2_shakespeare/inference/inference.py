from pathlib import Path

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from gpt_2_shakespeare.params import GptParams


class GPTInference:
    def __init__(self, params: GptParams, checkpoint: Path) -> None:
        self._tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path=params.pretrained_model_name_or_path
        )
        self._model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=params.pretrained_model_name_or_path,
            state_dict=torch.load(checkpoint),
        )
        self._model.eval()
        self._gen_len = params.max_gen_len

    def _process_text(self, input: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer_out = self._tokenizer(input, return_tensors="pt")
        input_ids = tokenizer_out["input_ids"]
        attention_mask = tokenizer_out["attention_mask"]
        return input_ids, attention_mask

    def __call__(self, input: str) -> str:
        # Inference on CPU
        input_ids, attention_mask = self._process_text(input)
        output = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self._gen_len,
            do_sample=True,
        )
        out = self._tokenizer.decode(output[0], skip_special_tokens=True)
        return " ".join([line.strip() for line in out.split()])
