import logging
from pathlib import Path

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    TextDataset,
    get_scheduler,
)

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from gpt_2_shakespeare.params import GptParams


class Trainer:
    def __init__(self, params: GptParams):
        self._params = params

        self._tokenizer = GPT2Tokenizer.from_pretrained(
            pretrained_model_name_or_path=params.pretrained_model_name_or_path
        )

        # Init dataset
        self._dataset = TextDataset(
            tokenizer=self._tokenizer,
            file_path=params.data_path,
            block_size=128,
        )

        # Init data loaders
        self._train_dataloader = DataLoader(
            self._dataset, batch_size=params.batch_size
        )

        # Init model from base models
        # TODO: add custom models checkpoints support
        self._model = GPT2LMHeadModel.from_pretrained(
            pretrained_model_name_or_path=params.pretrained_model_name_or_path
        )

        # Checkpoints dir
        if not params.checkpoints_dir.exists():
            params.checkpoints_dir.mkdir()

    @property
    def tokenizer(self) -> GPT2Tokenizer:
        return self._tokenizer

    @property
    def num_batches(self) -> int:
        return len(self._train_dataloader)

    @property
    def data_size(self) -> int:
        return len(self._dataset)

    def train(self) -> None:
        # Init optimizer
        optimizer = AdamW(
            self._model.parameters(),
            lr=self._params.learning_rate,
        )

        # Init scheduler
        num_epochs = self._params.num_train_epochs
        num_training_steps = num_epochs * len(self._train_dataloader)
        progress_bar = tqdm(range(num_training_steps))
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )

        # Load to device
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        model = self._model.to(device)

        # Train loop
        model.train()
        for epoch in range(num_epochs):
            for batch in self._train_dataloader:
                # batch = {k: v.to(device) for k, v in batch.items()}

                batch = batch.to(device)
                outputs = model(batch, labels=batch)
                loss = outputs.loss
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            logging.info(f"Epoch: {epoch} loss: {loss.cpu().detach().numpy()}")
            torch.save(
                model.state_dict(),
                self._params.checkpoints_dir / f"gpt2s-{epoch}.pt",
            )

        last_ckpt = self._params.checkpoints_dir / f"gpt2s-{epoch}.pt"
        logging.warning("Training completed!")
        logging.info(f"Last checkpoint: {last_ckpt}")
