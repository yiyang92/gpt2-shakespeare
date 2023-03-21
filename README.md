# Gpt2 for Shakespeare

A simple fine-tuning of the GPT2 model on the Shakespeare's novels and FastAPI-based
service.

## Setup

The best way to use is to setup a conda environment for the project and install it.

For training:

```bash
conda env create -f requirements/conda-environment.yaml && conda clean -q -y -a 
```

Then install as:

```bash
pip install -e .
```

## Train (fine-tune)

For fine-tuning a model with default parameters:

```bash
train_gpt2 -p gpt_finetune_shake_speare
```

Fine-tuned checkpoints can be found at [OneDrive](https://1drv.ms/f/s!ArZtD2QJQALxb1VWuPiAWEAzUkY?e=QNY6Ja)

## Generate (locally)

For generation, use:

```bash
generate_gpt2 -p gpt_finetune_shake_speare -c gpt2s-4.pt -i "be or not to be?"
```

where gpt2s-4.pt is the name of the fine-tuned saved parameters dictionary.

## Service

Download [checkpoint](https://1drv.ms/f/s!ArZtD2QJQALxb1VWuPiAWEAzUkY?e=QNY6Ja) and place it into checkpoints folder in the directory root.
Launch service with:

```bash
serve_gpt2
```

If initialization was successfull, you can use curl to get the generated text as:

```bash
curl -d '{"text":"be or not to be"}' -H "Content-Type: application/json" -X POST http://127.0.0.1:8000/generate
```
