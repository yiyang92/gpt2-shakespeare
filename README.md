# Gpt2 for Shakespeare

A simple fine-tuning of the GPT2 model on the Shakespeare's novels and FastAPI-based
service.

## Setup
The best way to use is to setup a conda environment for the project and install it.

For training:

```bash
conda env create -f requirements/train/conda-environment.yaml && conda clean -q -y -a 
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