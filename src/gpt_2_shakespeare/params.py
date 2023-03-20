from pathlib import Path

from gpt_2_shakespeare.utils.params import Params, params_decorator, Registry


@params_decorator
class GptParams(Params):
    pretrained_model_name_or_path: str = "gpt2"
    data_path: Path = Path("")
    checkpoints_dir: Path = Path("checkpoints")

    num_train_epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 2e-5

    max_gen_len: int = 500

    def overwrite_default_attributes(self):
        pass

    def __post_init__(self):
        self.overwrite_default_attributes()


@Registry.register
class GptFinetuneShakeSpeare(GptParams):
    def overwrite_default_attributes(self):
        self.data_path = Path("data/tinyshakespeare.txt").absolute()
