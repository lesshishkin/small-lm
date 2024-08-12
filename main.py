from executors.trainer import Trainer
from executors.inferencer import Inferencer
from configs.experiment_config import experiment_cfg
from data.tinystories.validation_prompts import prompts


def run_inference(prompt=None):
    inferencer = Inferencer(experiment_cfg)
    if prompt is None:
        prompt = input('Prompt:')
    inferencer.predict(prompt)


def train_previous():
    trainer = Trainer(experiment_cfg)
    trainer.load('experiments/checkpoints/last_checkpoint_2')
    trainer.fit()


def train_new():
    trainer = Trainer(experiment_cfg)
    trainer.fit()


def sft_finetuning():
    trainer = Trainer(experiment_cfg, fine_tune=True)
    trainer.load('experiments/checkpoints/last_checkpoint_en_3', only_model=True)
    trainer.fit()


def inference_valid_prompts(checkpoint_name):
    inferencer = Inferencer(experiment_cfg)
    with open(f'results_{checkpoint_name}.txt', 'a') as file:
        for prompt in prompts:
            response = inferencer.predict(prompt)
            file.write(f"{prompt}\n")
            file.write(f"{response}\n\n")


if __name__ == '__main__':
    inference_valid_prompts('sft_4_en')

