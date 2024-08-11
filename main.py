from executors.trainer import Trainer
from executors.inferencer import Inferencer
from configs.experiment_config import experiment_cfg


def run_inference():
    inferencer = Inferencer(experiment_cfg)
    print('Prompt:')
    sentence = input()
    inferencer.predict(sentence)


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


if __name__ == '__main__':
    sft_finetuning()