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


if __name__ == '__main__':
    train_previous()
