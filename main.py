from executors.trainer import Trainer
from executors.inferencer import Inferencer
from configs.experiment_config import experiment_cfg


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


if __name__ == '__main__':
    prompt = """Words: dare, turkey, independent
Features: MoralValue
Summary: Lily learns a lesson about respecting others and listening to those who know more than her after she disobeys a farmer's warning and is pecked by a turkey.
Story:"""
    run_inference(prompt)