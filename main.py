from executors.trainer import Trainer
from configs.experiment_config import experiment_cfg


# def train():
#     trainer = Trainer(experiment_cfg)
#     trainer.fit()


if __name__ == '__main__':
    trainer = Trainer(experiment_cfg)
    trainer.batch_overfit()
