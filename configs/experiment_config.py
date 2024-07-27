import os

from easydict import EasyDict

from configs.data_config import data_cfg
from configs.model_config import model_cfg
from utils.enums import InferenceType

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))

experiment_cfg = EasyDict()
experiment_cfg.seed = 0
experiment_cfg.num_epochs = 1
experiment_cfg.num_steps = 169_777

# Train parameters
experiment_cfg.train = EasyDict()
experiment_cfg.train.batch_size = 16
experiment_cfg.train.learning_rate = 1e-5
experiment_cfg.train.warmup_steps = 500
experiment_cfg.train.label_smoothing = 0.0
experiment_cfg.train.optimizer = 'AdamW'  # from (Adam, AdamW)
experiment_cfg.train.optimizer_params = {
    'Adam': {'betas': (0.9, 0.999), 'eps': 1e-8}, 'AdamW': {'betas': (0.9, 0.98), 'eps': 1e-9, 'weight_decay': 1e-4}
}
experiment_cfg.train.continue_train = False
experiment_cfg.train.checkpoint_from_epoch = None
experiment_cfg.train.log_frequency = 10
experiment_cfg.train.log_window = 10
experiment_cfg.train.validation_frequency = 5000
experiment_cfg.train.validation_batch_size = 16
experiment_cfg.train.inference_frequency = 2
experiment_cfg.train.log_output_frequency = 1000

# Overfit parameters
experiment_cfg.overfit = EasyDict()
experiment_cfg.overfit.num_iterations = 500

# Neptune parameters
experiment_cfg.neptune = EasyDict()
experiment_cfg.neptune.env_path = os.path.join(ROOT_DIR, '.env')
experiment_cfg.neptune.project = 'lesshishkin/tinyllm'
experiment_cfg.neptune.experiment_name = ''
experiment_cfg.neptune.run_id = None
experiment_cfg.neptune.dependencies_path = os.path.join(ROOT_DIR, 'requirements.txt')

# Checkpoints parameters
experiment_cfg.checkpoints_dir = os.path.join(ROOT_DIR, 'experiments', experiment_cfg.neptune.experiment_name,
                                              'checkpoints')
experiment_cfg.checkpoint_save_frequency = 10_000
experiment_cfg.checkpoint_name = 'checkpoint_%s'
experiment_cfg.best_checkpoint_name = 'best_checkpoint'

# Inference parameters
experiment_cfg.inference = EasyDict()
# experiment_cfg.inference.type = InferenceType.temperature
experiment_cfg.inference.type = InferenceType.greedy
experiment_cfg.inference.temperature_value = 1
experiment_cfg.inference.eps = 1e-9
experiment_cfg.inference.stop_predict = 300  # Maximum number of inference steps (i.e. generated sequence length)
experiment_cfg.inference.model_path = os.path.join(experiment_cfg.checkpoints_dir, 'last_checkpoint_2')
experiment_cfg.model = model_cfg
experiment_cfg.data = data_cfg
