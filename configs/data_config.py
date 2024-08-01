from easydict import EasyDict
import os


# TinyStories Dataset
data_cfg = EasyDict()
data_cfg.name = 'TinyStoriesDataset'
data_cfg.path_to_data = '/kaggle/input/tinystories'
# data_cfg.path_to_data = 'data/tinystories'
data_cfg.tokenized_train_data_path = os.path.join(data_cfg.path_to_data, 'ru_part_2.pickle')
data_cfg.tokenized_valid_data_path = os.path.join(data_cfg.path_to_data, 'valid_v2_ru_tokenized.pickle')
data_cfg.vocabulary_size = 25_000
data_cfg.special_tokens = ["<PAD>", "<UNK>", "<ВOS>", "<EOS>"]
data_cfg.start_of_word = '▁'
data_cfg.tokenizer_path = os.path.join(data_cfg.path_to_data, 'ru_tinystories_tokenizer.model')
