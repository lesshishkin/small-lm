from easydict import EasyDict
import os


# TinyStories Dataset
data_cfg = EasyDict()
data_cfg.name = 'TinyStoriesDataset'
# data_cfg.path_to_data = '/kaggle/input/tinystories'
data_cfg.path_to_data = 'data/tinystories'
data_cfg.tokenized_train_data_path = os.path.join(data_cfg.path_to_data, 'en_part_1.pickle')
data_cfg.tokenized_valid_data_path = os.path.join(data_cfg.path_to_data, 'valid_v2_en_tokenized.pickle')
data_cfg.vocabulary_size = 25_000
data_cfg.special_tokens = ["<PAD>", "<UNK>", "<ВOS>", "<EOS>"]
data_cfg.start_of_word = '▁'
data_cfg.tokenizer_path = os.path.join(data_cfg.path_to_data, 'en_tokenizer.model')

# Instruct Dataset
data_cfg.instruct_dataset = EasyDict()
data_cfg.instruct_dataset.name = 'TinyStoriesInstructDataset'
data_cfg.instruct_dataset.path_to_data = '/kaggle/input/tinystories-instruct'
# data_cfg.instruct_dataset.path_to_data = 'data/tinystories'
data_cfg.instruct_dataset.tokenized_train_data_path = os.path.join(data_cfg.instruct_dataset.path_to_data, 'instruct_part_4.pickle')
data_cfg.instruct_dataset.vocabulary_size = 25_000
data_cfg.instruct_dataset.special_tokens = ["<PAD>", "<UNK>", "<ВOS>", "<EOS>"]
data_cfg.instruct_dataset.start_of_word = '▁'
data_cfg.instruct_dataset.tokenizer_path = os.path.join(data_cfg.instruct_dataset.path_to_data, 'en_tokenizer.model')
