from easydict import EasyDict
import os


# TinyStories Dataset
data_cfg = EasyDict()
data_cfg.name = 'TinyStoriesDataset'
# data_cfg.path_to_data = '/kaggle/input/tinystories'
data_cfg.path_to_data = 'data/tinystories'
data_cfg.tokenized_train_data_path = os.path.join(data_cfg.path_to_data, 'en_part_1.pickle')
# data_cfg.tokenized_train_data_path = os.path.join(data_cfg.path_to_data, 'train_v2_en_tokenized_part_1.pickle')
# data_cfg.tokenized_train_data_path = os.path.join(data_cfg.path_to_data, 'train_v2_en_tokenized_part_2.pickle')
# data_cfg.tokenized_train_data_path = os.path.join(data_cfg.path_to_data, 'train_v2_en_tokenized_part_3.pickle')
# data_cfg.tokenized_train_data_path = os.path.join(data_cfg.path_to_data, 'train_v2_en_tokenized_part_4.pickle')
data_cfg.tokenized_valid_data_path = os.path.join(data_cfg.path_to_data, 'valid_v2_en_tokenized.pickle')
data_cfg.vocabulary_size = 25_000
data_cfg.special_tokens = ["<PAD>", "<UNK>", "<ВOS>", "<EOS>"]
data_cfg.start_of_word = '▁'
data_cfg.tokenizer_path = os.path.join(data_cfg.path_to_data, 'en_tokenizer.model')
