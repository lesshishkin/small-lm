from easydict import EasyDict

data_cfg = EasyDict()

data_cfg.translation_dataset = EasyDict()
data_cfg.translation_dataset.name = 'RussianStoriesDataset'
data_cfg.translation_dataset.path_to_data = 'data/tinystories'
data_cfg.translation_dataset.lang = 'ru'
data_cfg.translation_dataset.vocabulary_size = 25_000
data_cfg.translation_dataset.sort = True

data_cfg.translation_dataset.preprocessing = EasyDict()
data_cfg.translation_dataset.preprocessing.raw_data_path_template = 'raw_data_%s_%s.txt'  # set type
data_cfg.translation_dataset.preprocessing.tokenizer_path = 'tokenizer.pickle'
data_cfg.translation_dataset.preprocessing.preprocessed_data_path_template = 'tokenized_data_%.pickle'  # set type
data_cfg.translation_dataset.preprocessing.special_tokens = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]
data_cfg.translation_dataset.preprocessing.end_of_word = '</w>'
data_cfg.translation_dataset.preprocessing.min_frequency = 2
