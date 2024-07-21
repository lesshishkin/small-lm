from executors.trainer import Trainer
from configs.experiment_config import experiment_cfg
from utils.tokenizer_tools import create_tokenizer, create_tokenized_dataset_file, count_tokens_in_tokenized_dataset, \
    find_longest_seq, create_vocab_file
import youtokentome as yttm


# def train():
#     trainer = Trainer(experiment_cfg)
#     trainer.fit()


if __name__ == '__main__':
    # trainer = Trainer(experiment_cfg)
    # trainer.batch_overfit()
    dataset = "data/tinystories/valid_v2_en.txt"
    bpe_model = "data/tinystories/en_tokenizer.model"
    tokenized_dataset = "data/tinystories/valid_v2_en_tokenized.pickle"
    vocab_file = "data/tinystories/en_vocab.txt"
    vocab_size = 25_000

    create_vocab_file(bpe_model, vocab_file)