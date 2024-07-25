from executors.trainer import Trainer
from executors.inferencer import Inferencer
from configs.experiment_config import experiment_cfg
from utils.tokenizer_tools import create_tokenizer, create_tokenized_dataset_file, count_tokens_in_tokenized_dataset, \
    find_longest_seq, create_vocab_file, divide_dataset
import youtokentome as yttm


def run_inference():
    inferencer = Inferencer(experiment_cfg)
    print('Prompt:')
    sentence = input()
    inferencer.predict(sentence)


if __name__ == '__main__':
    run_inference()
