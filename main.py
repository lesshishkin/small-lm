import youtokentome as yttm
# from executors.trainer import Trainer
# from configs.experiment_config import experiment_cfg


# def train():
#     trainer = Trainer(experiment_cfg)
#     trainer.fit()


def create_tokenizer():
    train_data_path = "data/tinystories/train_v2_ru_prepared.txt"
    model_path = "data/tinystories/ru_tinystories_tokenizer.model"

    yttm.BPE.train(data=train_data_path, vocab_size=25000, model=model_path)


def create_tokenized_dataset_file(input_path, output_path, tokenizer_model_path):
    import pickle

    with open(input_path, 'r', encoding='utf-8') as file:
        data = file.read()

    data = data.split('<|endoftext|>')
    bpe = yttm.BPE(model=tokenizer_model_path)
    final_data = bpe.encode(data, bos=True, eos=True)

    with open(output_path, 'wb') as pickle_file:
        pickle.dump(final_data, pickle_file)


def load_tokenizer(model_path):
    # Loading model
    bpe = yttm.BPE(model=model_path)
    #
    # test_text = 'Пришла жаба в гости к хрбше, а Том сидел в углу и думал'
    # # Two types of tokenization
    # print(bpe.encode([test_text], output_type=yttm.OutputType.ID))
    # print(bpe.encode([test_text], output_type=yttm.OutputType.SUBWORD))

    # Получение словаря
    vocab = bpe.vocab()

    # Сохранение словаря в файл
    with open('vocab.txt', 'w') as f:
        for token in vocab:
            f.write(f"{token}\n")


if __name__ == '__main__':
    create_tokenized_dataset_file(input_path="data/tinystories/train_v2_ru.txt",
                                  output_path="data/tinystories/train_v2_ru_tokenized.pickle",
                                  tokenizer_model_path="data/tinystories/ru_tinystories_tokenizer.model")
