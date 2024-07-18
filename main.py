import youtokentome as yttm

def create_tokenizer():
    train_data_path = "train_data.txt"
    model_path = "example.model"

    yttm.BPE.train(data=train_data_path, vocab_size=25000, model=model_path)


if __name__ == '__main__':
    create_tokenizer()

