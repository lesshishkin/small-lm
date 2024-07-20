import youtokentome as yttm


def find_longest_seq(dataset_path):
    import pickle
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    print(len(max(data, key=len)))
    # longest: 1071

def count_tokens_in_tokenized_dataset(dataset_path):
    import pickle
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    num_tokens = 0
    num_texts = 0
    for line in data:
        num_tokens += len(line)
        num_texts += 1

    print('Num tokens:', num_tokens)
    print('Num texts: ', num_texts)
    # Num tokens: 382_625_157
    # Num texts:  2_717_495

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