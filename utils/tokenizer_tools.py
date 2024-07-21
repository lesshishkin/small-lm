import youtokentome as yttm


def find_longest_seq(dataset_path):
    import pickle
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    print(len(max(data, key=len)))
    # RU longest: 1071
    # EN longest: 1189


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
    # RU
    # Num tokens: 382_625_157
    # Num texts:  2_717_495


def create_tokenizer(input_path, output_path, vocab_size):
    train_data_path = input_path
    model_path = output_path

    yttm.BPE.train(data=train_data_path, vocab_size=vocab_size, model=model_path)


def create_tokenized_dataset_file(input_path, output_path, tokenizer_model_path):
    import pickle

    with open(input_path, 'r', encoding='utf-8') as file:
        data = file.read()

    data = data.split('<|endoftext|>')
    bpe = yttm.BPE(model=tokenizer_model_path)
    final_data = bpe.encode(data, bos=True, eos=True)

    with open(output_path, 'wb') as pickle_file:
        pickle.dump(final_data, pickle_file)


def create_vocab_file(model_path, output_path):
    bpe = yttm.BPE(model=model_path)
    vocab = bpe.vocab()

    with open(output_path, 'w', encoding='utf-8') as file:
        for item in vocab:
            file.write(f'{item}\n')


def divide_dataset():
    import pickle

    with open('data/tinystories/train_v2_en_tokenized.pickle', 'rb') as file:
        data = pickle.load(file)

    # Проверка, что данные действительно являются списком списков
    if not all(isinstance(i, list) for i in data):
        raise ValueError("Содержимое файла должно быть списком списков")

    # Определение размера каждой части
    part_size = len(data) // 4

    # Разделение данных на 4 части
    parts = [data[i * part_size: (i + 1) * part_size] for i in range(4)]

    # Обработка оставшихся элементов, если их количество не делится на 4
    if len(data) % 4 != 0:
        parts[-1].extend(data[4 * part_size:])

    # Сохранение каждой части в отдельный pickle файл
    for i, part in enumerate(parts):
        with open(f'train_v2_en_tokenized_part_{i + 1}.pickle', 'wb') as file:
            pickle.dump(part, file)

    print(
        "Данные успешно разделены и сохранены в файлы output_part_1.pkl, output_part_2.pkl, output_part_3.pkl, output_part_4.pkl")
