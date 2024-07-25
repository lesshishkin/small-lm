import torch
from models.tinyllm import TinyLLM
from utils.common_functions import set_seed
from utils.data_utils import get_sequence_mask
from utils.enums import InferenceType
from torch.nn.functional import softmax
import youtokentome as yttm


class Inferencer:
    """A class for model inferencing."""
    # TODO Доделать

    def __init__(self, config, init_logger=True):
        self.config = config
        set_seed(self.config.seed)
        self.tokenizer = yttm.BPE(model=self.config.data.tokenizer_path, n_threads=-1)
        self._prepare_model()
        print('Model ready')

    def _prepare_model(self):
        """Preparing model, optimizer and loss function."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = TinyLLM(self.config.model,
                             vocab_size=self.tokenizer.vocab_size(),
                             device=self.device).to(self.device)

        self.load(self.config.inference.model_path)

    def load(self, filepath: str):
        """Loads trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    # @torch.no_grad()
    # def predict(self, model_path: str, dataloader: DataLoader, inference_config):
    #     """Gets model predictions for a given dataloader."""
    #     # TODO переделать предикт
    #     self.load(model_path)
    #     self.model.eval()
    #
    #     target_lang_preprocessor = self.train_dataset.preprocessors[self.config.data.target_lang]
    #     all_predictions, all_sample_ids = [], []
    #
    #     for sample in dataloader:
    #         sample_id, encoder_inputs, _, _, _, _ = sample
    #         encoder_inputs = encoder_inputs.to(self.device)
    #         prediction = self.inference(encoder_inputs, inference_config)
    #
    #         all_predictions.extend(target_lang_preprocessor.decode(prediction, batch=True))
    #         all_sample_ids.extend(sample_id.view(-1).cpu().tolist())
    #
    #     return all_predictions, all_sample_ids

    @torch.no_grad()
    def predict(self, sentence):
        # пока по одному:
        tokenized_seq = torch.tensor(self.tokenizer.encode(sentence, bos=True)).unsqueeze(0)
        predictions = self.inference(tokenized_seq, inference_config=self.config.inference)

        return predictions

    @torch.no_grad()
    def inference(self, sequence: torch.Tensor, inference_config):
        """Makes inference with auto-regressive decoding for the given sequence."""
        # TODO переделать инференс
        self.model.eval()
        # пока будем инференсить по одному
        # инициализируем список результатов
        # делаем цикл, пока не получим еос токен либо пока не достигнем максимума итераций:
        #   даем на вход модели последовательность токенов
        #   получаем предсказания
        #   берем аргмакс или применяем другой тип инференса от последнего токена
        #   добавляем предсказанный токен к списку предсказанных токенов
        #   добавляем предсказанный токен к списку для входа в модель
        # принт очередного токена
        # декодируем последовательность и возвращаем
        batch_size = sequence.size(0)
        sos_token_id = self.config.data.special_tokens.index("<BOS>")
        eos_token_id = self.config.data.special_tokens.index("<EOS>")
        inference_step = 0
        start_pos = 0
        decoded_sequence = torch.ones((batch_size, 1), dtype=torch.int32, device=self.device) * sos_token_id
        finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        mask = get_sequence_mask(sequence, mask_future_positions=True, device=self.device)

        while not finished_sequences.all() and inference_step < inference_config.stop_predict:
            output = self.model(sequence, start_pos, mask)

            if inference_config.type == InferenceType.greedy.value:
                current_token = torch.argmax(output, dim=-1)[:, inference_step].view(-1, 1) + 1
            elif inference_config.type == InferenceType.temperature.value:
                output = output / (inference_config.temperature_value + inference_config.eps)
                probabilities = softmax(output, dim=-1)
                current_token = probabilities[:, inference_step, :].multinomial(num_samples=1) + 1
            else:
                raise Exception('Unknown inference type!')

            decoded_sequence = torch.hstack([decoded_sequence, current_token])
            finished_sequences |= current_token.squeeze() == eos_token_id
            inference_step += 1

        eos_subsequence_mask = torch.cummax(decoded_sequence == eos_token_id, dim=1).values
        decoded_sequence = decoded_sequence.masked_fill(eos_subsequence_mask, eos_token_id)

        return decoded_sequence.cpu().tolist()
