from abc import abstractmethod
import math
import torch
import torch.nn.functional as F
from transformers import BertForPreTraining, BertTokenizer


class ModelMixin:
    def __init__(self, stop_words, sentence_length=50, *args, **kwargs):
        self.stop_words = stop_words or {".", "?", "!", "。", "？", "！"}
        self.stop_words_outer = self.stop_words | {",", "，", ";", "；"}
        self.sentence_length = sentence_length  # 长句切割幅度， 防止bert模型太慢了

    @staticmethod
    @abstractmethod
    def from_pretrained(path, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def save(self, path, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def train(self, x, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def score(self, x, temperature=1.0, verbose=False, *args, **kwargs):
        raise NotImplementedError

    def perplexity(self, x, temperature=1.0, verbose=False, *args, **kwargs):
        l_score = self.score(x=x, temperature=temperature, verbose=verbose, *args, **kwargs)
        ppl = math.pow(2, -1 * l_score)
        return ppl

    def convert_inputs_to_sentences(self, x):
        if isinstance(x, str):
            x = x.split(" ")
        last_outer_idx = 0
        split_ids = [-1]
        for i, w in enumerate(x):
            if w in self.stop_words_outer:
                last_outer_idx = i
            if i - split_ids[-1] > self.sentence_length:
                if last_outer_idx == split_ids[-1]:
                    raise ValueError(
                        f"Sentence `{''.join(x[last_outer_idx: i + 1])}` is longer than `sentence_length (curr={self.sentence_length})`, please set it larger.")
                split_ids.append(last_outer_idx)
            elif w in self.stop_words:
                split_ids.append(i)
        if split_ids[-1] != len(x) - 1:
            split_ids.append(len(x) - 1)

        sentences = list()
        for start, end in zip(split_ids[:-1], split_ids[1:]):
            sentences.append(x[start + 1: end + 1])
        return sentences


class MaskedBert(ModelMixin):
    def __init__(self, stop_words=None, sentence_length=50, device="cpu"):
        super(MaskedBert, self).__init__(stop_words=stop_words, sentence_length=sentence_length)
        self.model = None
        self.tokenizer = None
        self.mask_id = -1
        self.device = device

    @staticmethod
    def from_pretrained(path, sentence_length=50, device="cpu", stop_words=None, *args, **kwargs):
        model = BertForPreTraining.from_pretrained(path)
        tokenizer = BertTokenizer.from_pretrained(path)
        self = MaskedBert(device=device, stop_words=stop_words, sentence_length=sentence_length)
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.mask_id = int(tokenizer.convert_tokens_to_ids("[MASK]"))

        return self

    def save(self, path, *args, **kwargs):
        pass

    def train(self, x, *args, **kwargs):
        pass

    def score(self, x, temperature=1.0, batch_size=100, verbose=False, *args, **kwargs):
        self.model.eval()

        sentences = self.convert_inputs_to_sentences(x)
        all_probability = list()
        all_words = list()
        for sentence in sentences:
            inputs = self.tokenizer("".join(sentence), return_tensors="pt")
            input_ids, token_type_ids, attention_mask = inputs["input_ids"], inputs["token_type_ids"], inputs[
                "attention_mask"]
            origin_ids = input_ids[0][1: -1]
            length = input_ids.shape[-1] - 2

            batch_indice = list()
            for i in range(length // batch_size):
                batch_indice.append([i * batch_size, (i + 1) * batch_size])
            if length % batch_size != 0:
                batch_indice.append([batch_size * (length // batch_size), length])

            for start, end in batch_indice:
                ids_list = list()
                for i in range(start, end):
                    # mask one
                    tmp = input_ids.clone()
                    tmp[0][i + 1] = self.mask_id
                    ids_list.append(tmp)
                new_input_ids = torch.cat(ids_list, dim=0)
                new_attention_mask = attention_mask.expand(end - start, length + 2)
                new_token_type_ids = token_type_ids.expand(end - start, length + 2)
                inputs = {
                    'input_ids': new_input_ids.to(self.device),
                    'token_type_ids': new_token_type_ids.to(self.device),
                    'attention_mask': new_attention_mask.to(self.device)
                }
                outputs = self.model(**inputs).prediction_logits
                outputs = F.softmax(outputs / temperature, dim=-1).detach().cpu().numpy()
                probability = [outputs[i][start + i + 1][ids] for i, ids in enumerate(origin_ids[start: end])]
                all_probability += probability
                all_words += self.tokenizer.convert_ids_to_tokens(origin_ids[start: end])

        if len(all_probability) == 0:
            l_score = 0
        else:
            l_score = sum([math.log(p, 2) for p in all_probability]) / len(all_probability)

        if verbose:
            words = list()
            for s in sentences:
                words += s
            for word, prob in zip(all_words, all_probability):
                print(f"{word} | {prob:.8f}")
            print(f"l score: {l_score:.8f}")

        return l_score


def calc_fluency(sentences):
    '''
    Count fluency index by bert.
    Mask one word and guess it by bert.
    '''
    print('Calculating fluency')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MaskedBert.from_pretrained(path="bert-base-cased", device=device, sentence_length=50)
    ppl = model.perplexity(x=sentences, verbose=False, temperature=1.0, batch_size=16)
    return ppl
