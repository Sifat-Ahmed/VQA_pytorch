from .preprocessing import process_answer, process_sentence, load_glove_model
from bnlp import BasicTokenizer
from collections import defaultdict
from tqdm import tqdm
import numpy as np

class Vocabulary:
    def __init__(self, maxlen = 10, min_frequency = 3, padding = "post"):
        self._maxlen = maxlen
        self._min_frequency = min_frequency
        self._tokenizer = BasicTokenizer()
        self._vocabulary = defaultdict()
        self._words = list()
        self._unique_words = list()
        self._word_counts = defaultdict()
        self._padding = padding
        assert self._padding == "pre" or self._padding == "post", "Padding mode error (pre/post)"


    def _tokenize(self, text):
        text = process_sentence(text)
        return self._tokenizer.tokenize(text)

    def _build_vocab(self):
        word_index = 1
        for data in self._text_data:
            tokenized = self._tokenize(data)

            for word in tokenized:
                self._words.append(word)
                if word not in self._vocabulary.keys():
                    self._vocabulary[word] = word_index

                    word_index += 1

                if word not in self._word_counts.keys():
                    self._word_counts[word] = 1
                else: self._word_counts[word] += 1

        self._unique_words = set(self._words)

    def vocab_size(self):
        return len(self._unique_words) + 1

    def get_word_counts(self):
        return np.unique(self._words, return_counts=True)


    def _transform_to_sequence(self, text):
        sequence = list()
        for word in self._tokenize(text):
            if self._vector:
                try:
                    sequence.append(np.array(self._vector[word]))
                except:
                    pass
                finally:
                    sequence.append(np.array([0.0] * 300))
            else:
                sequence.append(self._vocabulary[word])


        return np.array(sequence), len(sequence)

    def transform(self):
        pass

    def get_sequences(self, text_data, vector = None, return_length = False):
        self._text_data = text_data
        self._build_vocab()
        self._sequences = list()
        self._lengths = list()
        self._vector = vector


        for data in tqdm(self._text_data, leave=True, position=0, colour = "blue", desc="Converting text to sequence"):
            padded = np.array([[0.0] * 300] * self._maxlen) if self._vector else np.array([0.0] * self._maxlen)

            seq, length = self._transform_to_sequence(data)

            if self._padding == "pre":
                if len(seq) >= self._maxlen: seq = seq[:self._maxlen]
                for i in range(0, len(seq)):
                    padded[i] = seq[i]

            else:
                if len(seq) >= self._maxlen: seq = seq[len(seq) - self._maxlen: ]
                for i in range(-1, -len(seq)-1, -1):
                    padded[i] = seq[i]

            self._sequences.append(np.array(padded))

        return np.array(self._sequences)


if __name__ == "__main__":
    test = ["পিজ্জার টুকরোতে কি ", "কি কোনো ব্রোকলি", "কোনো ব্রোকলি আছে" ]

    vocab = Vocabulary()
    #vector = load_glove_model("/home/workstaion/workspace/potatochips/vqa/bn_glove.39M.300d.txt")

    seq = vocab.get_sequences(test)

    for s in seq:
        print(s)

