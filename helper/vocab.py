from .preprocessing import process_answer, process_sentence
from bnlp import BasicTokenizer
from collections import defaultdict
import numpy as np


class Vocabulary:
    def __init__(self, maxlen = 3, min_frequency = 3, padding = "post"):
        self._maxlen = maxlen
        self._min_frequency = min_frequency
        self._tokenizer = BasicTokenizer()
        self._vocabulary = defaultdict()
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
                if word not in self._vocabulary.keys():
                    self._vocabulary[word] = word_index
                    self._unique_words.append(word)
                    word_index += 1

                if word not in self._word_counts.keys():
                    self._word_counts[word] = 1
                else: self._word_counts[word] += 1


    def _transform_to_sequence(self, text):
        sequence = list()
        for word in self._tokenize(text):
            sequence.append(self._vocabulary[word])

        return sequence

    def build(self, text_data):
        self._text_data = text_data
        self._build_vocab()
        self._sequences = list()

        for data in self._text_data:
            padded = np.zeros(self._maxlen)
            seq = self._transform_to_sequence(data)

            if self._padding == "pre":
                if len(seq) >= self._maxlen: seq = seq[:self._maxlen]
                for i in range(0, len(seq)):
                    padded[i] = seq[i]

            else:
                if len(seq) >= self._maxlen: seq = seq[len(seq) - self._maxlen: ]
                for i in range(-1, -len(seq)-1, -1):
                    padded[i] = seq[i]

            self._sequences.append(padded)

        return self._sequences


if __name__ == "__main__":
    test = ["I eat rice", "I like to eat rice", "Play football, eat rice" ]

    vocab = Vocabulary()
    seq = vocab.build(test)
    print(seq)