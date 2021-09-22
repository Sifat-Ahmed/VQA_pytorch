from .preprocessing import process_answer, process_sentence, load_glove_model
from bnlp import BasicTokenizer
from collections import defaultdict
from tqdm import tqdm


class Vocabulary:
    def __init__(self, maxlen = 30, min_frequency = 3, padding = "post"):
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
            if self._vector:
                try:
                    sequence.append(list(self._vector[word]))
                except:
                    pass
                finally:
                    sequence.append(list([0.0] * 300))
            else:
                sequence.append(self._vocabulary[word])

        return sequence


    def transform(self):
        pass

    def get_sequences(self, text_data, vector = None):
        self._text_data = text_data
        self._build_vocab()
        self._sequences = list()
        self._vector = vector


        for data in tqdm(self._text_data, leave=True, position=0, colour = "blue", desc="Converting text to sequence"):
            padded = [0.0] * self._maxlen
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
    test = ["পিজ্জার টুকরোতে কি ", "কি কোনো ব্রোকলি", "কোনো ব্রোকলি আছে" ]

    vocab = Vocabulary()
    vector = load_glove_model("/home/workstaion/workspace/potatochips/vqa/bn_glove.39M.300d.txt")

    seq = vocab.get_sequences(test,vector)

    for s in seq:
        print(s)
        print()