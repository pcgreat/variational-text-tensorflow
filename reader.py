import itertools
import os
from collections import Counter

import nltk
from nltk.tokenize import TreebankWordTokenizer

from utils import *

EOS_TOKEN = "_eos_"


class TextReader(object):
    def __init__(self, data_path):
        train_path = os.path.join(data_path, "train.txt")
        valid_path = os.path.join(data_path, "valid.txt")
        test_path = os.path.join(data_path, "test.txt")
        vocab_path = os.path.join(data_path, "vocab.pkl")

        self.tokenizer = TreebankWordTokenizer()

        if os.path.exists(vocab_path):
            self._load(vocab_path, train_path, valid_path, test_path)
        else:
            self._build_vocab(train_path, vocab_path)
            self.train_data = self._file_to_data(train_path)
            self.valid_data = self._file_to_data(valid_path)
            self.test_data = self._file_to_data(test_path)

        self.idx2word = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)

    def _read_text_to_toks(self, file_path):
        l = []
        with open(file_path) as f:
            for line in f:
                toks = self.tokenize(line)
                l.extend(toks)
        return l

    def _build_vocab(self, file_path, vocab_path):
        counter = Counter(self._read_text_to_toks(file_path))

        count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
        self.vocab = dict(zip(words, range(len(words))))

        save_pkl(vocab_path, self.vocab)

    def _file_to_data(self, file_path):
        data = []
        with open(file_path) as f:
            for text in f:
                data.append(np.array(list(map(self.vocab.get,
                                              nltk.tokenize.word_tokenize(text.strip().lower())))))

        save_npy(file_path + ".npy", data)
        return data

    def _load(self, vocab_path, train_path, valid_path, test_path):
        self.vocab = load_pkl(vocab_path)

        self.train_data = load_npy(train_path + ".npy")
        self.valid_data = load_npy(valid_path + ".npy")
        self.test_data = load_npy(test_path + ".npy")

    def get_data_from_type(self, data_type):
        if data_type == "train":
            raw_data = self.train_data
        elif data_type == "valid":
            raw_data = self.valid_data
        elif data_type == "test":
            raw_data = self.test_data
        else:
            raise Exception(" [!] Unkown data type %s: %s" % data_type)

        return raw_data

    def onehot(self, data, min_length=None):
        if min_length == None:
            min_length = self.vocab_size
        return np.bincount(data, minlength=min_length)

    def iterator(self, data_type="train"):
        raw_data = self.get_data_from_type(data_type)
        return itertools.cycle([[self.onehot(data), data] for data in raw_data if data != []])

    def tokenize(self, text):
        text = text.strip().lower()
        toks = self.tokenizer.tokenize(text)
        return toks

    def get(self, text):
        toks = self.tokenize(text)
        try:
            data = np.array(list(map(self.vocab.get, toks)))
            return self.onehot(data), data
        except:
            unknowns = []
            for word in text:
                if self.vocab.get(word) == None:
                    unknowns.append(word)
            raise Exception(" [!] unknown words: %s" % ",".join(unknowns))

    def random(self, data_type="train"):
        raw_data = self.get_data_from_type(data_type)
        idx = np.random.randint(len(raw_data))

        data = raw_data[idx]
        return self.onehot(data), data
