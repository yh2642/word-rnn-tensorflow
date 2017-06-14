# -*- coding: utf-8 -*-
import os
import codecs
import collections
from six.moves import cPickle
import numpy as np
import re
import ujson
import itertools
import gzip

class TextLoader():
    def __init__(self, data_dir, batch_size, seq_length, encoding=None):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        # Let's not read voca and data from file. We many change them.
        if True or not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(data_dir, vocab_file, tensor_file, encoding, seq_length)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.batch_generator = self.generate_batches()
        self.reset_batch_pointer()


    def clean_str(self, string):
        """
        Tokenization/string cleaning for all datasets except for SST.
        Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data
        """
        string = re.sub(r"[^가-힣A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def build_vocab(self, sentences, vocab):
        """
        Builds a vocabulary mapping from word to index based on the sentences.
        Returns vocabulary mapping and inverse vocabulary mapping.
        """
        # Build vocabulary
        # word_counts = collections.Counter(reduce(lambda x, y: x+y, [sent.keys() for sent in sentences]))
        # Mapping from index to word
        # vocabulary_inv = [x[0] for x in word_counts.most_common()]
        vocabulary_inv = list(sorted(list(vocab)))
        # Mapping from word to index
        vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
        return [vocabulary, vocabulary_inv]

    @staticmethod
    def sample_generator(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith(".gz"):
                for line in gzip.open(data_dir + filename):
                    yield line

    def preprocess(self, data_dir, vocab_file, tensor_file, encoding, seq_length):
        x_text = 0
        vocab = set()
        for line in self.sample_generator(data_dir):
            profile_serials = ujson.loads(line.strip())
            if len(profile_serials) < seq_length:
                continue
            mul_cut = len(profile_serials) / seq_length
            profile_serials = profile_serials[:mul_cut*seq_length]
            for profile in profile_serials:
                for word in profile.keys():
                    vocab.add(word)
            x_text += len(profile_serials)
            if x_text > 100000:
                break
        self.vocab, self.words = self.build_vocab(x_text, vocab)
        self.vocab_size = len(self.words)

        with open(vocab_file, 'wb') as f:
            cPickle.dump(self.words, f)
        #The same operation like this [self.vocab[word] for word in x_text]
        # index of words as our basic data
        self.tensor = x_text
        # Save the data to data.npy
        # np.save(tensor_file, self.tensor)


    #
    # def preprocess(self, input_file, vocab_file, tensor_file, encoding):
    #     with codecs.open(input_file, "r", encoding=encoding) as f:
    #         data = f.read()
    #
    #     # Optional text cleaning or make them lower case, etc.
    #     #data = self.clean_str(data)
    #     x_text = data.split()
    #
    #     self.vocab, self.words = self.build_vocab(x_text)
    #     self.vocab_size = len(self.words)
    #
    #     with open(vocab_file, 'wb') as f:
    #         cPickle.dump(self.words, f)
    #
    #     #The same operation like this [self.vocab[word] for word in x_text]
    #     # index of words as our basic data
    #     self.tensor = np.array(list(map(self.vocab.get, x_text)))
    #     # Save the data to data.npy
    #     np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.words = cPickle.load(f)
        self.vocab_size = len(self.words)
        self.vocab = dict(zip(self.words, range(len(self.words))))
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.size / (self.batch_size *
                                                   self.seq_length))

    def create_batches(self):
        self.num_batches = int(self.tensor / (self.batch_size *
                                              (self.seq_length + 1)))
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.num_batches * self.batch_size * (self.seq_length+1)
        # xdata = self.tensor
        # ydata = np.copy(self.tensor)
        #
        # ydata[:-1] = xdata[1:, ]
        # ydata[-1] = xdata[0, ]
        # self.x_batches = map(lambda x: x.reshape(self.batch_size, self.seq_length, -1), np.split(xdata, self.num_batches, 0))
        # self.y_batches = map(lambda x: x.reshape(self.batch_size, self.seq_length, -1), np.split(ydata, self.num_batches, 0))
    #
    # def create_batches(self):
    #     self.num_batches = int(self.tensor.shape[0] / (self.batch_size *
    #                                                self.seq_length))
    #     if self.num_batches==0:
    #         assert False, "Not enough data. Make seq_length and batch_size small."
    #
    #     self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
    #     xdata = self.tensor
    #     ydata = np.copy(self.tensor)
    #
    #     ydata[:-1] = xdata[1:, ]
    #     ydata[-1] = xdata[0, ]
    #     self.x_batches = map(lambda x: x.reshape(self.batch_size, self.seq_length, -1), np.split(xdata, self.num_batches, 0))
    #     self.y_batches = map(lambda x: x.reshape(self.batch_size, self.seq_length, -1), np.split(ydata, self.num_batches, 0))

    def generate_batches(self):
        def gen_vec(profile):
            vec = np.zeros(self.vocab_size)
            for cat_id, score in profile.items():
                idx = self.vocab[cat_id]
                vec[idx] = score
            return vec
        batch_x = []
        batch_y = []
        for line in self.sample_generator(self.data_dir):
            profile_serials = ujson.loads(line.strip())
            if len(profile_serials) != self.seq_length + 1:
                continue
            xdata = np.array(list(map(gen_vec, profile_serials)))
            ydata = np.copy(xdata)

            ydata[:-1] = xdata[1:, ]
            ydata[-1] = xdata[0, ]
            batch_x.append(xdata[:-1])
            batch_y.append(ydata[:-1])
            if len(batch_x) == self.batch_size and len(batch_y) == self.batch_size:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []

    def next_batch(self):
        # x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        x, y = self.batch_generator.next()
        x.astype('float32')
        y.astype('float32')
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0
