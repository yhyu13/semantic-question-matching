import numpy as np
import pandas as pd
import pickle
import os

from nltk import word_tokenize
from nltk.stem import SnowballStemmer

import tensorflow as tf

from embeddings import word_index, reverse_dict


def check_str(s):
    if type(s) == str:
        return s
    else:
        return "NaN"


def lower_list(l):
    return ([elt.lower() for elt in l])


def tokenize_dict(q_dict, lower=True, stemmer=None):
    if stemmer == "english":
        snowball = SnowballStemmer("english")
        return {k: lower_list([snowball.stem(token) for token in word_tokenize(q_dict[k])]) if lower
        else [snowball.stem(token) for token in word_tokenize(q_dict[k])]
                for k in q_dict.keys()}
    else:
        return {k: lower_list(word_tokenize(q_dict[k])) if lower
        else word_tokenize(q_dict[k])
                for k in q_dict.keys()}


def sent2ids(sent, w2idx):
    return [w2idx[w] if w in w2idx.keys() else w2idx["<UNK>"] for w in sent]


def ids2sent(ids, idx2w):
    return [idx2w[i] for i in ids]


def pad_sequence(ids, padlen, pad_tok=0):
    return ids[:padlen] + [pad_tok] * max(padlen - len(ids), 0)


def sequence_dict(tok_dict, w2idx):
    seq_dict = {k: sent2ids(tok_dict[k], w2idx) for k in tok_dict.keys()}
    return seq_dict


class Data_iterator(object):
    def __init__(self, data, batch=1):
        self.q1, self.q2, self.l1, self.l2, self.y = data
        self.batch = batch
        self.i = 0
        self.max = len(self.q1)

    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.max:
            ranged = (self.i, min(self.i + self.batch, self.max))
            self.i += self.batch
            return self.q1[ranged[0]:ranged[1]], self.q2[ranged[0]:ranged[1]], self.l1[ranged[0]:ranged[1]],\
                   self.l2[ranged[0]:ranged[1]], self.y[ranged[0]:ranged[1]]
            raise StopIteration()


class QuoraDataset(object):
    def __init__(self, filename, sep=',', w2idx=None, padlen=40, save_path=None):
        self.filename = filename
        self.padlen = padlen
        self.w2idx = w2idx
        self.sep = sep
        self.save_path = save_path

        if self.save_path is not None and os.path.exists(self.save_path):
            self.reload()
        else:
            self.build()
            if self.save_path is not None:
                self.save()

    def save(self):
        with open(self.save_path, "wb") as file:
            pickle.dump(self.__dict__, file)

    def reload(self):
        with open(self.save_path, "rb") as file:
            tmp_dict = pickle.load(file)

        self.__dict__.update(tmp_dict)

    def build(self):
        self.df = pd.read_csv(self.filename, sep=self.sep)

        print("Building question dictionary...")
        self.q_dict = {}

        for _, row in self.df.iterrows():
            i, qid1 = row["id"], row["qid1"]
            if qid1 not in self.q_dict:
                self.q_dict[qid1] = check_str(self.df["question1"][self.df["id"] == i])

        for _, row in self.df.iterrows():
            i, qid2 = row["id"], row["qid2"]
            if qid2 not in self.q_dict:
                self.q_dict[qid2] = check_str(self.df["question2"][self.df["id"] == i])

        print("Tokenizing questions...")
        self.tok_dict = tokenize_dict(self.q_dict)

        if self.w2idx is None:
            self.w2idx, self.idx2w = word_index(self.tok_dict.values())
        else:
            self.idx2w = reverse_dict(self.w2idx)

        self.seq_dict = sequence_dict(self.tok_dict, self.w2idx)

        dsize = len(self.df["qid1"])
        self.q1 = [pad_sequence(self.seq_dict[np.array(self.df["qid1"])[k]], self.padlen) for k in range(dsize)]
        self.l1 = [min(len(self.seq_dict[np.array(self.df["qid1"])[k]]), self.padlen) for k in range(dsize)]
        self.q2 = [pad_sequence(self.seq_dict[np.array(self.df["qid2"])[k]], self.padlen) for k in range(dsize)]
        self.l2 = [min(len(self.seq_dict[np.array(self.df["qid2"])[k]]), self.padlen) for k in range(dsize)]
        self.y = np.array(self.df["is_duplicate"])

        print("Done")

    def data(self, len=None):
        if len is not None:
            return self.q1[:len], self.q2[:len], self.l1[:len], self.l2[:len], self.y[:len]
        else:
            return self.q1, self.q2, self.l1, self.l2, self.y
