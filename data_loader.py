import numpy as np
import pandas as pd
import gensim as gensim
import nltk
import json
import os
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


#################
# Preprocessing #
#################

import re
import nltk
import string
#printable = set(string.printable)

def clean_sent(sentence):
    # convert to lower case
    sentence = str(sentence).lower()
    # refit dashes (single words)
    sentence = re.sub(' - ','-',sentence)
    # clean punctuation
    for p in '/+-^*÷#!"(),.:;<=>?@[\]_`{|}~\'¿€$%&£±':
        sentence = sentence.replace(p,' '+p+' ') # good/bad to good / bad
    # strip leading and trailing white space
    sentence = sentence.strip()
    # nltk tokenizer
    tokenized_sentence = nltk.tokenize.word_tokenize(sentence)
    # remove non ascii characters
    #tokenized_sentence = [w for w in tokenized_sentence if w in printable]
    return tokenized_sentence



class Data_loader(object):

    def __init__(self):

        # load data
        df = pd.read_csv("data/quora_duplicate_questions.tsv",delimiter='\t')
        df = df[['question1','question2','is_duplicate']]

        # duplicate questions
        df_true_duplicate = df[df['is_duplicate']==1]
        self.Xs = df_true_duplicate['question1'].values  # Xs[k] and Ys[k] are duplicates
        self.Ys = df_true_duplicate['question2'].values
        print(self.Xs[0])
        print(self.Ys[0])
        print(len(self.Xs))

        # NOT duplicate questions
        df_false_duplicate = df[df['is_duplicate']==0]
        #df_false_duplicate = df_false_duplicate[:len(df_true_duplicate)] # balance dataset
        self.Xa = df_false_duplicate['question1'].values  # Xa[k] and Ya[k] are NOT duplicates
        self.Ya = df_false_duplicate['question2'].values


    def load_w2v(self, w2v_size):
        saving_path = 'w2v/' + 'embeddings'+str(w2v_size)+'.p'
        vocab_path = 'w2v/' + 'vocab'+str(w2v_size)+'.p'

        if not os.path.exists(saving_path):

            print("Creating word_embeddings")

            # corpus
            corpus = np.concatenate([self.Xa.reshape(-1,1),self.Xs.reshape(-1,1),self.Ya.reshape(-1,1),self.Ys.reshape(-1,1)],0).reshape(-1,1)
            #corpus = [nltk.tokenize.word_tokenize(str(sent[0]).lower()) for sent in corpus]
            corpus = [clean_sent(sent[0]) for sent in corpus]

            # initialize W2V model
            my_model = gensim.models.word2vec.Word2Vec(size=w2v_size, min_count=2, sg=1) # build model  # 52 643 not unique words
            my_model.build_vocab(corpus)

            # update with GoogleNews or Glove
            #my_model.intersect_word2vec_format(my_path + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
            my_model.intersect_word2vec_format('w2v/' + 'glove.6B.'+str(w2v_size)+'d.txt',binary=False)     # 44 373 retrieved (84%)

            # fine tune on quora corpus
            #my_model.train(corpus, total_examples=my_model.corpus_count, epochs=my_model.iter)

            # word embeddings
            weights = my_model.wv.syn0
            np.save(open(saving_path, 'wb'), weights)

            # word mapping (dictionary)
            vocab = dict([(k, v.index) for k, v in my_model.wv.vocab.items()])
            with open(vocab_path, 'w') as f:
                f.write(json.dumps(vocab))

        else:
            print("Loading from saved word_embeddings")
            with open(saving_path, 'rb') as f:
                weights = np.load(f)

        print("Loading vocab")
        with open(vocab_path, 'r') as f:
            data = json.loads(f.read())
        word2idx = data
        idx2word = dict([(v, k) for k, v in data.items()])

        print('Embedding Matrix', weights.shape)
        print('Vocabulary size',len(word2idx))
        print(idx2word[0])

        return weights, word2idx, idx2word


    def corpus2ids(self, w2v_size):

        if not os.path.exists('data/Xs_idx'+str(w2v_size)+'.npy'):
            weights, word2idx, idx2word = self.load_w2v(w2v_size)
            print('Creating w2v index based representation')

            def sent2ids(sent):
                sent_ids = []
                for w in clean_sent(sent): #nltk.tokenize.word_tokenize(str(sent).lower()):
                    try:
                        sent_ids.append(word2idx[w])
                    except:
                        for _ in range(3):
                            sent_ids.append(word2idx['*']) # UNKNOWN WORD #################################
                return np.asarray(sent_ids)

            Xs_ids, Ys_ids = {}, {}
            for i,(q1,q2) in enumerate(zip(self.Xs, self.Ys)):
                Xs_ids[i] = sent2ids(q1)
                Ys_ids[i] = sent2ids(q2)

            Xa_ids, Ya_ids = {}, {}
            for i,(q1,q2) in enumerate(zip(self.Xa, self.Ya)):
                Xa_ids[i] = sent2ids(q1)
                Ya_ids[i] = sent2ids(q2)

            # save
            np.save('data/Xs_idx'+str(w2v_size)+'.npy',Xs_ids)
            np.save('data/Ys_idx'+str(w2v_size)+'.npy',Ys_ids)
            np.save('data/Xa_idx'+str(w2v_size)+'.npy',Xa_ids)
            np.save('data/Ya_idx'+str(w2v_size)+'.npy',Ya_ids)

        Xs_ids = np.load('data/Xs_idx'+str(w2v_size)+'.npy').item() # A question = a list of word_index
        Ys_ids = np.load('data/Ys_idx'+str(w2v_size)+'.npy').item()
        Xa_ids = np.load('data/Xa_idx'+str(w2v_size)+'.npy').item()
        Ya_ids = np.load('data/Ya_idx'+str(w2v_size)+'.npy').item()

        return Xs_ids, Ys_ids, Xa_ids, Ya_ids


    def create_batches(self, data_size, batch_size=64, shuffle=True):
      """create index by batches."""
      batches = []

      ids = np.arange(data_size)
      if shuffle:
        np.random.shuffle(np.asarray(ids))
      for i in range(np.floor(data_size / batch_size).astype(int)):
        start = i * batch_size
        end = (i + 1) * batch_size
        batches.append(ids[start:end])

      return batches


    def fetch_data_ids(self, inputs, outputs, idx_batch, padlen=20): # works for ids --> ids

      def pad_sequence(ids, pad_tok=0):
        return ids[:padlen] + [pad_tok] * max(padlen - len(ids),0), len(ids)

      batch_size = len(idx_batch)

      input_batch = np.zeros([batch_size, padlen])
      input_length = np.zeros([batch_size])
      output_batch = np.zeros([batch_size, padlen])
      output_length = np.zeros([batch_size])

      for i, idx in enumerate(idx_batch):
        input_ = inputs[idx] # sentence input in BOW/TFIDF format [(word_id, freq)], in LSI format [(topic_id, value)]
        output_ = outputs[idx]

        input_batch[i], input_length[i] = pad_sequence(list(input_)) # pad
        output_batch[i], output_length[i] = pad_sequence(list(output_))

      return input_batch, input_length, output_batch, output_length





if __name__ == '__main__':

    # Init data_loader
    data_loader = Data_loader()

    # W2v
    weights, word2idx, idx2word = data_loader.load_w2v()
    Xs_ids, Ys_ids, Xa_ids, Ya_ids = data_loader.corpus2ids()

    #k = 75
    #print(Xs_ids[k])
    #print([idx2word[i] for i in Xs_ids[k]])
    #print(Ys_ids[k])
    #print([idx2word[i] for i in Ys_ids[k]])

    # Random batch indices
    batches = data_loader.create_batches(len(Xs_ids), batch_size=64)
    input_batch, input_length, output_batch, output_length = data_loader.fetch_data_ids(Xs_ids, Ys_ids, batches[0], padlen=20)
    print(input_batch.shape)
    print(input_batch[0])
    print([idx2word[i] for i in input_batch[0]])
    print([idx2word[i] for i in output_batch[0]])

