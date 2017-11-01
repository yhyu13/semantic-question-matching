import numpy as np
import pandas as pd
import gensim as gensim
import nltk
import json
import os
#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class Data_loader(object):

    def __init__(self):

        # parameters
        self.num_topics=200 # lsi

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

        # build dictonary
        if not os.path.exists('data/vocab.dict'):
            # corpus
            corpus = np.concatenate([self.Xa.reshape(-1,1),self.Xs.reshape(-1,1),self.Ya.reshape(-1,1),self.Ys.reshape(-1,1)],0).reshape(-1,1)
            corpus = [nltk.tokenize.word_tokenize(sent[0].lower()) for sent in corpus]
            # dictionary = mapping between the words and ids
            dictionary = gensim.corpora.Dictionary(corpus)
            # remove words that appear only once or in more than 90%
            dictionary.filter_extremes(no_below=2, no_above=0.90)
            # assign new word ids to all words.
            dictionary.compactify()
            # store the dictionary, for future reference
            dictionary.save('data/vocab.dict')
        
        # load word --> integer mapping
        self.dictionary = gensim.corpora.Dictionary.load('data/vocab.dict')
        self.vocab_size = len(self.dictionary.keys())


    def load_BOW(self):

        # build BOW corpus
        if not os.path.exists('data/corpus_BOW.mm'):
            # corpus to BOW
            corpus = np.concatenate([self.Xa.reshape(-1,1),self.Xs.reshape(-1,1),self.Ya.reshape(-1,1),self.Ys.reshape(-1,1)],0).reshape(-1,1)
            corpus_bow = [self.dictionary.doc2bow(nltk.tokenize.word_tokenize(sent[0].lower())) for sent in corpus]
            Xs_bow = [self.dictionary.doc2bow(nltk.tokenize.word_tokenize(sent.lower())) for sent in self.Xs]
            Ys_bow = [self.dictionary.doc2bow(nltk.tokenize.word_tokenize(sent.lower())) for sent in self.Ys]
            Xa_bow = [self.dictionary.doc2bow(nltk.tokenize.word_tokenize(sent.lower())) for sent in self.Xa]
            Ya_bow = [self.dictionary.doc2bow(nltk.tokenize.word_tokenize(sent.lower())) for sent in self.Ya]
            # store BOW representation to disk, for later use
            gensim.corpora.MmCorpus.serialize('data/corpus_BOW.mm', corpus_bow)
            gensim.corpora.MmCorpus.serialize('data/corpus_Xs_BOW.mm', Xs_bow)
            gensim.corpora.MmCorpus.serialize('data/corpus_Ys_BOW.mm', Ys_bow)
            gensim.corpora.MmCorpus.serialize('data/corpus_Xa_BOW.mm', Xa_bow)
            gensim.corpora.MmCorpus.serialize('data/corpus_Ya_BOW.mm', Ya_bow)

        # load doc --> BOW representation
        corpus_bow = gensim.corpora.MmCorpus('data/corpus_BOW.mm')
        Xs_bow = gensim.corpora.MmCorpus('data/corpus_Xs_BOW.mm') # A question = a list of tuples (word_id, freq)
        Ys_bow = gensim.corpora.MmCorpus('data/corpus_Ys_BOW.mm')
        Xa_bow = gensim.corpora.MmCorpus('data/corpus_Xa_BOW.mm')
        Ya_bow = gensim.corpora.MmCorpus('data/corpus_Ya_BOW.mm')
        return corpus_bow, Xs_bow, Ys_bow, Xa_bow, Ya_bow


    def load_TFIDF(self):
        
        corpus_bow, Xs_bow, Ys_bow, Xa_bow, Ya_bow = self.load_BOW()

        # convert BOW vectors (counts) to TfIdf representation (weights)
        tfidf = gensim.models.TfidfModel(corpus_bow)
        corpus_idf = tfidf[corpus_bow]
        Xs_idf = tfidf[Xs_bow] # A question = a list of tuples (word_id, value)
        Ys_idf = tfidf[Ys_bow]
        Xa_idf = tfidf[Xa_bow]
        Ya_idf = tfidf[Ya_bow]
        return corpus_idf, Xs_idf, Ys_idf, Xa_idf, Ya_idf


    def load_LSI(self):

        # build LSI corpus
        if not os.path.exists('data/Quora.lsi'):
            # convert Tfidf vectors to Latent Semantic space (num_topics in 200-500 'golden standard' Gensim) and save model
            corpus_idf, Xs_idf, Ys_idf, Xa_idf, Ya_idf = self.load_TFIDF()
            lsi = gensim.models.LsiModel(corpus_idf, id2word=self.dictionary, num_topics=self.num_topics) # LSI transformation
            lsi.save('data/Quora.lsi')

        if not os.path.exists('data/Xs_lsi.npy'):
            # load LSA semantic corpus representation
            corpus_idf, Xs_idf, Ys_idf, Xa_idf, Ya_idf = self.load_TFIDF()
            lsi = gensim.models.LsiModel.load('data/Quora.lsi')
            Xs_lsi = lsi[Xs_idf]
            Ys_lsi = lsi[Ys_idf]
            Xa_lsi = lsi[Xa_idf]
            Ya_lsi = lsi[Ya_idf]

            # save LSI doc-topics matrices
            Xs_lsi_, Ys_lsi_ = [], []
            for q1, q2 in zip(Xs_lsi,Ys_lsi):
                q1, q2 = np.asarray(q1), np.asarray(q2)
                try:
                    q1, q2 = np.transpose(q1,[1,0])[1], np.transpose(q2,[1,0])[1]
                    Xs_lsi_.append(q1)
                    Ys_lsi_.append(q2)
                except:
                    pass
            Xs_lsi_, Ys_lsi_ = np.asarray(Xs_lsi_), np.asarray(Ys_lsi_)

            Xa_lsi_, Ya_lsi_ = [], []
            for q1, q2 in zip(Xa_lsi,Ya_lsi):
                q1, q2 = np.asarray(q1), np.asarray(q2)
                try:
                    q1, q2 = np.transpose(q1,[1,0])[1], np.transpose(q2,[1,0])[1]
                    Xa_lsi_.append(q1)
                    Ya_lsi_.append(q2)
                except:
                    pass
            Xa_lsi_, Ya_lsi_ = np.asarray(Xa_lsi_), np.asarray(Ya_lsi_)

            np.save('data/Xs_lsi.npy',Xs_lsi_)
            np.save('data/Ys_lsi.npy',Ys_lsi_)
            np.save('data/Xa_lsi.npy',Xa_lsi_)
            np.save('data/Ya_lsi.npy',Ya_lsi_)

        Xs_lsi = np.load('data/Xs_lsi.npy') # A question = an array (value_i) where value_i corresponds to topic_i
        Ys_lsi = np.load('data/Ys_lsi.npy')
        Xa_lsi = np.load('data/Xa_lsi.npy')
        Ya_lsi = np.load('data/Ya_lsi.npy')
        return Xs_lsi, Ys_lsi, Xa_lsi,Ya_lsi


    def load_w2v(self):
        my_path = 'C:\\Users\\ASUS\\Documents\\Telecom\\PRIM\\code\\_deepNLU\\w2v\\'
        saving_path = my_path + "embeddings.p"
        vocab_path = my_path + "vocab.p"

        if not os.path.exists(saving_path):

            print("Creating word_embeddings")

            # corpus
            corpus = np.concatenate([self.Xa.reshape(-1,1),self.Xs.reshape(-1,1),self.Ya.reshape(-1,1),self.Ys.reshape(-1,1)],0).reshape(-1,1)
            corpus = [nltk.tokenize.word_tokenize(str(sent[0]).lower()) for sent in corpus]

            # initialize W2V model
            my_model = gensim.models.word2vec.Word2Vec(size=300, min_count=1, sg=1) # build model
            my_model.build_vocab(corpus)

            # update with GoogleNews or Glove
            #my_model.intersect_word2vec_format(my_path + 'GoogleNews-vectors-negative300.bin.gz', binary=True)
            my_model.intersect_word2vec_format(my_path + "glove.6B.300d.txt",binary=False)

            # fine tune on quora corpus
            my_model.train(corpus, total_examples=my_model.corpus_count, epochs=my_model.iter)

            # trim memory
            #my_model.init_sims(replace=True)
            #word_vectors = my_model.wv
            #del my_model

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


    def corpus2ids(self):
        weights, word2idx, idx2word = self.load_w2v()

        if not os.path.exists('data/Xs_idx.npy'):

            print('Creating w2v index based representation')

            def sent2ids(sent):
                sent_ids = []
                for w in nltk.tokenize.word_tokenize(str(sent).lower()):
                    sent_ids.append(word2idx[w])
                    #except: sent_ids.append(word2idx['?']) # UNKNOWN WORD #################################
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
            np.save('data/Xs_idx.npy',Xs_ids)
            np.save('data/Ys_idx.npy',Ys_ids)
            np.save('data/Xa_idx.npy',Xa_ids)
            np.save('data/Ya_idx.npy',Ya_ids)


        Xs_ids = np.load('data/Xs_idx.npy').item() # A question = a list of word_index
        Ys_ids = np.load('data/Ys_idx.npy').item()
        Xa_ids = np.load('data/Xa_idx.npy').item()
        Ya_ids = np.load('data/Ya_idx.npy').item()

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


    def bow2array(self, bow_sent):
        res = np.zeros(self.vocab_size)
        for word, freq in bow_sent:
            res[word]=freq
        return res

    
    def fetch_data_lsi(self, inputs, outputs, idx_batch): # works for lsi --> lsi
      """fetch input data by batch."""
      batch_size = len(idx_batch)

      input_batch = np.zeros([batch_size,len(inputs[0])])
      output_batch = np.zeros([batch_size,len(outputs[0])])

      for i, idx in enumerate(idx_batch):
        input_ = inputs[idx] # sentence input in BOW/TFIDF format [(word_id, freq)], in LSI format [(topic_id, value)]
        output_ = outputs[idx] 
        try:
            input_batch[i] = input_ # dim vocab_size if BOW or TFIDF, num_topics if LSI
            output_batch[i] = output_
        except:
            pass #print('Error batch',i,': input_:',input_.shape,': output_:',output_.shape)
      return input_batch, output_batch


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
    #data_loader.load_LSI()

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

