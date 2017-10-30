"""NVDM Tensorflow implementation by Yishu Miao"""
from __future__ import print_function

import numpy as np
import tensorflow as tf
import math
import os
import utils as utils

from data_loader import Data_loader

np.random.seed(0)
tf.set_random_seed(0)



class NVDM(object):
    """ Neural Variational Document Model -- BOW VAE.
    """
    def __init__(self, vocab_size=200, n_hidden=500, n_topic=50, n_sample=1, learning_rate=5e-5, batch_size=64, non_linearity=tf.nn.tanh, test=False):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.x = tf.placeholder(tf.float32, [batch_size, vocab_size], name='input')
        self.y = tf.placeholder(tf.float32, [batch_size, vocab_size], name='output')

        # encoder
        with tf.variable_scope('encoder'): 
          self.enc_vec = utils.mlp(self.x, [self.n_hidden], self.non_linearity)
          self.mean = utils.linear(self.enc_vec, self.n_topic, scope='mean')
          self.logsigm = utils.linear(self.enc_vec, self.n_topic, bias_start_zero=True, matrix_start_zero=True, scope='logsigm')
          self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.logsigm - tf.exp(2 * self.logsigm), 1)
        
        with tf.variable_scope('decoder'):
          if self.n_sample ==1:  # single sample
            eps = tf.random_normal((batch_size, self.n_topic), 0, 1)
            doc_vec = tf.multiply(tf.exp(self.logsigm), eps) + self.mean
            #logits = tf.nn.log_softmax(utils.linear(doc_vec, self.vocab_size, scope='projection')) ############ 
            logits = tf.nn.tanh(utils.linear(doc_vec, self.vocab_size, scope='projection')) ############ 
            self.y = tf.nn.tanh(self.y)
            #self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.y), 1) ############ self.x to self.y
            self.recons_loss = tf.reduce_mean(tf.square(logits-self.y),1) #-tf.reduce_sum(tf.multiply(logits, self.y), 1) ############ self.x to self.y
          # multiple samples
          else:
            eps = tf.random_normal((self.n_sample*batch_size, self.n_topic), 0, 1)
            eps_list = tf.split(0, self.n_sample, eps)
            recons_loss_list = []
            for i in xrange(self.n_sample):
              if i > 0: tf.get_variable_scope().reuse_variables()
              curr_eps = eps_list[i]
              doc_vec = tf.matmul(tf.exp(self.logsigm), curr_eps) + self.mean
              logits = tf.nn.log_softmax(utils.linear(doc_vec, self.vocab_size, scope='projection'))
              recons_loss_list.append(-tf.reduce_sum(tf.matmul(logits, self.x), 1))
            self.recons_loss = tf.add_n(recons_loss_list) / self.n_sample

        self.objective = self.recons_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        fullvars = tf.trainable_variables()

        enc_vars = utils.variable_parser(fullvars, 'encoder')
        dec_vars = utils.variable_parser(fullvars, 'decoder')

        enc_grads = tf.gradients(self.objective, enc_vars)
        dec_grads = tf.gradients(self.objective, dec_vars)

        self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
        self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))

def train(sess, model, data_loader, training_epochs=1000,  alternate_epochs=10, test=False):

  Xs_lsi, Ys_lsi, Xa_lsi,Ya_lsi = data_loader.load_LSI()
  
  for epoch in range(training_epochs):
    batches = data_loader.create_batches(len(Xs_lsi)) ##########################
    #-------------------------------
    # train
    for switch in range(0, 2):
      if switch == 0:
        optim = model.optim_dec
        print_mode = 'updating decoder'
      else:
        optim = model.optim_enc
        print_mode = 'updating encoder'
      for i in range(alternate_epochs):
        loss_sum = 0.0
        kld_sum = 0.0

        for idx_batch in batches:
          # fetch duplicate questions
          input_batch, output_batch = data_loader.fetch_data(Xs_lsi, Ys_lsi, idx_batch) ##########################

          # choose i and f iid in {0,1}. random task: qi -> qf or qf -> qi.
          r1, r2 = np.random.rand(1), np.random.rand(1) # two iid rrv
          if r1 <0.5: # reformulate
            if r2<0.5:
              input_feed = {model.x.name: input_batch, model.y.name: output_batch}
            else:
              input_feed = {model.x.name: output_batch, model.y.name: input_batch}
          else: # repeat
            if r2<0.5:
              input_feed = {model.x.name: input_batch, model.y.name: input_batch}
            else:
              input_feed = {model.x.name: output_batch, model.y.name: output_batch}

          # auto encode
          _, (loss, kld) = sess.run((optim, [model.objective, model.kld]), input_feed)
          loss_sum += np.sum(loss)
          kld_sum += np.sum(kld)
        print('| Epoch train: {:d} |'.format(epoch+1), print_mode, '{:d}'.format(i),
               '| Loss doc ppx: {:.5f}'.format(np.mean(loss_sum)),
               '| KLD: {:.5}'.format(np.mean(kld_sum)))
    #-------------------------------
    # dev
    '''
    loss_sum = 0.0
    kld_sum = 0.0
    ppx_sum = 0.0
    word_count = 0
    doc_count = 0
    for idx_batch in dev_batches:
      data_batch, count_batch, mask = utils.fetch_data(dev_set, dev_count, idx_batch)
      input_feed = {model.x.name: data_batch, model.mask.name: mask}
      loss, kld = sess.run([model.objective, model.kld],
                           input_feed)
      loss_sum += np.sum(loss)
      kld_sum += np.sum(kld) / np.sum(mask)  
      word_count += np.sum(count_batch)
      count_batch = np.add(count_batch, 1e-12)
      ppx_sum += np.sum(np.divide(loss, count_batch))
      doc_count += np.sum(mask) 
    print_ppx = np.exp(loss_sum / word_count)
    print_ppx_perdoc = np.exp(ppx_sum / doc_count)
    print_kld = kld_sum/len(dev_batches)
    print('| Epoch dev: {:d} |'.format(epoch+1), 
           '| Perplexity: {:.9f}'.format(print_ppx),
           '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
           '| KLD: {:.5}'.format(print_kld))
    '''  
    #-------------------------------
    # test
    if test:
      loss_sum = 0.0
      kld_sum = 0.0
      ppx_sum = 0.0
      word_count = 0
      doc_count = 0
      for idx_batch in test_batches:
        data_batch, count_batch, mask = utils.fetch_data(test_set, test_count, idx_batch)
        input_feed = {model.x.name: data_batch, model.mask.name: mask}
        loss, kld = sess.run([model.objective, model.kld],
                             input_feed)
        loss_sum += np.sum(loss)
        kld_sum += np.sum(kld)/np.sum(mask) 
        word_count += np.sum(count_batch)
        count_batch = np.add(count_batch, 1e-12)
        ppx_sum += np.sum(np.divide(loss, count_batch))
        doc_count += np.sum(mask) 
      print_ppx = np.exp(loss_sum / word_count)
      print_ppx_perdoc = np.exp(ppx_sum / doc_count)
      print_kld = kld_sum/len(test_batches)
      print('| Epoch test: {:d} |'.format(epoch+1), 
             '| Perplexity: {:.9f}'.format(print_ppx),
             '| Per doc ppx: {:.5f}'.format(print_ppx_perdoc),
             '| KLD: {:.5}'.format(print_kld))   


def main(argv=None):

    nvdm = NVDM()
    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    training_set = Data_loader()
    train(sess, nvdm, training_set)

  
if __name__ == '__main__':
    tf.app.run()