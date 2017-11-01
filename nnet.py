import tensorflow as tf
import numpy as np
from tensorflow.contrib import legacy_seq2seq

import utils as utils


# Tensor summaries for TensorBoard visualization
def variable_summaries(name,var, with_max_min=False):
  with tf.name_scope(name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    if with_max_min == True:
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))



class Model(object):

    def __init__(self, embedding_weights, config):

        # Data config
        self.batch_size = config.batch_size
        self.padlen = config.padlen # sentences length
        self.vocab_size = embedding_weights.shape[0] # number of tokens in dictionary
        self.word_embeddings = tf.Variable(embedding_weights, name="word_embeddings", dtype=tf.float32, trainable=True)

        # Network config
        self.hidden_size = config.hidden_size # hidden dimension 
        self.initializer = tf.contrib.layers.xavier_initializer() # variables initializer

        # Training config
        self.global_step = tf.Variable(0, trainable=False, name="global_step") # global step
        self.n_sample = config.n_sample # MC approx
        self.lr_start = config.lr_start # initial learning rate
        self.lr_decay_rate=config.lr_decay_rate # learning rate decay rate
        self.lr_decay_step= config.lr_decay_step # learning rate decay step
        self.is_training = config.is_training

        ### Placeholders
        self.q1 = tf.placeholder(tf.int32, shape=[None, self.padlen], name="question1")
        self.len1 = tf.placeholder(tf.int32, shape=[None], name="question1_length")
        self.q2 = tf.placeholder(tf.int32, shape=[None, self.padlen], name="question2")
        self.len2 = tf.placeholder(tf.int32, shape=[None], name="question2_length")
        #y = tf.placeholder(tf.int64, shape=[None,2], name="is_duplicate")

        self.autodecode()
        self.build_optim()
        self.merged = tf.summary.merge_all()

    def autodecode(self):

        ### Embedding layer
        with tf.variable_scope("word_embeddings"):
            # Embedding
            q1_embed = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.q1, name="q1_embedded")
            # Batch Normalization
            q1_bn = tf.layers.batch_normalization(q1_embed, axis=2, training=self.is_training, name='layer_norm', reuse=None)
                
        ### Encoder layer
        with tf.variable_scope("encoder"):
            # LSTM cell
            cell_fw = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=self.initializer)
            encoder_output, encoder_state = tf.nn.dynamic_rnn(cell_fw, q1_bn, sequence_length=self.len1, dtype=tf.float32)
            encoded_state = encoder_state[0] # last state, output tuple
            # tf.contrib.layers.flatten(lstm1)

        ### Variational inference
        with tf.variable_scope("variational_inference"):
            
            mean = utils.linear(encoded_state, self.hidden_size, scope='mean') # [batch size, n_hidden]
            logsigm = utils.linear(encoded_state, self.hidden_size, scope='logsigm')
            self.kld = -0.5 * tf.reduce_sum(1 - tf.square(mean) + 2 * logsigm - tf.exp(2 * logsigm), 1) # [batch size]
            variable_summaries('kld',self.kld)
            self.kld = tf.reduce_mean(self.kld)

            if self.n_sample ==1:  # single sample
                eps = tf.random_normal((self.batch_size, self.hidden_size), 0, 1)
                doc_vec = tf.multiply(tf.exp(logsigm), eps) + mean  # this is the latent intent
                doc_vec = doc_vec, encoder_state[1] # tuple state

            else:
                eps = tf.random_normal((self.n_sample*self.batch_size, self.hidden_size), 0, 1)
                eps_list = tf.split(eps, axis=0, num_or_size_splits=self.n_sample)
                doc_vec_list = []
                for i in range(self.n_sample):
                    curr_eps = eps_list[i]
                    curr_doc = tf.multiply(tf.exp(logsigm), curr_eps) + mean
                    doc_vec_list.append((curr_doc, encoder_state[1]))


        ### Decoder layer (train)
        with tf.variable_scope("decoder"):
            # Embedding
            q2_embed = tf.nn.embedding_lookup(params=self.word_embeddings, ids=self.q2[:,:-1], name="q2_embedded") # [batch size, pad length-1, h]
            # Batch Normalization
            q2_bn = tf.layers.batch_normalization(q2_embed, axis=2, training=self.is_training, name='layer_norm', reuse=None)
            q2_bn = tf.unstack(q2_bn, axis=1) # List [batch size, h] of size pad_length-1
            # LSTM cell
            cell_fw2 = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=self.initializer)

            if self.n_sample ==1:  # single sample
                decoder_output, _ = legacy_seq2seq.rnn_decoder(decoder_inputs=q2_bn, initial_state=doc_vec, cell=cell_fw2) ###################### USE DYNAMIC RNN DECODER WITH LEN2-1
                decoder_output = tf.stack(decoder_output, axis=1) # [batch_size, pad_length-1, h]

            else:
                decoder_output_list = []
                for i, doc_vec in enumerate(doc_vec_list):
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    decoder_output, _ = legacy_seq2seq.rnn_decoder(decoder_inputs=q2_bn, initial_state=doc_vec, cell=cell_fw2) ###################### USE DYNAMIC RNN DECODER WITH LEN2-1
                    decoder_output_list.append(tf.stack(decoder_output, axis=1))

        ### Projection layer
        with tf.variable_scope("linear_projection"):

            W_proj =tf.get_variable("weights",[1,self.hidden_size, self.vocab_size], initializer=self.initializer)

            if self.n_sample ==1:  # single sample
                # project
                logits = tf.nn.conv1d(decoder_output, W_proj, 1, "VALID", name="logits")  # [batch_size, pad_length-1, vocab_size]

            else:
                logits_list = []
                for decoder_output in decoder_output_list:
                    logits = tf.nn.conv1d(decoder_output, W_proj, 1, "VALID", name="logits")
                    logits_list.append(logits)

        ### Loss
        with tf.variable_scope('loss'):

            if self.n_sample ==1:  # single sample
                #q2 = tf.one_hot(q2,weights.shape[0], axis=-1) # [batch size, pad length, vocab size]
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.q2[:,1:], logits=logits) # predict word i+1 given i, <i
                losses_mask = tf.sequence_mask(lengths=self.len2-1, maxlen=self.padlen-1, dtype=tf.float32)
                self.loss = tf.reduce_sum(losses * losses_mask) / tf.reduce_sum(losses_mask)
                tf.summary.scalar('loss',self.loss)

            else:
                loss_list = []
                for logits in logits_list:
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.q2[:,1:], logits=logits) # predict word i+1 given i, <i
                    losses_mask = tf.sequence_mask(lengths=self.len2-1, maxlen=self.padlen-1, dtype=tf.float32)
                    loss = tf.reduce_sum(losses * losses_mask) / tf.reduce_sum(losses_mask)
                    loss_list.append(loss)
                self.loss = tf.add_n(loss_list) / self.n_sample
                tf.summary.scalar('loss',self.loss)


    def build_optim(self):
        # Minimize op
        lr = tf.train.exponential_decay(self.lr_start, self.global_step, self.lr_decay_rate, self.lr_decay_step, staircase=True, name="learning_rate1")
        # Optimizer
        opt = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.9,beta2=0.99, epsilon=0.0000001)

        fullvars = tf.trainable_variables()
        enc_vars = utils.variable_parser(fullvars, ['word_embeddings', 'encoder', 'variational_inference'])
        print('\n Encoder variables',[v.name for v in enc_vars])
        dec_vars = utils.variable_parser(fullvars, ['decoder', 'linear_projection'])
        print('\n Decoder variables',[v.name for v in dec_vars])

        enc_grads_and_vars = opt.compute_gradients(self.loss, var_list=enc_vars)
        dec_grads_and_vars = opt.compute_gradients(self.loss, var_list=dec_vars)

        self.optim_enc = opt.apply_gradients(enc_grads_and_vars)
        self.optim_dec = opt.apply_gradients(dec_grads_and_vars)